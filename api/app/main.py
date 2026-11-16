"""显微成像原片单实例 API。

安全边界：
  * 明文原片只在请求/响应的内存流中存在，密文 blob 直接落盘；
  * 明文 DEK 只在单次请求内存中存活，元数据库只存信封密文；
  * 主密钥材料永远只在密钥服务内，本服务通过内部 HTTP 调用包裹/解包。
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

BLOB_DIR = os.environ.get("BLOB_DIR", "/data/blobs")
DB_PATH = os.environ.get("METADATA_DB", "/data/metadata.db")
KEY_SERVICE_URL = os.environ.get("KEY_SERVICE_URL", "http://keystore:8001")
KEY_SERVICE_TOKEN = os.environ.get("KEY_SERVICE_TOKEN", "dev-only-key-service-token")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 ** 3)))

# 安全约束：Starlette 的 multipart 解析器默认把 >1MiB 的表单文件假脱机到
# 磁盘临时文件，等于让明文原片落盘。将假脱机阈值提高到上传上限，使允许范围
# 内的 multipart 一律留在内存（超限请求在入口处即被 Content-Length 拒绝）。
# 首选仍是原始 application/octet-stream 流式上传，完全不经表单解析器。
from starlette.formparsers import MultiPartParser
MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES

from .crypto_stream import IntegrityError, decrypt_stream, encrypt_stream
from .keyclient import KeyClient, KeyServiceError
from .metadata import IdempotencyConflict, MetadataDB
from .rotation import RotationWorker

db = MetadataDB(DB_PATH)
keys = KeyClient(KEY_SERVICE_URL, KEY_SERVICE_TOKEN)
worker: RotationWorker | None = None
main_loop: asyncio.AbstractEventLoop | None = None

# 在途幂等令牌注册表：单实例内，同一令牌并发上传时第二个请求立即得到 409，
# 不生成第二件原片/第二份密文。事件循环单线程，set 的检查与插入之间无 await，
# 原子成立；DB 的 UNIQUE(idem_token) 约束作为第二道防线。
_inflight_tokens: set[str] = set()


# --------------------------------------------------------------------------
# 生命周期：引导主密钥 -> 崩溃恢复（在途上传判失败/轮换接续）-> 启动后台线程
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker, main_loop
    os.makedirs(BLOB_DIR, exist_ok=True)
    main_loop = asyncio.get_running_loop()
    # 等待密钥服务就绪并确保存在当前主密钥（重启幂等）
    for attempt in range(10):
        try:
            await keys.ensure_bootstrap()
            break
        except KeyServiceError:
            await asyncio.sleep(1.0)
    # 崩溃恢复：上次 receiving 的上传一律失败，禁止"晚发布"依赖旧密钥
    stale = db.mark_stale_receiving_failed()
    for row in stale:
        _safe_remove_blob(row["blob_path"])
    if stale:
        print(f"[startup] {len(stale)} interrupted upload(s) marked failed")
    # 密文残片对账：删除元数据库不再登记的无主 .msef（崩溃/清理失败的兜底）
    removed = _reap_orphan_blobs()
    if removed:
        print(f"[startup] removed {removed} orphaned ciphertext blob(s)")
    # 崩溃对账：上次可能已在密钥服务切换了 MK、却没来得及建本地作业
    try:
        key_status = await keys.master_status()
        current_mk = key_status.get("current_key_id")
        for key in key_status.get("keys", []):
            if (key["key_id"] != current_mk
                    and key["state"] == "active"
                    and not db.has_rotation_for(key["key_id"])):
                job_id = db.create_rotation(
                    key["key_id"], current_mk,
                    db.count_under_key(key["key_id"], only_migratable=True))
                print(f"[startup] reconciled missing rotation job {job_id} "
                      f"for {key['key_id']}")
    except KeyServiceError as exc:
        print(f"[startup] rotation reconciliation skipped: {exc}")
    worker = RotationWorker(db, keys, main_loop)
    worker.start()
    yield
    worker.stop()
    await keys.aclose()
    db.close()


app = FastAPI(
    title="Microscopy Image Warehouse API",
    version="1.0.0",
    description="原片流式加密存取，每件独立 DEK；支持主密钥在线轮换与进度查询。",
    lifespan=lifespan,
)


def _key_error(exc: KeyServiceError) -> HTTPException:
    return HTTPException(status_code=exc.status_code,
                         detail={"code": exc.code, "message": exc.message})


def _safe_remove_blob(blob_path: str) -> bool:
    """尽力删除密文残片；文件不存在视为已清理，不抛异常。"""
    try:
        os.remove(blob_path)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        print(f"[warn] could not remove blob {blob_path}: {exc!r}")
        return False


def _reap_orphan_blobs() -> int:
    """删除 BLOB_DIR 下未被元数据库登记的 .msef 残片（启动对账兜底）。"""
    try:
        names = os.listdir(BLOB_DIR)
    except FileNotFoundError:
        return 0
    known = {os.path.basename(p) for p in db.known_blob_paths()}
    removed = 0
    for name in names:
        if not name.endswith(".msef") or name in known:
            continue
        if _safe_remove_blob(os.path.join(BLOB_DIR, name)):
            removed += 1
    return removed


# --------------------------------------------------------------------------
# 上传
# --------------------------------------------------------------------------
async def _stream_to_blob(request: Request, blob_path: str,
                          dek: bytes) -> dict[str, Any]:
    """读 HTTP 请求体 -> 分帧加密 -> 密文直接写盘（无明文临时文件）。"""
    writer = open(blob_path, "wb")  # 二进制密文文件
    loop = asyncio.get_running_loop()

    async def write(data: bytes):
        await loop.run_in_executor(None, writer.write, data)
        await loop.run_in_executor(None, writer.flush)

    try:
        stats = await encrypt_stream(request.stream(), dek, write)
    finally:
        await loop.run_in_executor(None, writer.close)
    return stats


async def _finalize_with_rebind(image_id: str, stats: dict[str, Any]) -> dict[str, Any]:
    """最终化原片；若其 MK 在上传期间已被轮换切换，则先 rewrap 改绑当前 MK。

    是否需要改绑只依据本地轮换记录判断（不额外依赖密钥服务在线）：
    该行绑定的正是最近一次轮换的旧 MK 时，说明它是跨越轮换的在途上传。
    防止在途上传晚发布依赖旧（甚至已销毁）主密钥的原片。
    """
    meta = db.get_image(image_id)
    if meta is None:
        raise HTTPException(
            status_code=409,
            detail={"code": "upload_aborted",
                    "message": "upload metadata row no longer exists"})
    job = db.latest_rotation()
    if job and meta["master_key_id"] == job["old_key_id"]:
        try:
            res = await keys.rewrap(meta["master_key_id"], meta["wrapped_dek"],
                                    f"finalize:{image_id}")
        except KeyServiceError as exc:
            if exc.status_code == 410:
                # 旧 MK 已销毁且该行信封未能提前迁移：绝不能晚发布不可读原片
                raise HTTPException(
                    status_code=409,
                    detail={"code": "old_key_destroyed",
                            "message": "upload cannot be finalized; its key was "
                                       "rotated and destroyed before completion"})
            raise
        db.rewrap_row(image_id, meta["master_key_id"],
                      res["key_id"], res["wrapped_dek"])
    # 发布前核对落盘密文：ready 回执必须对应真实可读、尺寸一致的密文原片
    try:
        actual_enc_size = os.path.getsize(meta["blob_path"])
    except OSError as exc:
        raise HTTPException(status_code=502,
                            detail=f"ciphertext blob unavailable: {exc!r}") from exc
    if actual_enc_size != stats["enc_size"]:
        raise HTTPException(
            status_code=502,
            detail=f"ciphertext size mismatch: disk={actual_enc_size} "
                   f"expected={stats['enc_size']}")
    published = db.finalize_image(image_id, stats["plain_size"], stats["enc_size"],
                                  stats["sha256"])
    if not published:
        # 行已被并发失败/重启恢复判失败：绝不能晚发布
        raise HTTPException(
            status_code=409,
            detail={"code": "upload_aborted",
                    "message": "upload was aborted before finalization"})
    return db.get_image(image_id)


def _check_content_length(request: Request):
    cl = request.headers.get("content-length")
    if cl is not None and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail="upload exceeds MAX_UPLOAD_BYTES")


# --------------------------------------------------------------------------
# 上传（POST /images 与 PUT /images/{image_id} 共用同一条幂等入仓管线）
# --------------------------------------------------------------------------
def _idempotency_conflict(holder: dict[str, Any]) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={"code": "idempotency_conflict",
                "message": "another upload using the same X-Idempotency-Token "
                           "is in progress",
                "image_id": holder.get("image_id"),
                "status": holder.get("status")})


def _replay_receipt(existing: dict[str, Any], response: Response) -> dict[str, Any]:
    """完成后的幂等重放：回已发布原片的编号与回执。

    只读取既有行，绝不重新入湖，因此正文、SHA-256 摘要与密钥绑定均不变。
    """
    response.status_code = 200
    return {"image_id": existing["image_id"], "status": existing["status"],
            "idempotent": True, "size": existing["plain_size"],
            "sha256": existing["sha256"],
            "master_key_id": existing["master_key_id"]}


def _abort_upload(image_id: str, blob_path: str, error: str) -> None:
    """失败收口：行判 failed 并释放幂等令牌，密文残片同步删除。

    保证失败请求不留无主密文，且原令牌/失败编号可立即用于重传。
    """
    db.fail_image(image_id, error)
    _safe_remove_blob(blob_path)


async def _ingest(request: Request, response: Response, image_id: str,
                  idem_token: str | None, filename: str | None,
                  content_type: str | None, named: bool) -> dict[str, Any]:
    _check_content_length(request)

    # 1) 权威判断以元数据库为准（进程重启后内存注册表清空也不受影响）
    if idem_token:
        existing = db.get_by_token(idem_token)
        if existing is not None:
            if existing["status"] == "ready":
                # 完成后重放：返回已发布原片，不改正文/摘要/密钥绑定
                return _replay_receipt(existing, response)
            # receiving：同令牌并发，另一笔仍在途 -> 明确冲突
            # failed 行的令牌在判失败时已释放，查不到；兜底同样按冲突处理
            raise _idempotency_conflict(existing)
        # 2) 进程内快速拦截同令牌并发（DB 的 UNIQUE 约束为第二道防线）。
        #    检查与插入之间无 await，事件循环单线程 -> 原子。
        if idem_token in _inflight_tokens:
            raise HTTPException(
                status_code=409,
                detail={"code": "idempotency_in_flight",
                        "message": "an upload with this X-Idempotency-Token is "
                                   "already in progress"})
        _inflight_tokens.add(idem_token)
    try:
        return await _run_ingest(request, image_id, idem_token, filename,
                                 content_type, named)
    finally:
        if idem_token is not None:
            _inflight_tokens.discard(idem_token)


async def _run_ingest(request: Request, image_id: str,
                      idem_token: str | None, filename: str | None,
                      content_type: str | None, named: bool) -> dict[str, Any]:
    blob_path = os.path.join(BLOB_DIR, image_id + ".msef")

    if named:
        current = db.get_image(image_id)
        if current is not None:
            if current["status"] in ("receiving", "ready"):
                raise HTTPException(
                    status_code=409,
                    detail={"code": "image_id_exists",
                            "message": f"image_id already exists "
                                       f"({current['status']})",
                            "image_id": image_id})
            # failed：允许按号复用失败编号，删除旧失败行并清理其残片
            old = db.delete_failed_image(image_id)
            if old is not None:
                _safe_remove_blob(old["blob_path"])

    try:
        dek_resp = await keys.generate_dek()
    except KeyServiceError as exc:
        raise _key_error(exc)
    master_key_id = dek_resp["key_id"]
    wrapped_dek = dek_resp["wrapped_dek"]
    dek = bytes.fromhex(dek_resp["dek"])  # 仅存于本请求内存

    try:
        try:
            db.create_image(
                image_id, filename,
                content_type or request.headers.get("content-type"),
                master_key_id, wrapped_dek, blob_path, idem_token)
        except IdempotencyConflict as exc:
            # 并发穿透的兜底（前面已有两层拦截）：明确 409，不产生密文残片
            raise _idempotency_conflict(exc.holder)
        try:
            stats = await _stream_to_blob(request, blob_path, dek)
            if stats["plain_size"] > MAX_UPLOAD_BYTES:
                raise HTTPException(413, detail="upload exceeds MAX_UPLOAD_BYTES")
            final = await _finalize_with_rebind(image_id, stats)
        except HTTPException:
            _abort_upload(image_id, blob_path, "upload rejected")
            raise
        except Exception as exc:  # 写入失败：判失败并清理密文残片
            _abort_upload(image_id, blob_path, f"upload failed: {exc!r}")
            raise HTTPException(status_code=502,
                                detail=f"upload failed: {exc!r}") from exc
    finally:
        del dek  # 尽快丢弃明文 DEK 引用
    return {"image_id": image_id, "status": "ready",
            "size": stats["plain_size"], "sha256": stats["sha256"],
            "master_key_id": final["master_key_id"]}


@app.post("/images", status_code=201,
          summary="上传原片（流式加密；支持 X-Idempotency-Token 幂等）")
async def upload_image(request: Request, response: Response,
                       x_idempotency_token: str | None = Header(default=None),
                       x_filename: str | None = Header(default=None),
                       x_content_type: str | None = Header(default=None)):
    image_id = "img-" + uuid.uuid4().hex
    return await _ingest(request, response, image_id, x_idempotency_token,
                         x_filename, x_content_type, named=False)


@app.put("/images/{image_id}", status_code=201,
         summary="按调用方指定编号上传原片（失败编号可复用；支持幂等令牌）")
async def upload_image_named(image_id: str, request: Request, response: Response,
                             x_idempotency_token: str | None = Header(default=None),
                             x_filename: str | None = Header(default=None),
                             x_content_type: str | None = Header(default=None)):
    if not image_id or len(image_id) > 128 or "/" in image_id or ".." in image_id:
        raise HTTPException(400, detail="invalid image_id")
    return await _ingest(request, response, image_id, x_idempotency_token,
                         x_filename, x_content_type, named=True)


# --------------------------------------------------------------------------
# 下载 / 查询
# --------------------------------------------------------------------------
def _cipher_chunks(path: str):
    async def gen():
        loop = asyncio.get_running_loop()
        f = await loop.run_in_executor(None, open, path, "rb")
        try:
            while True:
                chunk = await loop.run_in_executor(None, f.read, 1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            await loop.run_in_executor(None, f.close)
    return gen()


@app.get("/images/{image_id}", summary="按编号下载原片（流式解密，逐帧验签）")
async def download_image(image_id: str):
    meta = db.get_image(image_id)
    if meta is None:
        raise HTTPException(404, detail="image not found")
    if meta["status"] != "ready":
        raise HTTPException(409, detail=f"image is {meta['status']}")
    if not os.path.exists(meta["blob_path"]):
        raise HTTPException(410, detail="encrypted blob missing on storage")
    try:
        # 解包一次：明文 DEK 只在本次请求内存；随后即使密钥服务变慢或
        # 主密钥被销毁，已解包的慢下载也能完整读完（解密不再依赖密钥服务）。
        dek = await keys.unwrap(meta["master_key_id"], meta["wrapped_dek"])
    except KeyServiceError as exc:
        raise _key_error(exc)

    async def plain_iter():
        local_dek = dek  # 局部别名；生成器结束后随帧栈释放，不留长生命周期引用
        try:
            async for chunk in decrypt_stream(
                    _cipher_chunks(meta["blob_path"]), local_dek,
                    expected_sha256=meta["sha256"]):
                yield chunk
        except IntegrityError as exc:
            raise HTTPException(500, detail=f"integrity check failed: {exc}")
        finally:
            del local_dek

    headers = {"X-Image-Id": image_id,
               "X-SHA256": meta["sha256"] or ""}
    if meta["filename"]:
        headers["Content-Disposition"] = (
            f'attachment; filename="{meta["filename"]}"')
    return StreamingResponse(
        plain_iter(),
        media_type=meta["content_type"] or "application/octet-stream",
        headers=headers)


@app.get("/images", summary="列出原片元数据")
async def list_images(limit: int = 100, offset: int = 0):
    return {"items": db.list_images(min(limit, 500), offset)}


@app.get("/images/{image_id}/meta", summary="单件原片元数据/状态")
async def image_meta(image_id: str):
    meta = db.get_image(image_id)
    if meta is None:
        raise HTTPException(404, detail="image not found")
    meta.pop("wrapped_dek", None)
    return meta


# --------------------------------------------------------------------------
# 轮换
# --------------------------------------------------------------------------
class RotationRequest(BaseModel):
    confirm: bool = True


@app.post("/rotations", status_code=202,
          summary="发起主密钥轮换（迁移信封后销毁旧 MK；未完拒绝新轮换）")
async def create_rotation(_: RotationRequest):
    active = db.active_rotation()
    if active is not None:
        raise HTTPException(
            409, detail={"code": "rotation_in_progress",
                         "job_id": active["job_id"], "state": active["state"]})
    try:
        rot = await keys.rotate_master()  # 新入湖立即改用新 MK
        old_key, new_key = rot["old_key_id"], rot["new_key_id"]
        total = (db.count_under_key(old_key, only_migratable=True)
                 if old_key else 0)
        job_id = db.create_rotation(old_key, new_key, total)
    except KeyServiceError as exc:
        raise _key_error(exc)
    return {"job_id": job_id, "state": "migrating", "old_key_id": old_key,
            "new_key_id": new_key, "total_to_migrate": total}


@app.get("/rotations/latest", summary="查询最近一次轮换进度")
async def latest_rotation():
    job = db.latest_rotation()
    if job is None:
        return {"message": "no rotation has ever run"}
    return _rotation_view(job)


@app.get("/rotations/{job_id}", summary="按作业号查询轮换进度")
async def rotation_status(job_id: str):
    job = db.get_rotation(job_id)
    if job is None:
        raise HTTPException(404, detail="rotation job not found")
    return _rotation_view(job)


def _rotation_view(job: dict) -> dict:
    total = max(job["total"], job["migrated"] + job["failed"], 1)
    pct = round(100.0 * (job["migrated"] + job["failed"]) / total, 2)
    return {**{k: v for k, v in job.items()}, "progress_percent": pct}


@app.get("/keys/status", summary="密钥视角状态（代理密钥服务只读接口）")
async def key_status():
    try:
        return await keys.master_status()
    except KeyServiceError as exc:
        raise _key_error(exc)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "image-api", "time": time.time()}
