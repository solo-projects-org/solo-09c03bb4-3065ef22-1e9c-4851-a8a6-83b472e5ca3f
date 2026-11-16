"""幂等入仓修复的端到端验证（真实 uvicorn 子进程）。

覆盖缺陷单与需求：
  C1 同令牌并发 POST：至多一件 ready，其余明确 409（不得 500、不得无主密文）
  C2 同编号并发 PUT（无令牌）：至多一件 ready，其余 409
  C3 完成后重放：回已发布原片编号；正文/摘要/密钥绑定不变；不新增密文
  C4 失败（客户端断线）后用原令牌重传成功；失败不留无主密文
  C5 重启中断后用原令牌 + 原编号重传成功；重启即清理残片并释放令牌
  C6 按号上传可复用失败编号
  C7 所有 ready 回执对应逐字节可读、摘要一致的原片；无孤儿密文
  C8 超限 413 同样释放令牌、不留密文，可凭原令牌重试
"""
from __future__ import annotations

import os
import socket
import sqlite3
import threading
import time

import httpx

from conftest import _download, _upload

OCTET = "application/octet-stream"


def _blob_files(stack) -> list:
    return [p for p in stack["blob_dir"].iterdir() if p.is_file()]


def _wait_status(stack, image_id, status, timeout=15.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = stack["client"].get(f"/images/{image_id}/meta")
        if r.status_code == 200:
            last = r.json()
            if last["status"] == status:
                return last
        time.sleep(0.2)
    raise AssertionError(f"{image_id} did not reach {status}; last={last}")


def _abrupt_disconnect(stack, image_id: str, token: str | None,
                       partial: bytes) -> None:
    """发送只写了一半的请求体后直接断开 TCP，模拟仪器断线。"""
    host, port = "127.0.0.1", stack["api"].port
    headers = (
        f"PUT /images/{image_id} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: {OCTET}\r\n"
        f"Content-Length: {len(partial) + 4096}\r\n"
        f"Connection: close\r\n"
    )
    if token:
        headers += f"X-Idempotency-Token: {token}\r\n"
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(headers.encode() + b"\r\n" + partial)
        sock.shutdown(socket.SHUT_RDWR)


def test_c1_same_token_concurrent_post_single_ready_explicit_conflict(stack):
    client = stack["client"]
    token = "concurrent-post-token"
    payload = os.urandom(2 * 1024 * 1024)  # 足够大，制造同窗口并发
    n = 4
    barrier = threading.Barrier(n)
    results: list[httpx.Response] = []
    errors: list[Exception] = []

    def worker():
        try:
            barrier.wait()
            r = client.post(
                "/images", content=payload,
                headers={"Content-Type": OCTET,
                         "X-Idempotency-Token": token})
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(results) == n
    # 绝不允许 500
    assert all(r.status_code < 500 for r in results), [r.text for r in results]
    successes = [r for r in results if r.status_code in (200, 201)]
    conflicts = [r for r in results if r.status_code == 409]
    assert len(successes) == 1, [(r.status_code, r.text) for r in results]
    assert len(conflicts) == n - 1
    codes = {r.json()["detail"]["code"] for r in conflicts}
    assert codes <= {"idempotency_in_flight", "idempotency_conflict"}
    # 至多一件原片发布、一份密文
    ready = [i for i in client.get("/images").json()["items"]
             if i["status"] == "ready"]
    assert len(ready) == 1
    assert len(_blob_files(stack)) == 1
    image_id = successes[0].json()["image_id"]
    assert _download(client, image_id) == payload


def test_c2_same_id_concurrent_put_single_ready(stack):
    client = stack["client"]
    n = 4
    barrier = threading.Barrier(n)
    results: list[httpx.Response] = []
    payload = os.urandom(1024 * 1024)

    def worker():
        barrier.wait()
        results.append(_upload(client, payload, image_id="race-put-id"))

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r.status_code < 500 for r in results), [r.text for r in results]
    assert sorted(r.status_code for r in results).count(201) == 1
    assert sorted(r.status_code for r in results).count(409) == n - 1
    assert len(_blob_files(stack)) == 1
    assert _download(client, "race-put-id") == payload


def test_c3_replay_returns_published_receipt_unchanged(stack):
    client = stack["client"]
    token = "replay-token-1"
    first_body = b"first-body-wins" * 100
    h = {"Content-Type": OCTET, "X-Idempotency-Token": token}
    r1 = client.post("/images", content=first_body, headers=h)
    assert r1.status_code == 201
    receipt1 = r1.json()

    # 不同正文重放：必须被吸收，不改变任何东西
    r2 = client.post("/images", content=b"totally-different-body", headers=h)
    assert r2.status_code == 200
    receipt2 = r2.json()
    assert receipt2["idempotent"] is True
    assert receipt2["image_id"] == receipt1["image_id"]
    assert receipt2["sha256"] == receipt1["sha256"]
    assert receipt2["master_key_id"] == receipt1["master_key_id"]
    assert receipt2["size"] == receipt1["size"]

    meta = client.get(f"/images/{receipt1['image_id']}/meta").json()
    assert meta["sha256"] == receipt1["sha256"]
    assert meta["master_key_id"] == receipt1["master_key_id"]
    assert _download(client, receipt1["image_id"]) == first_body
    # 重放不产生第二份密文
    assert len(_blob_files(stack)) == 1


def test_c4_failed_upload_retry_with_same_token_no_orphan(stack):
    client = stack["client"]
    image_id = "disconnect-1"
    token = "retry-after-disconnect"
    _abrupt_disconnect(stack, image_id, token, b"partial-bytes" * 100)

    _wait_status(stack, image_id, "failed")
    # 失败不留无主密文
    assert not _blob_files(stack)

    # 原令牌重传（POST 场景：令牌复活，编号由服务端重新分配）
    body = b"complete-body" * 50
    r = _upload(client, body, token=token)
    assert r.status_code == 201, r.text
    new_id = r.json()["image_id"]
    assert _download(client, new_id) == body
    assert len(_blob_files(stack)) == 1


def test_c5_restart_interrupted_retry_same_token_and_id(stack):
    client = stack["client"]
    image_id = "interrupted-restart-1"
    token = "restart-retry-token"
    # 制造一个带令牌的在途行与残片，等同上传建立后进程被 kill
    conn = sqlite3.connect(stack["data_dir"] / "metadata.db")
    mk = client.get("/keys/status").json()["current_key_id"]
    blob_path = stack["blob_dir"] / f"{image_id}.msef"
    blob_path.write_bytes(b"MSEF\x01" + os.urandom(99))  # 残片
    conn.execute(
        "INSERT INTO images (image_id, status, master_key_id, wrapped_dek, "
        "blob_path, idem_token, created_at) VALUES (?,?,?,?,?,?,?)",
        (image_id, "receiving", mk, "00" * 64, str(blob_path), token, time.time()))
    conn.commit()
    conn.close()

    stack["api"].restart()

    _wait_status(stack, image_id, "failed")
    # 重启已释放令牌并清理残片
    assert not blob_path.exists()
    assert not _blob_files(stack)
    conn = sqlite3.connect(stack["data_dir"] / "metadata.db")
    row = conn.execute(
        "SELECT idem_token FROM images WHERE image_id=?", (image_id,)).fetchone()
    conn.close()
    assert row[0] is None

    # 用原令牌 + 原编号重传成功
    body = b"after-restart-body" * 33
    r = _upload(client, body, image_id=image_id, token=token)
    assert r.status_code == 201, r.text
    assert r.json()["image_id"] == image_id
    assert _download(client, image_id) == body
    assert len(_blob_files(stack)) == 1


def test_c6_named_put_reuses_failed_image_id(stack):
    client = stack["client"]
    image_id = "reuse-failed-id"
    # 先制造一次失败（无令牌）
    _abrupt_disconnect(stack, image_id, None, b"half" * 50)
    _wait_status(stack, image_id, "failed")
    assert not _blob_files(stack)

    body = b"second-attempt-wins" * 20
    r = _upload(client, body, image_id=image_id)
    assert r.status_code == 201, r.text
    assert r.json()["image_id"] == image_id
    assert _download(client, image_id) == body
    assert len(_blob_files(stack)) == 1


def test_c7_every_ready_receipt_is_readable_and_no_orphan_blobs(stack):
    client = stack["client"]
    # 先混入一件失败上传
    _abrupt_disconnect(stack, "failed-mix-1", "orphan-check-token", b"x" * 100)
    _wait_status(stack, "failed-mix-1", "failed")
    # 两件成功上传
    bodies = {f"ready-{i}": os.urandom(2048 + i) for i in range(2)}
    for name, body in bodies.items():
        r = _upload(client, body, image_id=name)
        assert r.status_code == 201

    items = client.get("/images", params={"limit": 500}).json()["items"]
    ready_items = [i for i in items if i["status"] == "ready"]
    # 每件 ready 逐字节可读且摘要一致
    for item in ready_items:
        data = _download(client, item["image_id"])
        assert data  # 非空（本测试上传的均非空）
        import hashlib
        assert hashlib.sha256(data).hexdigest() == item["sha256"]

    # 磁盘密文数 == 元数据中 ready 行数（failed 已清理，无孤儿）
    assert len(_blob_files(stack)) == len(ready_items)
    conn = sqlite3.connect(stack["data_dir"] / "metadata.db")
    db_paths = {r[0] for r in conn.execute(
        "SELECT blob_path FROM images WHERE status='ready'").fetchall()}
    conn.close()
    disk_paths = {str(p) for p in _blob_files(stack)}
    assert disk_paths == db_paths


def test_c8_oversize_releases_token_and_leaves_no_blob(stack):
    client = stack["client"]
    token = "oversize-token"
    # 用裸请求声明超限 Content-Length：服务端在入口处即 413，不读正文
    host, port = "127.0.0.1", stack["api"].port
    req = (
        f"POST /images HTTP/1.1\r\nHost: {host}:{port}\r\n"
        f"Content-Type: {OCTET}\r\n"
        f"X-Idempotency-Token: {token}\r\n"
        f"Content-Length: {21 * 1024 ** 3}\r\nConnection: close\r\n\r\n"
    ).encode()
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(req)
        resp = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            resp += chunk
    assert b"413" in resp.split(b"\r\n", 1)[0], resp[:120]
    assert not _blob_files(stack)
    # 令牌未被占用：可立即用于正常上传
    body = b"normal-after-413"
    r2 = _upload(client, body, token=token)
    assert r2.status_code == 201, r2.text
    assert _download(client, r2.json()["image_id"]) == body
