from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


PART_BYTES = 64 * 1024 * 1024
BLOCK_BYTES = 1024 * 1024
UPLOAD_ID = re.compile(r"[0-9a-f]{48}")
SHA256 = re.compile(r"[0-9a-f]{64}")


class UploadError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def copy_blocks(source: BinaryIO, target: BinaryIO, limit: int, digest) -> int:
    written = 0
    while block := source.read(BLOCK_BYTES):
        written += len(block)
        if written > limit:
            raise UploadError(400, "分片或文件字节数超过声明长度")
        target.write(block)
        digest.update(block)
    return written


class ChunkUploads:
    def __init__(self, root: Path, max_file_size: int, capacity: int, ttl: int, extensions: set[str]):
        self.root = root
        self.max_file_size = max_file_size
        self.capacity = capacity
        self.ttl = ttl
        self.extensions = extensions

    @contextmanager
    def transaction(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(self.root / "index.sqlite3", timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("""CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL, expires REAL NOT NULL
                )""")
                yield connection
        except sqlite3.OperationalError as exc:
            if getattr(exc, "sqlite_errorcode", None) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or str(exc) == "database is locked":
                raise UploadError(503, "上传存储繁忙，请重试") from exc
            raise
        finally:
            connection.close()

    def _expire(self, db, now: float) -> None:
        for row in db.execute("SELECT id FROM uploads WHERE expires <= ?", (now,)).fetchall():
            directory = self.root / row["id"]
            try:
                if directory.exists():
                    shutil.rmtree(directory)
            except PermissionError:
                # An open transcription handle can temporarily prevent removal on Windows.
                continue
            db.execute("DELETE FROM uploads WHERE id = ?", (row["id"],))

    def expire(self) -> None:
        with self.transaction() as db:
            self._expire(db, time.time())

    def _find(self, db, upload_id: str | None):
        if upload_id is None or not UPLOAD_ID.fullmatch(upload_id):
            raise UploadError(400, "无效的 upload_id")
        row = db.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        if row is None or row["expires"] <= time.time():
            raise UploadError(404, "上传不存在或已过期")
        return row

    @staticmethod
    def _part_count(row) -> int:
        return (row["size"] + PART_BYTES - 1) // PART_BYTES

    def _state(self, row) -> dict:
        directory = self.root / row["id"]
        count = self._part_count(row)
        ready = (directory / "audio").is_file()
        received = [i for i in range(1, count + 1) if ready or (directory / f"{i}.part").is_file()]
        return {"upload_id": row["id"], "part_bytes": PART_BYTES, "part_count": count,
                "received_parts": received, "ready": ready, "expires_at": int(row["expires"])}

    def initialize(self, filename: str | None, size: int | None, sha256: str | None) -> dict:
        if (not filename or len(filename) > 255 or any(c in filename for c in "/\\")
                or any(ord(c) < 32 for c in filename) or Path(filename).suffix.lower() not in self.extensions):
            raise UploadError(400, "无效的音频文件名或扩展名")
        if size is None or not 0 < size <= self.max_file_size:
            raise UploadError(400, f"音频文件大小必须介于 1 和 {self.max_file_size} 字节之间")
        sha256 = (sha256 or "").lower()
        if not SHA256.fullmatch(sha256):
            raise UploadError(400, "upload_sha256 必须是 64 位十六进制 SHA-256")
        now = time.time()
        upload_id = secrets.token_hex(24)
        with self.transaction() as db:
            self._expire(db, now)
            reserved = db.execute("SELECT COALESCE(SUM(size), 0) FROM uploads").fetchone()[0]
            if reserved + size > self.capacity:
                raise UploadError(507, "分片上传暂存空间不足")
            db.execute("INSERT INTO uploads VALUES (?, ?, ?, ?, ?)", (upload_id, filename, size, sha256, now + self.ttl))
            return self._state(self._find(db, upload_id))

    def status(self, upload_id: str | None) -> dict:
        with self.transaction() as db:
            return self._state(self._find(db, upload_id))

    def store_part(self, upload_id: str | None, number: int | None, source: BinaryIO) -> dict:
        with self.transaction() as db:
            row = self._find(db, upload_id)
            if number is None or not 1 <= number <= self._part_count(row):
                raise UploadError(400, "part_number 超出范围")
            directory = self.root / row["id"]
            directory.mkdir(mode=0o700, exist_ok=True)
            part = directory / f"{number}.part"
            draft = directory / "incoming.tmp"
            limit = min(PART_BYTES, row["size"] - (number - 1) * PART_BYTES)
            digest = hashlib.sha256()
            try:
                with draft.open("wb") as out:
                    size = copy_blocks(source, out, limit, digest)
                if size != limit:
                    raise UploadError(400, f"该分片应为 {limit} 字节")
                ready = directory / "audio"
                if ready.is_file():
                    with ready.open("rb") as original:
                        original.seek((number - 1) * PART_BYTES)
                        existing = hashlib.sha256()
                        remaining = limit
                        while remaining:
                            block = original.read(min(BLOCK_BYTES, remaining))
                            if not block:
                                raise UploadError(409, "已合并文件不完整")
                            remaining -= len(block)
                            existing.update(block)
                    same = existing.digest() == digest.digest()
                elif part.exists():
                    with part.open("rb") as prior:
                        existing = hashlib.sha256()
                        while block := prior.read(BLOCK_BYTES):
                            existing.update(block)
                        same = existing.digest() == digest.digest()
                else:
                    os.replace(draft, part)
                    return self._state(row)
                if not same:
                    raise UploadError(409, "该片号已收到不同内容")
                return self._state(row)
            finally:
                draft.unlink(missing_ok=True)

    def complete(self, upload_id: str | None) -> dict:
        with self.transaction() as db:
            row = self._find(db, upload_id)
            state = self._state(row)
            directory = self.root / row["id"]
            if len(state["received_parts"]) != state["part_count"]:
                raise UploadError(409, "分片尚未传齐，请先查询已收到的片号")
            if not state["ready"]:
                draft = directory / "assembled.tmp"
                digest = hashlib.sha256()
                size = 0
                try:
                    with draft.open("wb") as out:
                        for number in state["received_parts"]:
                            with (directory / f"{number}.part").open("rb") as part:
                                size += copy_blocks(part, out, row["size"] - size, digest)
                    if size != row["size"] or digest.hexdigest() != row["sha256"]:
                        raise UploadError(422, "合并文件的字节数或 SHA-256 校验失败")
                    os.replace(draft, directory / "audio")
                finally:
                    draft.unlink(missing_ok=True)
            for path in directory.glob("*.part"):
                path.unlink(missing_ok=True)
            return self._state(row)

    def open_complete(self, upload_id: str | None) -> tuple[str, BinaryIO]:
        with self.transaction() as db:
            row = self._find(db, upload_id)
            path = self.root / row["id"] / "audio"
            if not path.is_file():
                raise UploadError(409, "请先完成分片合并")
            return row["filename"], path.open("rb")
