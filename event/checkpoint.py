# event/checkpoint.py
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


# ============================================================
# 工具
# ============================================================

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(data.encode("utf-8")).hexdigest()


def _is_jsonable(x: Any) -> bool:
    try:
        json.dumps(x)
        return True
    except Exception:
        return False


# ============================================================
# Cursor
# ============================================================

@dataclass(frozen=True, slots=True)
class StreamCursor:
    """
    单一数据流游标
    """
    stream_id: str
    cursor: Mapping[str, Any]

    def validate(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id 不能为空")
        if not isinstance(self.cursor, Mapping):
            raise ValueError("cursor 必须为 Mapping")
        if not _is_jsonable(self.cursor):
            raise ValueError("cursor 必须可 JSON 序列化")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "cursor": dict(self.cursor),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StreamCursor":
        return cls(
            stream_id=str(data["stream_id"]),
            cursor=dict(data["cursor"]),
        )


# ============================================================
# Checkpoint
# ============================================================

@dataclass(frozen=True, slots=True)
class Checkpoint:
    """
    回放 / 续跑检查点（机构级定义）
    """
    run_id: str
    name: str

    checkpoint_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    version: int = 1

    stream_cursors: Tuple[StreamCursor, ...] = ()

    last_event_time: Optional[str] = None
    created_at: str = field(default_factory=_utc_now)

    fingerprint: Optional[str] = None

    # --------------------------------------------------------
    # 校验
    # --------------------------------------------------------

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id 不能为空")
        if not self.name:
            raise ValueError("name 不能为空")
        if self.version <= 0:
            raise ValueError("version 必须为正整数")

        # stream_id 唯一性
        ids = [sc.stream_id for sc in self.stream_cursors]
        if len(ids) != len(set(ids)):
            raise ValueError("stream_cursors 中存在重复 stream_id")

        for sc in self.stream_cursors:
            sc.validate()

        # last_event_time 必须为 ISO 字符串（如果存在）
        if self.last_event_time is not None:
            try:
                datetime.fromisoformat(self.last_event_time)
            except Exception as e:
                raise ValueError("last_event_time 非法 ISO 格式") from e

    # --------------------------------------------------------
    # 稳定内容哈希（制度级）
    # --------------------------------------------------------

    def content_hash(self) -> str:
        """
        注意：
        - 排除 checkpoint_id / created_at / version
        - 确保相同内容 → 相同 hash
        """
        payload = {
            "run_id": self.run_id,
            "name": self.name,
            "stream_cursors": [sc.to_dict() for sc in self.stream_cursors],
            "last_event_time": self.last_event_time,
            "fingerprint": self.fingerprint,
        }
        return _hash_payload(payload)

    # --------------------------------------------------------
    # 序列化
    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "checkpoint_id": self.checkpoint_id,
            "version": self.version,
            "stream_cursors": [sc.to_dict() for sc in self.stream_cursors],
            "last_event_time": self.last_event_time,
            "created_at": self.created_at,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Checkpoint":
        return cls(
            run_id=str(data["run_id"]),
            name=str(data["name"]),
            checkpoint_id=str(data["checkpoint_id"]),
            version=int(data["version"]),
            stream_cursors=tuple(
                StreamCursor.from_dict(x)
                for x in data.get("stream_cursors", [])
            ),
            last_event_time=data.get("last_event_time"),
            created_at=str(data["created_at"]),
            fingerprint=data.get("fingerprint"),
        )


# ============================================================
# Store 接口
# ============================================================

class CheckpointStore:
    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        raise NotImplementedError

    def load_latest(self, *, run_id: str, name: str) -> Optional[Checkpoint]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ============================================================
# InMemory 实现（测试用）
# ============================================================

class InMemoryCheckpointStore(CheckpointStore):
    def __init__(self) -> None:
        self._items: Dict[Tuple[str, str], Checkpoint] = {}

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        checkpoint.validate()
        key = (checkpoint.run_id, checkpoint.name)
        self._items[key] = checkpoint
        return checkpoint

    def load_latest(self, *, run_id: str, name: str) -> Optional[Checkpoint]:
        return self._items.get((run_id, name))


# ============================================================
# SQLite 实现（生产）
# ============================================================

class SQLiteCheckpointStore(CheckpointStore):
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(run_id, name, version)
            );
            """
        )
        self._conn.commit()

    def _next_version(self, run_id: str, name: str) -> int:
        cur = self._conn.execute(
            "SELECT MAX(version) FROM checkpoints WHERE run_id=? AND name=?",
            (run_id, name),
        )
        row = cur.fetchone()
        return int(row[0] or 0) + 1

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        checkpoint.validate()

        with self._lock:
            for _ in range(2):  # retry once
                version = self._next_version(checkpoint.run_id, checkpoint.name)
                cp = Checkpoint(
                    **{**checkpoint.to_dict(), "version": version}
                )
                payload = json.dumps(cp.to_dict(), separators=(",", ":"))
                try:
                    self._conn.execute(
                        """
                        INSERT INTO checkpoints
                        (run_id, name, checkpoint_id, version, payload, content_hash, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cp.run_id,
                            cp.name,
                            cp.checkpoint_id,
                            cp.version,
                            payload,
                            cp.content_hash(),
                            cp.created_at,
                        ),
                    )
                    self._conn.commit()
                    return cp
                except sqlite3.IntegrityError:
                    continue

            raise RuntimeError("Checkpoint version 冲突，保存失败")

    def load_latest(self, *, run_id: str, name: str) -> Optional[Checkpoint]:
        cur = self._conn.execute(
            """
            SELECT payload FROM checkpoints
            WHERE run_id=? AND name=?
            ORDER BY version DESC LIMIT 1
            """,
            (run_id, name),
        )
        row = cur.fetchone()
        if not row:
            return None
        return Checkpoint.from_dict(json.loads(row[0]))

    def close(self) -> None:
        self._conn.close()
