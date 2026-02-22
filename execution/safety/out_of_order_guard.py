# execution/safety/out_of_order_guard.py
"""Sequence-based reordering buffer for fills and order updates."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True, slots=True)
class SequencedMessage:
    key: str         # 分组 key（例如 order_id）
    seq: int         # 序列号
    payload: Any


class OutOfOrderGuard:
    """
    乱序消息防护

    按 key 分组维护 next_expected_seq：
    - seq == next → 释放当前 + 缓冲中连续后续
    - seq > next  → 缓冲等待
    - seq < next  → 丢弃（已处理）
    """

    def __init__(self, *, max_buffer_per_key: int = 100) -> None:
        self._lock = RLock()
        self._next_seq: Dict[str, int] = defaultdict(lambda: 0)
        self._buffer: Dict[str, Dict[int, Any]] = defaultdict(dict)
        self._max_buffer = max_buffer_per_key

    def process(self, msg: SequencedMessage) -> Sequence[Any]:
        """返回按序可释放的 payload 列表。空列表 = 缓冲或丢弃。"""
        with self._lock:
            expected = self._next_seq[msg.key]

            if msg.seq < expected:
                return []

            if msg.seq > expected:
                buf = self._buffer[msg.key]
                if len(buf) < self._max_buffer:
                    buf[msg.seq] = msg.payload
                return []

            released: list[Any] = [msg.payload]
            self._next_seq[msg.key] = msg.seq + 1

            buf = self._buffer[msg.key]
            while self._next_seq[msg.key] in buf:
                next_seq = self._next_seq[msg.key]
                released.append(buf.pop(next_seq))
                self._next_seq[msg.key] = next_seq + 1

            if not buf and msg.key in self._buffer:
                del self._buffer[msg.key]

            return released

    def flush(self, key: str) -> Sequence[Any]:
        with self._lock:
            buf = self._buffer.pop(key, {})
            self._next_seq.pop(key, None)
            return [buf[seq] for seq in sorted(buf)]

    def pending_count(self, key: Optional[str] = None) -> int:
        with self._lock:
            if key is not None:
                return len(self._buffer.get(key, {}))
            return sum(len(b) for b in self._buffer.values())

    def reset(self) -> None:
        with self._lock:
            self._next_seq.clear()
            self._buffer.clear()
