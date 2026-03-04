# execution/ingress/sequence_buffer.py
"""Sequence-based reordering buffer for incoming execution messages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar

try:
    from _quant_hotpath import RustSequenceBuffer as _RustSequenceBuffer
    _HAS_RUST = True
except ImportError:
    _RustSequenceBuffer = None  # type: ignore
    _HAS_RUST = False

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class BufferedItem:
    """缓冲区中的一条消息。"""
    seq: int
    payload: Any
    key: str = ""


class SequenceBuffer:
    """
    序列缓冲区 — 处理乱序消息。

    接收带序号的消息，按序号顺序释放。
    处理间隙（gap）和乱序到达。
    """

    def __init__(self, *, max_buffer_size: int = 1000) -> None:
        self._max_buffer = max_buffer_size
        self._expected_seq: Dict[str, int] = {}
        self._buffer: Dict[str, Dict[int, Any]] = {}

    def push(self, *, key: str, seq: int, payload: Any) -> Sequence[Any]:
        """
        推入一条消息，返回所有可以按序释放的消息。

        如果 seq == expected，则直接释放并检查缓冲区中后续消息。
        如果 seq > expected，缓冲等待中间消息到达。
        如果 seq < expected，丢弃（已处理过的重复消息）。
        """
        expected = self._expected_seq.get(key, 0)

        if seq < expected:
            return []  # 重复/过期

        if key not in self._buffer:
            self._buffer[key] = {}

        if seq > expected:
            buf = self._buffer[key]
            if len(buf) < self._max_buffer:
                buf[seq] = payload
            return []

        # seq == expected: 释放连续序列
        released: list[Any] = [payload]
        next_seq = seq + 1
        buf = self._buffer[key]
        while next_seq in buf:
            released.append(buf.pop(next_seq))
            next_seq += 1
        self._expected_seq[key] = next_seq
        return released

    def expected_seq(self, key: str) -> int:
        """获取某个 key 的期望序号。"""
        return self._expected_seq.get(key, 0)

    def buffered_count(self, key: str) -> int:
        """获取某个 key 的缓冲消息数。"""
        return len(self._buffer.get(key, {}))

    def reset(self, key: str) -> None:
        """重置某个 key 的状态。"""
        self._expected_seq.pop(key, None)
        self._buffer.pop(key, None)

    def clear(self) -> None:
        """清空所有状态。"""
        self._expected_seq.clear()
        self._buffer.clear()


class RustSequenceBufferAdapter:
    """Rust-backed SequenceBuffer with same API."""

    def __init__(self, *, max_buffer_size: int = 1000) -> None:
        self._inner = _RustSequenceBuffer(max_buffer_size=max_buffer_size)

    def push(self, *, key: str, seq: int, payload: Any) -> Sequence[Any]:
        return self._inner.push(key, seq, payload)

    def expected_seq(self, key: str) -> int:
        return self._inner.expected_seq(key)

    def buffered_count(self, key: str) -> int:
        return self._inner.buffered_count(key)

    def reset(self, key: str) -> None:
        self._inner.reset_key(key)

    def clear(self) -> None:
        self._inner.clear()


def make_sequence_buffer(**kwargs: Any) -> SequenceBuffer:
    if _HAS_RUST:
        return RustSequenceBufferAdapter(**kwargs)  # type: ignore[return-value]
    return SequenceBuffer(**kwargs)
