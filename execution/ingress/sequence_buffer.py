# execution/ingress/sequence_buffer.py
"""Sequence-based reordering buffer for incoming execution messages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from _quant_hotpath import RustSequenceBuffer as _RustSequenceBuffer

@dataclass(frozen=True, slots=True)
class BufferedItem:
    """缓冲区中的一条消息。"""
    seq: int
    payload: Any
    key: str = ""


class SequenceBuffer:
    """Rust-backed SequenceBuffer with same API."""

    def __init__(self, *, max_buffer_size: int = 1000) -> None:
        self._inner = _RustSequenceBuffer(max_buffer_size=max_buffer_size)

    def push(self, *, key: str, seq: int, payload: Any) -> Sequence[Any]:
        return self._inner.push(key, seq, payload)

    def expected_seq(self, key: str) -> int:
        return self._inner.expected_seq(key)

    def buffered_count(self, key: str) -> int:
        return self._inner.buffered_count(key)

    def flush(self, key: str) -> Sequence[Any]:
        return self._inner.flush(key)

    def pending_count(self, key: Optional[str] = None) -> int:
        return self._inner.pending_count(key)

    def reset(self, key: str) -> None:
        self._inner.reset_key(key)

    def clear(self) -> None:
        self._inner.clear()


def make_sequence_buffer(**kwargs: Any) -> SequenceBuffer:
    return SequenceBuffer(**kwargs)
