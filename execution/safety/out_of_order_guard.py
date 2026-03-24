# execution/safety/out_of_order_guard.py
"""Sequence-based reordering buffer for fills and order updates.

Delegates to RustSequenceBuffer for lock-free, high-performance reordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from _quant_hotpath import RustSequenceBuffer  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class SequencedMessage:
    key: str         # grouping key (e.g. order_id)
    seq: int         # sequence number
    payload: Any


class OutOfOrderGuard:
    """Rust-backed out-of-order message guard.

    Maintains per-key next_expected_seq:
    - seq == next -> release current + buffered continuations
    - seq > next  -> buffer for later
    - seq < next  -> discard (already processed)
    """

    def __init__(self, *, max_buffer_per_key: int = 100) -> None:
        self._rust = RustSequenceBuffer(max_buffer_size=max_buffer_per_key)

    def process(self, msg: SequencedMessage) -> Sequence[Any]:
        """Return list of payloads released in order. Empty = buffered or discarded."""
        return list(self._rust.push(msg.key, msg.seq, msg.payload))

    def flush(self, key: str) -> Sequence[Any]:
        return list(self._rust.flush(key))

    def pending_count(self, key: Optional[str] = None) -> int:
        return self._rust.pending_count(key)

    def reset(self) -> None:
        self._rust.clear()
