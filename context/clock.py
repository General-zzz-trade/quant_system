# context/clock.py
"""Context-level clock — wraps event.clock.Clock with additional tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from event.clock import Clock


@dataclass(frozen=True, slots=True)
class ClockSnapshot:
    """时钟快照 — 不可变。"""
    ts: Any
    bar_index: int
    wall_time_ms: int


def clock_snapshot(clock: Clock) -> ClockSnapshot:
    """从 Clock 生成不可变快照。"""
    return ClockSnapshot(
        ts=clock.ts,
        bar_index=clock.bar_index,
        wall_time_ms=int(time.time() * 1000),
    )


def is_clock_advancing(prev_ts: Any, new_ts: Any) -> bool:
    """检查时钟是否前进（不允许倒流）。"""
    if prev_ts is None:
        return True
    try:
        return new_ts >= prev_ts
    except TypeError:
        return False
