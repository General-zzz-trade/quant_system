"""Helper classes for MultiTimeframeSignal.

Extracted from multi_tf_signal.py: _ZScoreBuffer, _HoldState, _BarAcc.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Optional

import numpy as np


@dataclass
class _ZScoreBuffer:
    """Rolling z-score normalization for raw predictions."""
    window: int = 720
    warmup: int = 180
    _buf: Deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._buf = deque(maxlen=self.window)

    def push(self, value: float) -> float:
        """Append value and return z-score (0 if warmup not met)."""
        self._buf.append(value)
        if len(self._buf) < self.warmup:
            return 0.0
        arr = np.array(self._buf)
        std = float(np.std(arr))
        if std < 1e-12:
            return 0.0
        return (value - float(np.mean(arr))) / std

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.warmup


@dataclass
class _HoldState:
    """Min/max hold enforcement per symbol."""
    position: float = 0.0
    entry_bar: int = 0
    bar_count: int = 0

    def update(self, desired: float, z: float,
               min_hold: int, max_hold: int) -> float:
        """Apply hold constraints, return final position."""
        self.bar_count += 1

        if self.position != 0:
            held = self.bar_count - self.entry_bar
            if held >= max_hold:
                self.position = 0.0
            elif held >= min_hold:
                if self.position * z < -0.3 or abs(z) < 0.2:
                    self.position = 0.0

        if self.position == 0 and desired != 0:
            self.position = desired
            self.entry_bar = self.bar_count

        return self.position


@dataclass
class _BarAcc:
    """Lightweight 4h bar accumulator from 1h bars."""
    open: float = 0.0
    high: float = -1e18
    low: float = 1e18
    close: float = 0.0
    volume: float = 0.0
    count: int = 0
    start_ts: Optional[datetime] = None

    def push(self, o: float, h: float, l: float, c: float, v: float,  # noqa: E741
             ts: Optional[datetime] = None) -> bool:
        """Push 1h bar. Returns True when 4h bar completes (every 4 bars)."""
        if self.count == 0:
            self.open = o
            self.high = h
            self.low = l
            self.start_ts = ts
        else:
            self.high = max(self.high, h)
            self.low = min(self.low, l)
        self.close = c
        self.volume += v
        self.count += 1

        if self.count >= 4:
            return True
        return False

    def reset(self) -> None:
        self.open = 0.0
        self.high = -1e18
        self.low = 1e18
        self.close = 0.0
        self.volume = 0.0
        self.count = 0
        self.start_ts = None
