from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Deque, List, Optional, Sequence


@dataclass
class RollingWindow:
    """A numeric rolling window with O(1) updates for sum and sumsq."""

    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        self._q: Deque[float] = deque()
        self._sum: float = 0.0
        self._sumsq: float = 0.0

    def push(self, x: float) -> None:
        self._q.append(x)
        self._sum += x
        self._sumsq += x * x
        if len(self._q) > self.size:
            old = self._q.popleft()
            self._sum -= old
            self._sumsq -= old * old

    @property
    def full(self) -> bool:
        return len(self._q) == self.size

    @property
    def n(self) -> int:
        return len(self._q)

    @property
    def mean(self) -> Optional[float]:
        if self.n == 0:
            return None
        return self._sum / self.n

    @property
    def variance(self) -> Optional[float]:
        if self.n == 0:
            return None
        mu = self._sum / self.n
        v = self._sumsq / self.n - mu * mu
        return max(v, 0.0)

    @property
    def std(self) -> Optional[float]:
        v = self.variance
        return sqrt(v) if v is not None else None


def rolling_apply(values: Sequence[float], window: int, fn) -> List[Optional[float]]:
    """Apply a function on each rolling window.

    fn takes a list of floats and returns float.
    """

    if window <= 0:
        raise ValueError("window must be positive")

    out: List[Optional[float]] = [None] * len(values)
    q: Deque[float] = deque()

    for i, x in enumerate(values):
        q.append(float(x))
        if len(q) > window:
            q.popleft()
        if len(q) == window:
            out[i] = float(fn(list(q)))

    return out
