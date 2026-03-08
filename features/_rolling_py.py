from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Sequence


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
