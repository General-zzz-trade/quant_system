from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Mapping, Optional, Sequence, Union

Number = Union[int, float]

FeatureName = str
FeatureSeries = List[Optional[float]]


@dataclass(frozen=True)
class Bar:
    """A minimal OHLCV bar."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


Bars = Sequence[Bar]


def to_bars(
    rows: Iterable[Mapping[str, object]],
    *,
    ts_key: str = "ts",
    open_key: str = "open",
    high_key: str = "high",
    low_key: str = "low",
    close_key: str = "close",
    volume_key: str = "volume",
) -> List[Bar]:
    """Convert row dicts to Bar objects.

    ts value may be datetime or ISO string.
    """

    out: List[Bar] = []
    for r in rows:
        ts = r.get(ts_key)
        if isinstance(ts, datetime):
            dt = ts
        else:
            dt = datetime.fromisoformat(str(ts))
        _open = r.get(open_key, 0.0)
        _high = r.get(high_key, 0.0)
        _low = r.get(low_key, 0.0)
        _close = r.get(close_key, 0.0)
        _vol = r.get(volume_key, 0.0)
        out.append(
            Bar(
                ts=dt,
                open=float(_open) if isinstance(_open, (int, float, str)) else 0.0,
                high=float(_high) if isinstance(_high, (int, float, str)) else 0.0,
                low=float(_low) if isinstance(_low, (int, float, str)) else 0.0,
                close=float(_close) if isinstance(_close, (int, float, str)) else 0.0,
                volume=float(_vol) if isinstance(_vol, (int, float, str)) else 0.0,
            )
        )
    return out
