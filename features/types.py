from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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
        out.append(
            Bar(
                ts=dt,
                open=float(r.get(open_key, 0.0)),
                high=float(r.get(high_key, 0.0)),
                low=float(r.get(low_key, 0.0)),
                close=float(r.get(close_key, 0.0)),
                volume=float(r.get(volume_key, 0.0)),
            )
        )
    return out
