"""One closed bar and its conversion from an exchange kline row. Standard library only: the offline
safety suite imports this module without numpy or pandas."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

HOUR = 3_600_000
KLINE_EXTRAS = ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume")


@dataclass(frozen=True)
class Bar:
    open_time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    interval_ms: int = HOUR
    # Optional kline microstructure fields (trades, taker volumes, quote volume).
    # Excluded from equality; included in record() only when present so older
    # dataset fingerprints stay stable.
    extras: dict[str, Any] | None = field(default=None, compare=False)

    @property
    def available_at(self) -> int:
        return self.open_time + self.interval_ms

    def record(self) -> dict[str, Any]:
        out = {k: str(v) if isinstance(v, Decimal) else v for k, v in asdict(self).items() if k != "extras"}
        if self.extras:
            out["extras"] = {k: str(v) for k, v in sorted(self.extras.items())}
        return out


def bar_from_kline(row: dict[str, Any]) -> Bar:
    """A closed 1h bar from an adapter kline row (``start`` in milliseconds); the open time must be hour-aligned."""
    start = int(row["start"])
    if start % HOUR:
        raise ValueError(f"kline open time {start} is not hour-aligned")
    extras = None
    if all(k in row for k in KLINE_EXTRAS):
        extras = {k: Decimal(str(row[k])) for k in KLINE_EXTRAS}
    o, h, lo, c, v = (Decimal(str(row[k])) for k in ("open", "high", "low", "close", "volume"))
    return Bar(start, o, h, lo, c, v, extras=extras)


__all__ = ["HOUR", "Bar", "bar_from_kline"]
