from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from state._util import ensure_utc


@dataclass(frozen=True, slots=True)
class MarketState:
    """Market facts (SSOT).

    Stores only observed market facts; no predictions.
    """

    symbol: str

    # last traded/observed price
    last_price: Optional[Decimal] = None

    # last OHLCV bar facts (if bar-driven)
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    volume: Optional[Decimal] = None

    # last event timestamp in UTC
    last_ts: Optional[datetime] = None

    @classmethod
    def empty(cls, symbol: str) -> "MarketState":
        return cls(symbol=symbol)

    def with_tick(self, *, price: Decimal, ts: Optional[datetime]) -> "MarketState":
        return MarketState(
            symbol=self.symbol,
            last_price=price,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            last_ts=ensure_utc(ts) if ts is not None else self.last_ts,
        )

    def with_bar(
        self,
        *,
        o: Decimal,
        h: Decimal,
        l: Decimal,
        c: Decimal,
        v: Optional[Decimal],
        ts: Optional[datetime],
    ) -> "MarketState":
        return MarketState(
            symbol=self.symbol,
            last_price=c,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            last_ts=ensure_utc(ts) if ts is not None else self.last_ts,
        )
