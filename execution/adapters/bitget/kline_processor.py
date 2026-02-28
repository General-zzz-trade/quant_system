# execution/adapters/bitget/kline_processor.py
"""Convert raw Bitget kline data to MarketEvent."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from event.factory.market import MarketEventFactory
from event.types import MarketEvent


@dataclass(frozen=True, slots=True)
class BitgetKlineRaw:
    """Raw kline data from Bitget WebSocket candle channel."""
    symbol: str
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class BitgetKlineProcessor:
    """Converts BitgetKlineRaw to MarketEvent via MarketEventFactory.bar()."""

    source: str = "bitget.ws.kline"
    run_id: Optional[str] = None

    def process(self, raw: BitgetKlineRaw) -> Optional[MarketEvent]:
        try:
            ts = datetime.fromtimestamp(raw.ts_ms / 1000.0, tz=timezone.utc)
            return MarketEventFactory.bar(
                ts=ts,
                symbol=raw.symbol,
                open=Decimal(str(raw.open)),
                high=Decimal(str(raw.high)),
                low=Decimal(str(raw.low)),
                close=Decimal(str(raw.close)),
                volume=Decimal(str(raw.volume)),
                source=self.source,
                run_id=self.run_id,
            )
        except Exception:
            return None
