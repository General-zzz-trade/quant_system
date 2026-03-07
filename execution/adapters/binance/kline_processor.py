# execution/adapters/binance/kline_processor.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from event.factory.market import MarketEventFactory
from event.types import MarketEvent

from _quant_hotpath import rust_parse_kline


@dataclass(frozen=True, slots=True)
class KlineProcessor:
    """Converts Binance kline WebSocket JSON to MarketEvent.

    Only emits MarketEvent for closed klines (k.x == true) by default,
    preventing partial candle data from triggering strategy signals.
    """

    source: str = "binance.ws.kline"
    run_id: Optional[str] = None
    only_closed: bool = True

    def process_raw(self, raw: str) -> Optional[MarketEvent]:
        d = rust_parse_kline(raw, self.only_closed)
        if d is None:
            return None
        ts = datetime.fromtimestamp(d["ts_ms"] / 1000.0, tz=timezone.utc)
        return MarketEventFactory.bar(
            ts=ts,
            symbol=d["symbol"],
            open=Decimal(d["open"]),
            high=Decimal(d["high"]),
            low=Decimal(d["low"]),
            close=Decimal(d["close"]),
            volume=Decimal(d["volume"]),
            source=self.source,
            run_id=self.run_id,
        )
