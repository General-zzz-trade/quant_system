# execution/adapters/binance/kline_processor.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from event.factory.market import MarketEventFactory
from event.types import MarketEvent


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
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(payload, dict):
            return None

        # Binance combined stream format: {"stream": "...", "data": {...}}
        data = payload.get("data", payload)

        if str(data.get("e", "")).strip() != "kline":
            return None

        k = data.get("k")
        if not isinstance(k, dict):
            return None

        if self.only_closed and not k.get("x", False):
            return None

        ts_ms = k.get("t")
        if ts_ms is None:
            return None

        ts = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)

        return MarketEventFactory.bar(
            ts=ts,
            symbol=str(data.get("s", "")).upper(),
            open=Decimal(str(k["o"])),
            high=Decimal(str(k["h"])),
            low=Decimal(str(k["l"])),
            close=Decimal(str(k["c"])),
            volume=Decimal(str(k["v"])),
            source=self.source,
            run_id=self.run_id,
        )
