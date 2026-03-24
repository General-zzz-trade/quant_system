# execution/adapters/binance/rest_kline_source.py
"""REST-based kline source for gap-filling when WS is disconnected."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List
from urllib.request import Request, urlopen

from event.header import EventHeader
from event.types import EventType, MarketEvent

logger = logging.getLogger(__name__)


@dataclass
class RestKlineSource:
    """Fetches recent klines via Binance REST API for WS gap-fill."""

    base_url: str = "https://fapi.binance.com"
    timeout_s: float = 10.0
    source: str = "binance.rest.kline"

    def fetch_recent(
        self, symbol: str, interval: str = "1m", limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Fetch recent closed klines as dicts compatible with KlineProcessor."""
        url = (
            f"{self.base_url}/fapi/v1/klines"
            f"?symbol={symbol}&interval={interval}&limit={limit}"
        )
        req = Request(url=url, method="GET")
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except Exception as e:
            logger.warning("REST kline fetch failed for %s: %s", symbol, e)
            return []

        results = []
        for row in data:
            if not isinstance(row, list) or len(row) < 11:
                continue
            results.append({
                "t": int(row[0]),       # open time ms
                "o": str(row[1]),       # open
                "h": str(row[2]),       # high
                "l": str(row[3]),       # low
                "c": str(row[4]),       # close
                "v": str(row[5]),       # volume
                "T": int(row[6]),       # close time ms
                "x": True,             # always closed for REST klines
            })
        return results

    def fetch_as_events(
        self, symbol: str, interval: str = "1m", limit: int = 5,
    ) -> List[MarketEvent]:
        """Fetch recent klines and convert to MarketEvent objects."""
        raw_klines = self.fetch_recent(symbol, interval, limit)
        events = []
        for k in raw_klines:
            ts = datetime.fromtimestamp(int(k["t"]) / 1000.0, tz=timezone.utc)
            header = EventHeader.new_root(
                event_type=EventType.MARKET,
                version=MarketEvent.VERSION,
                source=self.source,
            )
            ev = MarketEvent(
                header=header,
                ts=ts,
                symbol=symbol.upper(),
                open=Decimal(k["o"]),
                high=Decimal(k["h"]),
                low=Decimal(k["l"]),
                close=Decimal(k["c"]),
                volume=Decimal(k["v"]),
            )
            events.append(ev)
        return events
