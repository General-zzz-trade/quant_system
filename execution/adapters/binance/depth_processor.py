"""Order book depth processor — converts Binance depth stream to structured data.

Processes both snapshot and incremental depth update messages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Tuple

from _quant_hotpath import rust_parse_depth

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """Single price level in the order book."""
    price: Decimal
    qty: Decimal


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Order book snapshot at a point in time."""
    symbol: str
    bids: Tuple[OrderBookLevel, ...]
    asks: Tuple[OrderBookLevel, ...]
    ts_ms: int
    last_update_id: int

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    @property
    def spread_bps(self) -> Optional[Decimal]:
        sp, mid = self.spread, self.mid_price
        if sp is not None and mid and mid > 0:
            return sp / mid * 10000
        return None


class DepthProcessor:
    """Processes Binance depth stream messages into OrderBookSnapshot."""

    def __init__(self, *, max_levels: int = 20) -> None:
        self._max_levels = max_levels

    def process_raw(self, raw: str) -> Optional[OrderBookSnapshot]:
        """Parse a depth stream message.

        Handles both combined stream format and direct format.
        """
        d = rust_parse_depth(raw, self._max_levels)
        if d is None:
            return None
        bids = tuple(
            OrderBookLevel(price=Decimal(p), qty=Decimal(q))
            for p, q in d["bids"]
            if Decimal(q) > 0
        )
        asks = tuple(
            OrderBookLevel(price=Decimal(p), qty=Decimal(q))
            for p, q in d["asks"]
            if Decimal(q) > 0
        )
        return OrderBookSnapshot(
            symbol=d["symbol"],
            bids=bids,
            asks=asks,
            ts_ms=d["ts_ms"],
            last_update_id=d["last_update_id"],
        )

    def process_snapshot(self, data: Dict) -> Optional[OrderBookSnapshot]:
        """Process a REST API depth snapshot response."""
        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])
        last_id = data.get("lastUpdateId", 0)

        bids = tuple(
            OrderBookLevel(price=Decimal(str(b[0])), qty=Decimal(str(b[1])))
            for b in bids_raw[:self._max_levels]
            if Decimal(str(b[1])) > 0
        )
        asks = tuple(
            OrderBookLevel(price=Decimal(str(a[0])), qty=Decimal(str(a[1])))
            for a in asks_raw[:self._max_levels]
            if Decimal(str(a[1])) > 0
        )

        return OrderBookSnapshot(
            symbol=data.get("symbol", ""),
            bids=bids,
            asks=asks,
            ts_ms=0,
            last_update_id=last_id,
        )
