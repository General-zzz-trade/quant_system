from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from execution.adapters.base import VenueAdapter
from execution.models.balances import AssetBalance, BalanceSnapshot
from execution.models.instruments import Instrument
from execution.models.positions import Position
from execution.models.orders import CanonicalOrder
from execution.models.fills import CanonicalFill
from execution.adapters.common.time import now_ms


@dataclass(slots=True)
class SimVenueAdapter(VenueAdapter):
    """In-memory simulated venue adapter for testing and backtests."""
    venue: str = "sim"
    product: str = "um"

    _instruments: Tuple[Instrument, ...] = ()
    _balances: Dict[str, AssetBalance] = field(default_factory=dict)
    _positions: Dict[str, Position] = field(default_factory=dict)
    _orders: Dict[str, CanonicalOrder] = field(default_factory=dict)
    _fills: Dict[str, CanonicalFill] = field(default_factory=dict)

    def list_instruments(self) -> Tuple[Instrument, ...]:
        return self._instruments

    def get_balances(self) -> BalanceSnapshot:
        return BalanceSnapshot(venue=self.venue, ts_ms=now_ms(), balances=tuple(self._balances.values()))

    def get_positions(self) -> Tuple[Position, ...]:
        return tuple(self._positions.values())

    def get_open_orders(self, *, symbol: Optional[str] = None) -> Tuple[CanonicalOrder, ...]:
        out = []
        for o in self._orders.values():
            if o.status in ("filled", "canceled", "rejected", "expired"):
                continue
            if symbol and o.symbol != symbol.upper():
                continue
            out.append(o)
        return tuple(out)

    def get_recent_fills(self, *, symbol: Optional[str] = None, since_ms: int = 0) -> Tuple[CanonicalFill, ...]:
        out = []
        for f in self._fills.values():
            if symbol and f.symbol != symbol.upper():
                continue
            if since_ms and f.ts_ms < since_ms:
                continue
            out.append(f)
        return tuple(out)
