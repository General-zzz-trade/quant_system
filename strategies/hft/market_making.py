"""Market making strategy — provides liquidity with two-sided quotes.

Manages inventory risk by adjusting quote prices based on position.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Quote:
    """A single quote (bid or ask)."""
    price: Decimal
    qty: Decimal
    side: str  # "bid" or "ask"


@dataclass(frozen=True, slots=True)
class QuotePair:
    """Two-sided quote."""
    bid: Quote
    ask: Quote
    spread_bps: float
    inventory_skew: float


@dataclass
class MarketMaker:
    """Simple market making strategy with inventory management.

    Quotes are placed symmetrically around mid-price with adjustments for:
    - Base spread (minimum profitability)
    - Inventory skew (lean quotes to reduce position)
    - Volatility (widen spread in volatile markets)
    """

    symbol: str
    base_spread_bps: float = 10.0
    qty_per_side: Decimal = Decimal("0.01")
    max_inventory: Decimal = Decimal("1.0")
    inventory_skew_bps: float = 2.0  # bps per unit of inventory

    _position: Decimal = Decimal("0")

    def update_position(self, qty: Decimal) -> None:
        """Update current inventory position."""
        self._position = qty

    def compute_quotes(
        self,
        mid_price: Decimal,
        *,
        volatility: Optional[float] = None,
    ) -> Optional[QuotePair]:
        """Compute bid/ask quotes based on mid-price and current state."""
        if mid_price <= 0:
            return None

        # Inventory check
        if abs(self._position) >= self.max_inventory:
            logger.warning("Max inventory reached: %s", self._position)
            # Only quote the reducing side
            if self._position > 0:
                # Only quote ask (to sell)
                ask_price = mid_price * (Decimal("1") + Decimal(str(self.base_spread_bps / 2 / 10000)))
                return QuotePair(
                    bid=Quote(price=Decimal("0"), qty=Decimal("0"), side="bid"),
                    ask=Quote(price=ask_price, qty=self.qty_per_side, side="ask"),
                    spread_bps=0, inventory_skew=float(self._position),
                )
            else:
                bid_price = mid_price * (Decimal("1") - Decimal(str(self.base_spread_bps / 2 / 10000)))
                return QuotePair(
                    bid=Quote(price=bid_price, qty=self.qty_per_side, side="bid"),
                    ask=Quote(price=Decimal("0"), qty=Decimal("0"), side="ask"),
                    spread_bps=0, inventory_skew=float(self._position),
                )

        # Spread adjustment for volatility
        spread_bps = self.base_spread_bps
        if volatility is not None:
            vol_adj = volatility * 10000  # Convert to bps
            spread_bps = max(spread_bps, vol_adj * 0.5)

        half_spread = Decimal(str(spread_bps / 2 / 10000))

        # Inventory skew: lean quotes to reduce position
        inventory_adj = Decimal(str(float(self._position) * self.inventory_skew_bps / 10000))

        bid_price = mid_price * (Decimal("1") - half_spread - inventory_adj)
        ask_price = mid_price * (Decimal("1") + half_spread - inventory_adj)

        return QuotePair(
            bid=Quote(price=bid_price, qty=self.qty_per_side, side="bid"),
            ask=Quote(price=ask_price, qty=self.qty_per_side, side="ask"),
            spread_bps=float(spread_bps),
            inventory_skew=float(self._position),
        )
