from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Optional

from state.snapshot import StateSnapshot


@dataclass(frozen=True, slots=True)
class FixedFractionSizer:
    """Target position based on equity * fraction * weight / price."""

    fraction: Decimal = Decimal("0.02")
    lot_size: Optional[Decimal] = None

    def _equity(self, snapshot: StateSnapshot) -> Decimal:
        acct = snapshot.account
        equity = acct.balance
        try:
            equity = equity + acct.unrealized_pnl
        except Exception:
            pass
        return equity

    def target_qty(self, snapshot: StateSnapshot, symbol: str, weight: Decimal) -> Decimal:
        m = snapshot.market
        price = m.close if getattr(m, "close", None) is not None else getattr(m, "last_price", None)
        price = Decimal(str(price))
        if price <= 0:
            return Decimal("0")
        notional = self._equity(snapshot) * self.fraction * abs(weight)
        qty = notional / price
        if self.lot_size and self.lot_size > 0:
            # floor to lot size
            steps = (qty / self.lot_size).to_integral_value(rounding=ROUND_DOWN)
            qty = steps * self.lot_size
        return qty
