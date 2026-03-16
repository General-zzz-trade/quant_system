from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

from state.snapshot import StateSnapshot

from _quant_hotpath import rust_fixed_fraction_qty as _rust_ff_qty


@dataclass(frozen=True, slots=True)
class FixedFractionSizer:
    """Target position based on equity * fraction * weight / price."""

    fraction: Decimal = Decimal("0.02")
    lot_size: Optional[Decimal] = None

    def _equity(self, snapshot: StateSnapshot) -> Decimal:
        acct = snapshot.account
        bf = getattr(acct, "balance_f", None)
        if bf is not None:
            equity = bf
            uf = getattr(acct, "unrealized_pnl_f", None)
            if uf is not None:
                equity += uf
            return Decimal(str(equity))
        equity = acct.balance
        try:
            equity = equity + acct.unrealized_pnl
        except Exception as e:
            logger.warning("Failed to add unrealized PnL to equity: %s", e)
        return equity

    def target_qty(self, snapshot: StateSnapshot, symbol: str, weight: Decimal) -> Decimal:
        m = snapshot.market
        cf = getattr(m, "close_f", None)
        if cf is not None:
            price = Decimal(str(cf))
        else:
            raw = m.close if getattr(m, "close", None) is not None else getattr(m, "last_price", None)
            price = Decimal(str(raw))
        if price <= 0:
            return Decimal("0")

        lot = float(self.lot_size) if self.lot_size else 0.0
        qty_f = _rust_ff_qty(
            float(self._equity(snapshot)),
            float(price),
            float(self.fraction),
            float(weight),
            lot,
        )
        return Decimal(str(qty_f))
