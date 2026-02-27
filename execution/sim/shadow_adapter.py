# execution/sim/shadow_adapter.py
"""ShadowExecutionAdapter — records orders without executing. For dry-run validation."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ShadowExecutionAdapter:
    """Records orders but does not execute them. Used for shadow/dry-run validation."""

    price_source: Callable[[str], Optional[Decimal]]
    fee_bps: Decimal = Decimal("4")
    slippage_bps: Decimal = Decimal("2")
    _log: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def send_order(self, order_event: Any) -> list:
        symbol = getattr(order_event, "symbol", "UNKNOWN")
        side = getattr(order_event, "side", "UNKNOWN")
        qty = getattr(order_event, "qty", Decimal("0"))
        ts = time.time()

        entry = {
            "ts": ts,
            "symbol": str(symbol),
            "side": str(side),
            "qty": str(qty),
            "simulated": True,
        }

        price = self.price_source(str(symbol))
        if price is not None:
            # Apply slippage: buy at higher price, sell at lower
            slip_mult = self.slippage_bps / Decimal("10000")
            if "BUY" in str(side).upper():
                fill_price = price * (Decimal("1") + slip_mult)
            else:
                fill_price = price * (Decimal("1") - slip_mult)

            fee = fill_price * Decimal(str(qty)) * self.fee_bps / Decimal("10000")
            entry["fill_price"] = str(fill_price)
            entry["fee"] = str(fee)
        else:
            entry["fill_price"] = None
            entry["fee"] = None

        self._log.append(entry)
        logger.info(
            "SHADOW order: %s %s %s @ %s",
            side, qty, symbol, entry.get("fill_price", "N/A"),
        )
        return []  # No real fill events

    @property
    def order_log(self) -> List[Dict[str, Any]]:
        return list(self._log)
