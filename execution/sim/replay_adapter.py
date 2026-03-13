# execution/sim/replay_adapter.py
"""ReplayExecutionAdapter — records orders AND produces FillEvents for causal chain closure."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

try:
    from _quant_hotpath import RustFillEvent as _FillEventCls
    _USE_RUST_FILL = True
except ImportError:
    _USE_RUST_FILL = False

from event.header import EventHeader
from event.types import EventType, FillEvent

logger = logging.getLogger(__name__)


@dataclass
class ReplayExecutionAdapter:
    """Execution adapter for replay: records orders and produces synthetic FillEvents.

    Unlike ShadowExecutionAdapter (which returns []), this adapter produces FillEvent
    objects so fills flow back through the dispatcher into the pipeline, updating
    position state. This closes the causal chain:

        market → features → signal → order → fill → position update

    Parameters:
        price_source: callable(symbol) -> Optional[Decimal] for fill price
        fee_bps: simulated fee in basis points
        slippage_bps: simulated slippage in basis points
    """

    price_source: Callable[[str], Optional[Decimal]]
    fee_bps: Decimal = Decimal("4")
    slippage_bps: Decimal = Decimal("2")
    _log: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def send_order(self, order_event: Any) -> list:
        symbol = getattr(order_event, "symbol", "UNKNOWN")
        side = getattr(order_event, "side", "UNKNOWN")
        qty = getattr(order_event, "qty", Decimal("0"))
        order_id = getattr(order_event, "order_id", str(uuid.uuid4()))
        ts = time.time()

        entry: Dict[str, Any] = {
            "ts": ts,
            "symbol": str(symbol),
            "side": str(side),
            "qty": str(qty),
            "order_id": str(order_id),
        }

        price = self.price_source(str(symbol))
        if price is None:
            entry["fill_price"] = None
            entry["fee"] = None
            self._log.append(entry)
            logger.warning("REPLAY order: no price for %s, skipping fill", symbol)
            return []

        # Apply slippage
        slip_mult = self.slippage_bps / Decimal("10000")
        if "BUY" in str(side).upper():
            fill_price = price * (Decimal("1") + slip_mult)
        else:
            fill_price = price * (Decimal("1") - slip_mult)

        fee = fill_price * abs(Decimal(str(qty))) * self.fee_bps / Decimal("10000")
        entry["fill_price"] = str(fill_price)
        entry["fee"] = str(fee)
        self._log.append(entry)

        logger.info(
            "REPLAY order: %s %s %s @ %s",
            side, qty, symbol, fill_price,
        )

        # Produce FillEvent so it flows back through dispatcher → pipeline → position update
        # Use RustFillEvent when available — RustStateStore requires `side` field
        if _USE_RUST_FILL:
            fill = _FillEventCls(
                symbol=str(symbol),
                side=str(side).lower(),
                qty=float(qty),
                price=float(fill_price),
            )
        else:
            fill = FillEvent(
                header=EventHeader.new_root(
                    event_type=EventType.FILL,
                    version=1,
                    source="replay_adapter",
                ),
                fill_id=str(uuid.uuid4()),
                order_id=str(order_id),
                symbol=str(symbol),
                qty=Decimal(str(qty)),
                price=fill_price,
            )
        return [fill]

    @property
    def order_log(self) -> List[Dict[str, Any]]:
        return list(self._log)
