# execution/bridge/live_execution_bridge.py
"""LiveExecutionBridge — unified adapter routing small orders direct, large orders to algos."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LiveExecutionConfig:
    """Configuration for live execution routing thresholds."""
    large_order_notional: Decimal = Decimal("10000")
    default_algo: str = "twap"


@dataclass
class LiveExecutionBridge:
    """Combines ExecutionBridge + AlgoExecutionAdapter for live trading.

    Small orders (notional < large_order_notional) go directly through
    ExecutionBridge.submit() for immediate execution.

    Large orders are delegated to AlgoExecutionAdapter for algorithmic
    slicing (TWAP, VWAP, Iceberg).

    Fill callbacks are injected into the pipeline via dispatcher_emit.
    """

    execution_bridge: Any  # ExecutionBridge
    algo_adapter: Optional[Any] = None  # AlgoExecutionAdapter
    config: LiveExecutionConfig = field(default_factory=LiveExecutionConfig)
    dispatcher_emit: Optional[Callable[[Any], None]] = None

    _order_count: int = field(default=0, init=False, repr=False)

    def send_order(self, order_event: Any) -> Iterable[Any]:
        """ExecutionAdapter protocol — routes to bridge or algo based on notional."""
        symbol = getattr(order_event, "symbol", "")
        qty = Decimal(str(getattr(order_event, "qty", 0)))
        price = getattr(order_event, "price", None)
        notional = qty * Decimal(str(price)) if price else qty

        if notional >= self.config.large_order_notional and self.algo_adapter is not None:
            logger.info(
                "Large order %s notional=%s routed to algo adapter",
                symbol, notional,
            )
            return self.algo_adapter.send_order(order_event)

        # Small order — direct execution via bridge
        logger.debug(
            "Small order %s notional=%s routed to direct execution",
            symbol, notional,
        )
        return self._execute_direct(order_event)

    def _execute_direct(self, order_event: Any) -> Iterable[Any]:
        """Submit order directly through the execution bridge."""
        try:
            ack = self.execution_bridge.submit(order_event)
        except Exception:
            logger.exception("ExecutionBridge.submit failed for %s", getattr(order_event, "symbol", "?"))
            return []

        if ack.ok:
            fill = self._ack_to_fill(order_event, ack)
            self._order_count += 1
            if self.dispatcher_emit is not None:
                try:
                    self.dispatcher_emit(fill)
                except Exception:
                    logger.exception("dispatcher_emit failed for fill")
            return [fill]

        logger.warning(
            "Order rejected by bridge: status=%s error=%s",
            ack.status, getattr(ack, "error", None),
        )
        return []

    @staticmethod
    def _ack_to_fill(order_event: Any, ack: Any) -> SimpleNamespace:
        """Convert an Ack from ExecutionBridge into a fill event SimpleNamespace."""
        result = ack.result or {}
        fill_price = result.get("price") or getattr(order_event, "price", None)
        fill_qty = result.get("qty") or getattr(order_event, "qty", Decimal("0"))

        return SimpleNamespace(
            event_type="fill",
            EVENT_TYPE="fill",
            header=SimpleNamespace(
                event_type="fill",
                ts=None,
                event_id=f"bridge-fill-{uuid.uuid4().hex[:12]}",
            ),
            fill_id=f"bridge-fill-{uuid.uuid4().hex[:12]}",
            order_id=getattr(order_event, "order_id", ""),
            symbol=getattr(order_event, "symbol", ""),
            side=getattr(order_event, "side", ""),
            qty=Decimal(str(fill_qty)) if fill_qty is not None else Decimal("0"),
            quantity=Decimal(str(fill_qty)) if fill_qty is not None else Decimal("0"),
            price=Decimal(str(fill_price)) if fill_price is not None else Decimal("0"),
            fee=Decimal(str(result.get("fee", "0"))),
            realized_pnl=Decimal("0"),
            venue=getattr(ack, "venue", ""),
            command_id=getattr(ack, "command_id", ""),
        )

    @property
    def order_count(self) -> int:
        return self._order_count
