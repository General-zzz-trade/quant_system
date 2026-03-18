# execution/bridge/live_execution_bridge.py
"""LiveExecutionBridge — unified adapter routing small orders direct, large orders to algos."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Iterable, Optional

from execution.models.acks import normalize_ack
from execution.models.fill_events import build_synthetic_ingress_fill_event
from execution.observability.incidents import synthetic_fill_to_alert
from execution.observability.rejections import rejection_event_to_alert
from execution.models.rejection_events import rejection_to_event
from execution.models.rejections import ack_to_rejection

logger = logging.getLogger(__name__)

_TERMINAL_FILL_STATUSES = {"FILLED", "PARTIALLY_FILLED"}


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
    on_reject: Optional[Callable[[Any], None]] = None
    on_reject_event: Optional[Callable[[Any], None]] = None
    alert_manager: Optional[Any] = None
    incident_logger: Optional[Callable[[Any], None]] = None

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
            raw_ack = self.execution_bridge.submit(order_event)
        except Exception:
            logger.exception("ExecutionBridge.submit failed for %s", getattr(order_event, "symbol", "?"))
            return []

        ack = normalize_ack(
            raw_ack,
            default_venue=str(getattr(order_event, "venue", "")),
            default_symbol=str(getattr(order_event, "symbol", "")),
        )

        if ack.ok:
            if self._ack_contains_fill_evidence(ack):
                fill = self._ack_to_fill(order_event, ack)
                self._order_count += 1
                self._emit_incident(
                    synthetic_fill_to_alert(fill),
                    symbol=getattr(order_event, "symbol", "?"),
                )
                if self.dispatcher_emit is not None:
                    try:
                        self.dispatcher_emit(fill)
                    except Exception:
                        logger.exception("dispatcher_emit failed for fill")
                return [fill]
            logger.info(
                "Order accepted without fill evidence: status=%s result_keys=%s",
                ack.status,
                sorted((ack.result or {}).keys()),
            )
            return []

        logger.warning(
            "Order rejected by bridge: status=%s error=%s",
            ack.status, getattr(ack, "error", None),
        )
        rejection = ack_to_rejection(ack)
        if rejection is not None and self.on_reject is not None:
            try:
                self.on_reject(rejection)
            except Exception:
                logger.exception("on_reject failed for %s", getattr(order_event, "symbol", "?"))
        rejection_event = rejection_to_event(rejection) if rejection is not None else None
        if rejection_event is not None and self.on_reject_event is not None:
            try:
                self.on_reject_event(rejection_event)
            except Exception:
                logger.exception("on_reject_event failed for %s", getattr(order_event, "symbol", "?"))
        if rejection_event is not None and (self.alert_manager is not None or self.incident_logger is not None):
            self._emit_incident(rejection_event_to_alert(rejection_event), symbol=getattr(order_event, "symbol", "?"))
        return []

    @staticmethod
    def _ack_to_fill(order_event: Any, ack: Any) -> Any:
        """Convert an Ack from ExecutionBridge into the standardized ingress fill event."""
        normalized = normalize_ack(
            ack,
            default_venue=str(getattr(order_event, "venue", "")),
            default_symbol=str(getattr(order_event, "symbol", "")),
        )
        result = normalized.result or {}
        fill_price = (
            result.get("avg_price")
            or result.get("avgPrice")
            or result.get("price")
            or getattr(order_event, "price", None)
            or Decimal("0")
        )
        fill_qty = (
            result.get("filled_qty")
            or result.get("executedQty")
            or result.get("cumExecQty")
            or result.get("qty")
            or getattr(order_event, "qty", None)
            or Decimal("0")
        )
        return build_synthetic_ingress_fill_event(
            source="bridge",
            symbol=getattr(order_event, "symbol", ""),
            qty=fill_qty if fill_qty is not None else Decimal("0"),
            side=getattr(order_event, "side", ""),
            price=fill_price if fill_price is not None else Decimal("0"),
            fee=result.get("fee", "0"),
            venue=normalized.venue,
            order_id=result.get("order_id") or getattr(order_event, "order_id", ""),
            identity_seed=(
                result.get("trade_id")
                or result.get("fill_id")
                or normalized.command_id
                or getattr(order_event, "command_id", "")
            ),
        )

    @staticmethod
    def _ack_contains_fill_evidence(ack: Any) -> bool:
        normalized = normalize_ack(ack)
        result = normalized.result or {}
        if not result:
            return False

        if result.get("fill_id") or result.get("trade_id"):
            return True

        status = str(result.get("status", normalized.status or "")).upper()
        has_fill_qty = any(
            result.get(key) not in (None, "", 0, 0.0, "0", "0.0")
            for key in ("filled_qty", "executedQty", "cumExecQty", "qty")
        )
        has_fill_price = any(
            result.get(key) not in (None, "", 0, 0.0, "0", "0.0")
            for key in ("avg_price", "avgPrice", "price")
        )
        has_fee_field = any(key in result for key in ("fee", "commission", "fee_asset"))

        if status in _TERMINAL_FILL_STATUSES and has_fill_qty:
            return True
        if has_fill_qty and has_fill_price and has_fee_field:
            return True
        return False

    @property
    def order_count(self) -> int:
        return self._order_count

    def _emit_incident(self, alert: Any, *, symbol: str) -> None:
        if self.alert_manager is not None:
            try:
                self.alert_manager.emit_direct(alert)
            except Exception:
                logger.exception("incident alert emit failed for %s", symbol)
        if self.incident_logger is not None:
            try:
                self.incident_logger(alert)
            except Exception:
                logger.exception("incident logger failed for %s", symbol)
