"""LiveEmitHandler — extracted from LiveRunner._emit closure for testability.

Handles ORDER and FILL event routing through gate chain, order state machine,
timeout tracking, event recording, and attribution.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LiveEmitHandler:
    """Callable event handler that replaces the _emit closure in LiveRunner.

    Dispatches ORDER and FILL events through the appropriate subsystems,
    then forwards all events to the coordinator.

    Parameters
    ----------
    coordinator : EngineCoordinator
        Event loop orchestrator; receives all events via emit().
    attribution_tracker : Any
        Tracks all events for P&L attribution.
    gate_chain : Any
        Processes ORDER events through correlation/risk/portfolio/alpha gates.
    order_state_machine : Any
        Tracks order lifecycle (register, transition).
    timeout_tracker : Any
        Tracks order submission/fill for timeout detection.
    event_recorder : Optional[Any]
        Records fill events for replay recovery.
    live_signal_tracker : Optional[Any]
        Tracks fills for signal attribution feedback.
    """

    def __init__(
        self,
        coordinator: Any,
        attribution_tracker: Any,
        gate_chain: Any,
        order_state_machine: Any,
        timeout_tracker: Any,
        event_recorder: Optional[Any] = None,
        live_signal_tracker: Optional[Any] = None,
    ) -> None:
        self._coordinator = coordinator
        self._attribution_tracker = attribution_tracker
        self._gate_chain = gate_chain
        self._order_state_machine = order_state_machine
        self._timeout_tracker = timeout_tracker
        self._event_recorder = event_recorder
        self._live_signal_tracker = live_signal_tracker

    def __call__(self, ev: Any) -> None:
        # Attribution: track all events
        self._attribution_tracker.on_event(ev)

        et = getattr(ev, "event_type", None)
        et_str = (str(et.value) if hasattr(et, "value") else str(et)).upper() if et else ""

        if et_str == "ORDER":
            self._handle_order(ev)
            return
        elif et_str == "FILL":
            self._handle_fill(ev)

        self._coordinator.emit(ev, actor="live")

    def _handle_order(self, ev: Any) -> None:
        original_qty = getattr(ev, "qty", getattr(ev, "quantity", None))
        sym = getattr(ev, "symbol", "")
        order_id = getattr(ev, "order_id", "?")

        # Run all gates with audit trail
        result, audit_trail = self._gate_chain.process_with_audit(ev, {})
        if result is None:
            logger.info(
                "ORDER_REJECTED symbol=%s order_id=%s original_qty=%s",
                sym, order_id, original_qty,
            )
            logger.info(
                "ORDER_AUDIT symbol=%s order_id=%s result=rejected gates=%s",
                sym, order_id,
                [{g: {"allowed": r.allowed, "scale": r.scale, "reason": r.reason}} for g, r in audit_trail],
            )
            return
        ev = result

        # Structured audit: log full gate chain transformation
        final_qty = getattr(ev, "qty", getattr(ev, "quantity", None))
        logger.info(
            "ORDER_AUDIT symbol=%s order_id=%s original_qty=%s final_qty=%s gates=%s",
            sym, order_id, original_qty, final_qty,
            [{g: {"allowed": r.allowed, "scale": r.scale, "reason": r.reason}} for g, r in audit_trail],
        )

        # NOTE: OrderStateMachine is an execution AUDIT TRAIL, not a position truth source.
        # Position truth lives in RustStateStore (updated via coordinator pipeline).
        # OSM tracks order lifecycle for: timeout detection, reconciliation, logging,
        # and open-order-count enforcement (RiskGate.max_open_orders).
        # No signal generation or position sizing should read from OSM.
        order_id = getattr(ev, "order_id", None) or getattr(ev, "client_order_id", None)
        if order_id:
            try:
                from decimal import Decimal
                raw_qty = getattr(ev, "qty", getattr(ev, "quantity", 0))
                raw_price = getattr(ev, "price", None)
                self._order_state_machine.register(
                    order_id=str(order_id),
                    client_order_id=getattr(ev, "client_order_id", None),
                    symbol=sym,
                    side=str(getattr(ev, "side", "")),
                    order_type=str(getattr(ev, "order_type", "LIMIT")),
                    qty=Decimal(str(raw_qty)),
                    price=Decimal(str(raw_price)) if raw_price is not None else None,
                )
            except Exception:
                logger.warning("OSM register failed for order %s", order_id, exc_info=True)
            self._timeout_tracker.on_submit(str(order_id), ev)

        self._coordinator.emit(ev, actor="live")

    def _handle_fill(self, ev: Any) -> None:
        # Track fills in state machine + timeout tracker
        order_id = getattr(ev, "order_id", None)
        if order_id:
            self._timeout_tracker.on_fill(str(order_id))
            try:
                from execution.state_machine.transitions import OrderStatus
                from decimal import Decimal
                fill_qty = getattr(ev, "qty", None)
                fill_price = getattr(ev, "price", None)
                self._order_state_machine.transition(
                    order_id=str(order_id),
                    new_status=OrderStatus.FILLED,
                    filled_qty=Decimal(str(fill_qty)) if fill_qty is not None else None,
                    avg_price=Decimal(str(fill_price)) if fill_price is not None else None,
                )
            except Exception:
                logger.debug("OSM transition failed for order %s", order_id, exc_info=True)

        # Record fill event to event_log for replay recovery
        if self._event_recorder is not None:
            self._event_recorder.record_fill(ev)

        # Attribution feedback — live signal tracker
        if self._live_signal_tracker is not None:
            fill_sym = getattr(ev, "symbol", "")
            self._live_signal_tracker.on_fill(ev, origin=fill_sym)
