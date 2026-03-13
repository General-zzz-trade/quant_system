# runner/builders/execution.py
"""Execution subsystem builder — extracted from LiveRunner.build()."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSubsystem:
    """Assembled execution subsystem components."""
    kill_bridge: Any
    venue_client: Any
    ws_order_gateway: Optional[Any] = None
    execution_bridge: Optional[Any] = None
    decision_bridge: Optional[Any] = None


def build_execution_subsystem(
    config: Any,
    *,
    venue_clients: Dict[str, Any],
    kill_switch: Any,
    coordinator: Any,
    emit_fn: Any,
    decision_modules: Optional[Sequence[Any]] = None,
    latency_tracker: Any = None,
    report: Any = None,
) -> ExecutionSubsystem:
    """Build execution adapter, kill switch bridge, and optional WS gateway."""
    from risk.kill_switch_bridge import KillSwitchBridge
    from engine.decision_bridge import DecisionBridge
    from engine.execution_bridge import ExecutionBridge

    venue_client = venue_clients.get(config.venue)
    if venue_client is None:
        raise ValueError(
            f"No venue client for '{config.venue}'. "
            f"Available: {list(venue_clients.keys())}"
        )

    # WS-API order gateway
    ws_order_gateway = None
    if config.use_ws_orders and not config.shadow_mode:
        try:
            from execution.adapters.binance.ws_order_adapter import WsOrderAdapter
            from execution.adapters.binance.rest import BinanceRestClient as _BRC

            if isinstance(venue_client, _BRC):
                ws_adapter = WsOrderAdapter(
                    rest_adapter=venue_client,
                    api_key=venue_client._cfg.api_key,
                    api_secret=venue_client._cfg.api_secret,
                    testnet=config.testnet,
                )
                ws_adapter.start()
                ws_order_gateway = ws_adapter
                venue_client = ws_adapter
                logger.info("WS-API order gateway enabled (testnet=%s)", config.testnet)
        except Exception as e:
            if report is not None:
                report.record("ws_order_gateway", False, str(e))
            logger.warning("WS order gateway setup failed — using REST", exc_info=True)

    kill_bridge = KillSwitchBridge(
        inner=venue_client,
        kill_switch=kill_switch,
        cancel_fn=getattr(venue_client, "cancel_all_orders", None),
    )

    # Decision + execution bridges
    decision_bridge = None
    execution_bridge = None
    if decision_modules is not None:
        decision_bridge = DecisionBridge(
            dispatcher_emit=emit_fn,
            modules=list(decision_modules),
        )
        execution_bridge = ExecutionBridge(
            adapter=kill_bridge,
            dispatcher_emit=emit_fn,
        )
        coordinator.attach_decision_bridge(decision_bridge)
        coordinator.attach_execution_bridge(execution_bridge)

    return ExecutionSubsystem(
        kill_bridge=kill_bridge,
        venue_client=venue_client,
        ws_order_gateway=ws_order_gateway,
        execution_bridge=execution_bridge,
        decision_bridge=decision_bridge,
    )
