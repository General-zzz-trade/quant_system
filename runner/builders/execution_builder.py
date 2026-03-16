# runner/builders/execution_builder.py
"""Phase 7: execution — venue client, preflight, WS order gateway, execution bridge.

Extracted from LiveRunner._build_execution().
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from engine.coordinator import EngineCoordinator
from engine.execution_bridge import ExecutionBridge
from risk.kill_switch import KillSwitch
from risk.kill_switch_bridge import KillSwitchBridge

logger = logging.getLogger(__name__)


def build_execution_phase(
    config: Any,
    venue_clients: Dict[str, Any],
    coordinator: EngineCoordinator,
    kill_switch: KillSwitch,
    _emit: Any,
    _record_fill: Callable,
    risk_gate: Any,
    report: Any,
    FillRecordingAdapter: type,
) -> tuple:
    """Phase 7: venue client, preflight, WS order gateway, execution bridge.

    Returns (venue_client, ws_order_gateway).
    """
    # ── 4) Execution adapter: KillSwitchBridge (production) ──
    venue_client = venue_clients.get(config.venue)
    if venue_client is None:
        raise ValueError(
            f"No venue client for '{config.venue}'. "
            f"Available: {list(venue_clients.keys())}"
        )

    # ── 4a) Pre-flight checks ────────────────────────────
    if config.enable_preflight:
        from execution.adapters.binance.rest import BinanceRestClient as _BRC
        if isinstance(venue_client, _BRC):
            from runner.preflight import PreflightChecker, PreflightError
            checker = PreflightChecker(venue_client)
            result = checker.run_all(
                symbols=config.symbols,
                min_balance=config.preflight_min_balance,
            )
            for check in result.checks:
                logger.info(
                    "Preflight %s: %s — %s",
                    "PASS" if check.passed else "FAIL",
                    check.name, check.message,
                )
            if not result.passed:
                raise PreflightError(result)

    # ── 4b) WS-API order gateway (optional fast path) ──────
    ws_order_gateway = None
    if config.use_ws_orders and not config.shadow_mode:
        try:
            from execution.adapters.binance.ws_order_adapter import WsOrderAdapter
            from execution.adapters.binance.rest import BinanceRestClient as _BRCWS

            if isinstance(venue_client, _BRCWS):
                ws_adapter = WsOrderAdapter(
                    rest_adapter=venue_client,
                    api_key=venue_client._cfg.api_key,
                    api_secret=venue_client._cfg.api_secret,
                    testnet=config.testnet,
                )
                ws_adapter.start()
                ws_order_gateway = ws_adapter
                venue_client = ws_adapter  # Replace venue_client with WS-first adapter
                logger.info("WS-API order gateway enabled (testnet=%s)", config.testnet)
            else:
                logger.warning("WS orders require BinanceRestClient — skipping")
        except Exception as e:
            report.record("ws_order_gateway", False, str(e))
            logger.warning("WS order gateway setup failed — using REST", exc_info=True)

    kill_bridge = KillSwitchBridge(
        inner=venue_client,
        kill_switch=kill_switch,
        cancel_fn=getattr(venue_client, "cancel_all_orders", None),
    )

    # Wrap with fill recording: intercept results from send_order
    if config.shadow_mode:
        from execution.sim.shadow_adapter import ShadowExecutionAdapter

        def _shadow_price(sym: str):
            from decimal import Decimal as _Dec
            view = coordinator.get_state_view()
            markets = view.get("markets", {})
            m = markets.get(sym)
            if m is None:
                return None
            cf = getattr(m, "close_f", None)
            if cf is not None:
                return _Dec(str(cf))
            close = getattr(m, "close", None)
            return _Dec(str(close)) if close is not None else None

        exec_adapter = ShadowExecutionAdapter(price_source=_shadow_price)
        logger.warning("SHADOW MODE — orders will be simulated, not executed")
    else:
        exec_adapter = FillRecordingAdapter(inner=kill_bridge, on_fill=_record_fill)
    exec_bridge = ExecutionBridge(adapter=exec_adapter, dispatcher_emit=_emit, risk_gate=risk_gate)
    coordinator.attach_execution_bridge(exec_bridge)

    return venue_client, ws_order_gateway
