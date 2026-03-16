# runner/builders/engine_builder.py
"""Phase 6: coordinator and pipeline — coordinator, risk gate, gate chain, emit handler.

Extracted from LiveRunner._build_coordinator_and_pipeline().
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from monitoring.engine_hook import EngineMonitoringHook
from risk.kill_switch import KillSwitch
from runner.recovery import EventRecorder

logger = logging.getLogger(__name__)


def build_coordinator_and_pipeline(
    config: Any,
    symbol_default: str,
    hook: Optional[EngineMonitoringHook],
    feat_hook: Any,
    tick_processors: Optional[Dict[str, Any]],
    _update_correlation: Callable,
    correlation_gate: Any,
    kill_switch: KillSwitch,
    order_state_machine: Any,
    timeout_tracker: Any,
    attribution_tracker: Any,
    live_signal_tracker: Any,
    alpha_health_monitor: Any,
    regime_sizer: Any,
    portfolio_allocator: Any,
    fetch_margin: Optional[Callable],
    report: Any,
) -> tuple:
    """Phase 6: coordinator, risk gate, portfolio aggregator, gate chain, emit handler.

    Returns (coordinator, risk_gate, portfolio_aggregator,
             _emit_handler, _emit, _event_recorder_ref).
    """
    # Event recording: chain onto on_pipeline_output if event_log available
    _event_recorder_ref: List[Optional[EventRecorder]] = [None]

    def _on_pipeline_output_with_recording(out: Any) -> None:
        if hook is not None:
            hook(out)
        rec = _event_recorder_ref[0]
        if rec is not None:
            rec.on_pipeline_output(out)

    coord_cfg = CoordinatorConfig(
        symbol_default=symbol_default,
        symbols=config.symbols,
        currency=config.currency,
        on_pipeline_output=_on_pipeline_output_with_recording,
        on_snapshot=_update_correlation,
        feature_hook=feat_hook,
        tick_processor=tick_processors,
    )
    coordinator = EngineCoordinator(cfg=coord_cfg)

    # ── RiskGate (pre-execution size/notional checks) ────
    from execution.safety.risk_gate import RiskGate, RiskGateConfig
    risk_gate = RiskGate(
        config=RiskGateConfig(),
        get_positions=lambda: coordinator.get_state_view().get("positions", {}),
        get_open_order_count=lambda: len(order_state_machine.active_orders()),
        is_killed=lambda: kill_switch.is_killed() is not None,
    )

    # ── Portfolio Risk Aggregator (Phase 2) ────────────
    portfolio_aggregator = None
    if config.enable_portfolio_risk:
        try:
            from risk.meta_builder_live import build_live_meta_builder
            from risk.aggregator import RiskAggregator
            from risk.rules.portfolio_limits import (
                GrossExposureRule, NetExposureRule, ConcentrationRule,
            )
            from decimal import Decimal

            _equity_source = fetch_margin if fetch_margin is not None else lambda: 10000.0
            meta_builder = build_live_meta_builder(coordinator, equity_source=_equity_source)
            portfolio_aggregator = RiskAggregator(
                rules=[
                    GrossExposureRule(max_gross_leverage=Decimal(str(config.max_gross_leverage))),
                    NetExposureRule(max_net_leverage=Decimal(str(config.max_net_leverage))),
                    ConcentrationRule(max_weight=Decimal(str(config.max_concentration))),
                ],
                meta_builder=meta_builder,
            )
            logger.info(
                "Portfolio risk enabled: gross<=%.1f, net<=%.1f, concentration<=%.1f",
                config.max_gross_leverage, config.max_net_leverage, config.max_concentration,
            )
        except Exception as e:
            report.record("portfolio_risk", False, str(e))
            logger.warning("Portfolio risk setup failed — continuing without", exc_info=True)

    # Build gate chain for ORDER event processing
    from runner.gate_chain import build_gate_chain
    gate_chain = build_gate_chain(
        correlation_gate=correlation_gate,
        risk_gate=risk_gate,
        get_state_view=coordinator.get_state_view,
        portfolio_aggregator=portfolio_aggregator,
        alpha_health_monitor=alpha_health_monitor,
        regime_sizer=regime_sizer,
        portfolio_allocator=portfolio_allocator,
        hook=hook,
        kill_switch=kill_switch,
    )

    from runner.emit_handler import LiveEmitHandler
    _emit_handler = LiveEmitHandler(
        coordinator=coordinator,
        attribution_tracker=attribution_tracker,
        gate_chain=gate_chain,
        order_state_machine=order_state_machine,
        timeout_tracker=timeout_tracker,
        event_recorder=None,  # patched after event_recorder is created
        live_signal_tracker=live_signal_tracker,
    )
    _emit = _emit_handler

    return (
        coordinator, risk_gate, portfolio_aggregator,
        _emit_handler, _emit, _event_recorder_ref,
    )
