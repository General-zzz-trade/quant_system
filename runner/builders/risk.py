# runner/builders/risk.py
"""Risk subsystem builder — extracted from LiveRunner.build()."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskSubsystem:
    """Assembled risk subsystem components."""
    kill_switch: Any
    correlation_computer: Any
    correlation_gate: Any
    risk_gate: Any
    order_state_machine: Any
    timeout_tracker: Any
    portfolio_aggregator: Optional[Any] = None
    alpha_health_monitor: Optional[Any] = None
    regime_sizer: Optional[Any] = None
    portfolio_allocator: Optional[Any] = None


def build_risk_subsystem(
    config: Any,
    *,
    coordinator: Any,
    metrics_exporter: Any = None,
    hook: Any = None,
    fetch_margin: Optional[Callable] = None,
    report: Any = None,
) -> RiskSubsystem:
    """Build all risk-related subsystems.

    Args:
        config: LiveRunnerConfig
        coordinator: EngineCoordinator (for state views)
        metrics_exporter: Optional PrometheusExporter
        hook: Optional EngineMonitoringHook
        fetch_margin: Optional margin ratio callable
        report: Optional _SubsystemReport for tracking
    """
    from risk.kill_switch import KillSwitch
    from risk.correlation_computer import CorrelationComputer
    from risk.correlation_gate import CorrelationGate, CorrelationGateConfig
    from execution.state_machine.machine import OrderStateMachine
    from execution.safety.risk_gate import RiskGate, RiskGateConfig
    from execution.safety.timeout_tracker import OrderTimeoutTracker

    kill_switch = KillSwitch()

    # Alpha health monitor
    alpha_health_monitor = None
    if config.enable_alpha_health:
        from monitoring.alpha_health import AlphaHealthMonitor, AlphaHealthConfig
        alpha_health_monitor = AlphaHealthMonitor(
            config=AlphaHealthConfig(), prometheus=metrics_exporter,
        )
        for sym in config.symbols:
            alpha_health_monitor.register(sym, horizons=list(config.alpha_health_horizons))
        if hook is not None:
            hook.alpha_health_monitor = alpha_health_monitor

    # Regime sizer
    regime_sizer = None
    if config.enable_regime_sizing:
        from portfolio.regime_sizer import RegimePositionSizer, RegimeSizerConfig
        regime_sizer = RegimePositionSizer(config=RegimeSizerConfig(
            low_vol_scale=config.regime_low_vol_scale,
            mid_vol_scale=config.regime_mid_vol_scale,
            high_vol_scale=config.regime_high_vol_scale,
        ))
        if hook is not None:
            hook.regime_sizer = regime_sizer

    # Portfolio allocator
    portfolio_allocator = None
    if config.enable_portfolio_risk:
        from portfolio.live_allocator import LivePortfolioAllocator, LiveAllocatorConfig
        portfolio_allocator = LivePortfolioAllocator(config=LiveAllocatorConfig(
            max_gross_leverage=config.max_gross_leverage,
            max_net_leverage=config.max_net_leverage,
            max_concentration=config.max_concentration,
        ))

    # Correlation
    correlation_computer = CorrelationComputer(window=60)
    correlation_gate = CorrelationGate(
        computer=correlation_computer,
        config=CorrelationGateConfig(max_avg_correlation=config.max_avg_correlation),
    )

    # Order state machine + risk gate
    order_state_machine = OrderStateMachine()
    risk_gate = RiskGate(
        config=RiskGateConfig(),
        get_positions=lambda: coordinator.get_state_view().get("positions", {}),
        get_open_order_count=lambda: len(order_state_machine.active_orders()),
        is_killed=lambda: kill_switch.is_killed() is not None,
    )

    timeout_tracker = OrderTimeoutTracker(
        timeout_sec=config.pending_order_timeout_sec,
    )

    # Portfolio risk aggregator
    portfolio_aggregator = None
    if config.enable_portfolio_risk:
        try:
            from risk.meta_builder_live import build_live_meta_builder
            from risk.aggregator import RiskAggregator
            from risk.rules.portfolio_limits import (
                GrossExposureRule, NetExposureRule, ConcentrationRule,
            )
            from decimal import Decimal

            equity_source = fetch_margin if fetch_margin is not None else lambda: 10000.0
            meta_builder = build_live_meta_builder(coordinator, equity_source=equity_source)
            portfolio_aggregator = RiskAggregator(
                rules=[
                    GrossExposureRule(max_gross_leverage=Decimal(str(config.max_gross_leverage))),
                    NetExposureRule(max_net_leverage=Decimal(str(config.max_net_leverage))),
                    ConcentrationRule(max_weight=Decimal(str(config.max_concentration))),
                ],
                meta_builder=meta_builder,
            )
        except Exception as e:
            if report is not None:
                report.record("portfolio_risk", False, str(e))
            logger.warning("Portfolio risk setup failed", exc_info=True)

    return RiskSubsystem(
        kill_switch=kill_switch,
        correlation_computer=correlation_computer,
        correlation_gate=correlation_gate,
        risk_gate=risk_gate,
        order_state_machine=order_state_machine,
        timeout_tracker=timeout_tracker,
        portfolio_aggregator=portfolio_aggregator,
        alpha_health_monitor=alpha_health_monitor,
        regime_sizer=regime_sizer,
        portfolio_allocator=portfolio_allocator,
    )
