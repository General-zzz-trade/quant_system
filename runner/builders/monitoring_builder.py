# runner/builders/monitoring_builder.py
"""Phase 2: monitoring subsystem — health monitor, monitoring hook, alpha health, regime sizer.

Extracted from LiveRunner._build_monitoring().
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from monitoring.health import SystemHealthMonitor, HealthConfig
from monitoring.engine_hook import EngineMonitoringHook

logger = logging.getLogger(__name__)


def build_monitoring(
    config: Any,
    kill_switch: Any,
    metrics_exporter: Any,
) -> tuple:
    """Phase 2: health monitor, monitoring hook, alpha health, regime sizer, staged risk, signal tracker.

    Returns (health, hook, alpha_health_monitor, regime_sizer, staged_risk, live_signal_tracker).
    """
    health: Optional[SystemHealthMonitor] = None
    hook: Optional[EngineMonitoringHook] = None

    if config.enable_monitoring:
        health = SystemHealthMonitor(
            config=HealthConfig(stale_data_sec=config.health_stale_data_sec),
        )
        hook = EngineMonitoringHook(health=health, metrics=metrics_exporter)

        # ── Drawdown circuit breaker ──
        from risk.drawdown_breaker import DrawdownCircuitBreaker, DrawdownBreakerConfig
        dd_breaker = DrawdownCircuitBreaker(
            kill_switch=kill_switch,
            config=DrawdownBreakerConfig(
                warning_pct=config.dd_warning_pct,
                reduce_pct=config.dd_reduce_pct,
                kill_pct=config.dd_kill_pct,
            ),
        )
        hook.drawdown_breaker = dd_breaker

    # ── Alpha Health Monitor (IC tracking + position scaling) ──
    alpha_health_monitor = None
    if config.enable_alpha_health:
        from monitoring.alpha_health import AlphaHealthMonitor, AlphaHealthConfig

        alpha_health_monitor = AlphaHealthMonitor(
            config=AlphaHealthConfig(),
            prometheus=metrics_exporter,
        )
        for sym in config.symbols:
            alpha_health_monitor.register(sym, horizons=list(config.alpha_health_horizons))
        if hook is not None:
            hook.alpha_health_monitor = alpha_health_monitor
        logger.info(
            "Alpha health monitor enabled: symbols=%s horizons=%s",
            config.symbols, config.alpha_health_horizons,
        )

    # ── Regime Position Sizer (Direction 17) ──
    regime_sizer = None
    if config.enable_regime_sizing:
        from portfolio.regime_sizer import RegimePositionSizer, RegimeSizerConfig
        regime_sizer = RegimePositionSizer(
            config=RegimeSizerConfig(
                low_vol_scale=config.regime_low_vol_scale,
                mid_vol_scale=config.regime_mid_vol_scale,
                high_vol_scale=config.regime_high_vol_scale,
            ),
        )
        logger.info(
            "Regime position sizer enabled: low=%.1f mid=%.1f high=%.1f",
            config.regime_low_vol_scale, config.regime_mid_vol_scale,
            config.regime_high_vol_scale,
        )

    # Wire regime sizer to monitoring hook
    if regime_sizer is not None and hook is not None:
        hook.regime_sizer = regime_sizer

    # ── Staged Risk Manager ──
    staged_risk = None
    if config.enable_regime_sizing:
        from risk.staged_risk import StagedRiskManager
        initial_equity = getattr(config, "initial_equity", 500.0)
        staged_risk = StagedRiskManager(initial_equity=initial_equity)
        logger.info("Staged risk enabled: equity=$%.0f stage=%s",
                     initial_equity, staged_risk.stage.label)

    # ── LiveSignalTracker (Direction 18: attribution feedback) ──
    from attribution.live_tracker import LiveSignalTracker
    live_signal_tracker = LiveSignalTracker(prometheus=metrics_exporter)
    if hook is not None:
        hook.live_signal_tracker = live_signal_tracker

    return health, hook, alpha_health_monitor, regime_sizer, staged_risk, live_signal_tracker
