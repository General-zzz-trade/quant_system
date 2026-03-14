# runner/live_runner.py
"""LiveRunner — full production live trading runner.

Assembles:
  - EngineCoordinator + EngineLoop
  - ExecutionBridge (production, via venue_clients)
  - KillSwitchBridge (production kill switch gate)
  - MarginMonitor (production margin ratio + funding rate monitoring)
  - ReconcileScheduler (periodic position/balance reconciliation)
  - GracefulShutdown (SIGTERM/SIGINT handling)
  - SystemHealthMonitor (stale data / drawdown alerts)
  - LatencyTracker (pipeline stage latency)
  - AlertManager (rule-based alerting)
  - SQLite persistent stores (optional)
  - Structured JSON logging (optional)

Usage:
    runner = LiveRunner.build(config, venue_clients={"binance": client}, ...)
    runner.start()  # blocks until stop() or signal
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.loop import EngineLoop, LoopConfig
from engine.guards import build_basic_guard, GuardConfig
from event.types import ControlEvent

from decision.regime_bridge import RegimeAwareDecisionModule
from decision.regime_policy import RegimePolicy

from engine.feature_hook import FeatureComputeHook
from execution.latency.tracker import LatencyTracker
from execution.observability.incidents import timeout_to_alert

from monitoring.alerts.base import Severity
from monitoring.alerts.manager import AlertManager, AlertRule
from monitoring.engine_hook import EngineMonitoringHook
from monitoring.health import SystemHealthMonitor, HealthConfig

from risk.kill_switch import KillMode, KillScope, KillSwitch
from risk.kill_switch_bridge import KillSwitchBridge
from risk.margin_monitor import MarginConfig, MarginMonitor

from runner.config import (
    LiveRunnerConfig,
    OperatorControlRecord,
    OperatorKillSwitchStatus,
    OperatorReconcileStatus,
    OperatorStatusSnapshot,
)
from runner.observability import OperatorObservabilityMixin
from runner.operator_control import OperatorControlMixin
from runner.graceful_shutdown import GracefulShutdown, ShutdownConfig
from runner.recovery import (
    EventRecorder,
    PeriodicCheckpointer,
    reconcile_and_heal,
    restore_all_auxiliary_state,
    save_all_auxiliary_state,
)

logger = logging.getLogger(__name__)


# ── Sub-builder helpers (extracted from build()) ─────────────────────────


def _find_module_attr(decision_bridge: Any, attr: str) -> Any:
    """Walk DecisionBridge.modules to find first module carrying `attr`.

    Handles RegimeAwareDecisionModule wrapping (checks .inner too).
    Returns None if not found or decision_bridge is None.
    """
    if decision_bridge is None:
        return None
    for mod in getattr(decision_bridge, 'modules', []):
        val = getattr(mod, attr, None)
        if val is not None:
            return val
        # Unwrap RegimeAwareDecisionModule
        inner = getattr(mod, 'inner', None)
        if inner is not None:
            val = getattr(inner, attr, None)
            if val is not None:
                return val
    return None


@dataclass
class _SubsystemReport:
    """Structured startup logging: track which subsystems succeeded/failed."""
    succeeded: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    def record(self, name: str, ok: bool, error: str = "") -> None:
        if ok:
            self.succeeded.append(name)
        else:
            self.failed[name] = error

    def log_summary(self) -> None:
        if self.succeeded:
            logger.info(
                "Subsystems OK (%d): %s", len(self.succeeded), ", ".join(self.succeeded),
            )
        if self.failed:
            logger.warning(
                "Subsystems FAILED (%d): %s",
                len(self.failed),
                "; ".join(f"{k}: {v}" for k, v in self.failed.items()),
            )


def _build_multi_tf_ensemble(
    config: "LiveRunnerConfig",
    inference_bridge: Any,
    metrics_exporter: Any,
    report: _SubsystemReport,
) -> Any:
    """Build multi-timeframe ensemble bridges (Direction 13).

    Returns potentially modified inference_bridge (dict of per-symbol bridges).
    """
    try:
        from decision.ensemble_combiner import EnsembleCombiner
        from alpha.inference.bridge import LiveInferenceBridge as _LIB
        import json as _json

        multi_tf = config.multi_tf_models or {}
        combiners: Dict[str, Any] = {}

        for sym, tf_names in multi_tf.items():
            bridges_for_sym = []
            for tf_name in tf_names:
                model_dir = Path(f"models_v8/{sym}_{tf_name}")
                if not model_dir.exists():
                    logger.warning("Multi-TF model dir not found: %s", model_dir)
                    continue

                cfg_path = model_dir / "config.json"
                if not cfg_path.exists():
                    logger.warning("Multi-TF config not found: %s", cfg_path)
                    continue

                with open(cfg_path) as f:
                    model_cfg = _json.load(f)

                from alpha.models.lgbm_alpha import LGBMAlphaModel
                tf_models = []
                for pkl in sorted(model_dir.glob("*.pkl")):
                    try:
                        m = LGBMAlphaModel(name=f"{sym}_{tf_name}_{pkl.stem}")
                        m.load(pkl)
                        tf_models.append(m)
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", pkl, e)
                if not tf_models:
                    logger.warning("No models loaded from %s", model_dir)
                    continue

                tf_min_hold = {sym: model_cfg.get("min_hold", 12)}
                tf_deadzone: Any = {sym: model_cfg.get("deadzone", 0.5)}
                tf_long_only = {sym} if model_cfg.get("long_only") else set()
                tf_bridge = _LIB(
                    models=tf_models,
                    metrics_exporter=metrics_exporter,
                    min_hold_bars=tf_min_hold,
                    deadzone=tf_deadzone,
                    max_hold=model_cfg.get("max_hold", 120),
                    long_only_symbols=tf_long_only,
                    ensemble_weights=model_cfg.get("ensemble_weights"),
                    monthly_gate=config.monthly_gate,
                    monthly_gate_window=config.monthly_gate_window,
                    vol_target=config.vol_target,
                    vol_feature=config.vol_feature,
                )
                bridges_for_sym.append((tf_name, tf_bridge))
                logger.info(
                    "Multi-TF bridge: %s/%s (min_hold=%d, deadzone=%.1f, max_hold=%d)",
                    sym, tf_name,
                    model_cfg.get("min_hold", 12),
                    model_cfg.get("deadzone", 0.5),
                    model_cfg.get("max_hold", 120),
                )

            if len(bridges_for_sym) > 1:
                combiner = EnsembleCombiner(
                    bridges=bridges_for_sym,
                    conflict_policy=config.ensemble_conflict_policy,
                )
                combiners[sym] = combiner
                logger.info(
                    "Ensemble combiner for %s: %d bridges (%s)",
                    sym, len(bridges_for_sym),
                    [n for n, _ in bridges_for_sym],
                )
            elif len(bridges_for_sym) == 1:
                combiners[sym] = bridges_for_sym[0][1]

        if combiners:
            if isinstance(inference_bridge, dict):
                inference_bridge.update(combiners)
            else:
                new_bridge: Dict[str, Any] = {}
                for sym in config.symbols:
                    if sym in combiners:
                        new_bridge[sym] = combiners[sym]
                    else:
                        new_bridge[sym] = inference_bridge
                inference_bridge = new_bridge
            logger.info("Multi-TF ensemble enabled for %d symbols", len(combiners))

        report.record("multi_tf_ensemble", True)
    except Exception as e:
        report.record("multi_tf_ensemble", False, str(e))
        logger.warning("Multi-TF ensemble setup failed - using single bridge", exc_info=True)

    return inference_bridge


def _build_alert_rules(
    alert_manager: Any,
    health: Optional[Any],
    kill_switch: Any,
    latency_tracker: Optional[Any],
    alpha_health_monitor: Optional[Any],
    correlation_computer: Any,
    config: "LiveRunnerConfig",
    report: _SubsystemReport,
) -> None:
    """Add default alert rules to the AlertManager."""
    from monitoring.alerts.base import Severity
    from monitoring.alerts.manager import AlertRule

    # Stale market data
    if health is not None:
        def _stale_data_condition(h=health, cfg=config) -> bool:
            age = h.get_status().data_age_sec
            return age is not None and age > cfg.health_stale_data_sec

        alert_manager.add_rule(AlertRule(
            name="stale_data",
            condition=_stale_data_condition,
            severity=Severity.WARNING,
            message_template="Market data is stale — check feed connectivity",
            cooldown_sec=120.0,
        ))

    # High drawdown (>15%)
    if health is not None:
        def _high_drawdown_condition(h=health) -> bool:
            dd = h.get_status().drawdown_pct
            return dd is not None and dd > 15.0

        alert_manager.add_rule(AlertRule(
            name="high_drawdown",
            condition=_high_drawdown_condition,
            severity=Severity.ERROR,
            message_template="Portfolio drawdown exceeds 15%",
            cooldown_sec=300.0,
        ))

    # Kill switch triggered
    def _kill_switch_condition(ks=kill_switch) -> bool:
        return ks.is_killed() is not None

    alert_manager.add_rule(AlertRule(
        name="kill_switch_triggered",
        condition=_kill_switch_condition,
        severity=Severity.CRITICAL,
        message_template="Kill switch has been triggered — trading halted",
        cooldown_sec=60.0,
    ))

    # Latency SLA breach
    if latency_tracker is not None:
        from execution.latency.report import LatencyReporter
        _reporter = LatencyReporter(latency_tracker)

        def _latency_sla_condition(reporter=_reporter, thresh=config.latency_p99_threshold_ms) -> bool:
            stats = reporter.compute_stats()
            for s in stats:
                if s.metric == "signal_to_fill" and s.count >= 10 and s.p99_ms > thresh:
                    return True
            return False

        alert_manager.add_rule(AlertRule(
            name="latency_sla_breach",
            condition=_latency_sla_condition,
            severity=Severity.WARNING,
            message_template="Latency SLA breach — signal_to_fill P99 exceeds threshold",
            cooldown_sec=300.0,
        ))

    # Alpha health degradation
    if alpha_health_monitor is not None:
        def _alpha_degradation_condition(
            ahm=alpha_health_monitor, syms=config.symbols,
        ) -> bool:
            return any(ahm.position_scale(sym) < 1.0 for sym in syms)

        alert_manager.add_rule(AlertRule(
            name="alpha_degradation",
            condition=_alpha_degradation_condition,
            severity=Severity.WARNING,
            message_template="Alpha health degraded — position scaling active",
            cooldown_sec=3600.0,
        ))

        def _alpha_retrain_needed_condition(
            ahm=alpha_health_monitor, syms=config.symbols,
        ) -> bool:
            return any(ahm.should_retrain(sym) for sym in syms)

        alert_manager.add_rule(AlertRule(
            name="alpha_retrain_needed",
            condition=_alpha_retrain_needed_condition,
            severity=Severity.ERROR,
            message_template="Alpha IC halted — model retraining required",
            cooldown_sec=86400.0,
        ))

    # High portfolio correlation
    def _high_correlation_condition(
        cc=correlation_computer, syms=config.symbols, thresh=config.max_avg_correlation,
    ) -> bool:
        avg = cc.portfolio_avg_correlation(list(syms))
        return avg is not None and avg > thresh

    alert_manager.add_rule(AlertRule(
        name="high_correlation",
        condition=_high_correlation_condition,
        severity=Severity.WARNING,
        message_template="Portfolio avg correlation exceeds threshold",
        cooldown_sec=300.0,
    ))

    report.record("alert_rules", True)


def _build_health_server(
    config: "LiveRunnerConfig",
    health: Any,
    alpha_health_monitor: Optional[Any],
    regime_sizer: Optional[Any],
    portfolio_allocator: Optional[Any],
    live_signal_tracker: Optional[Any],
    report: _SubsystemReport,
) -> tuple:
    """Build health HTTP server and operator control plane.

    Returns (health_server, control_plane).
    """
    if config.health_port is None or health is None:
        return None, None

    from monitoring.health_server import HealthServer
    from dataclasses import asdict as _asdict
    from runner.control_plane import OperatorControlPlane

    _stale_thresh = config.health_stale_data_sec
    health_token = None
    if config.health_auth_token_env:
        health_token = os.environ.get(config.health_auth_token_env)
        if not health_token:
            raise ValueError(
                "health_auth_token_env is set but env var is missing: "
                f"{config.health_auth_token_env}"
            )

    def _health_status_fn() -> Dict[str, Any]:
        st = health.get_status()
        d = _asdict(st)
        age = st.data_age_sec
        if age is not None and age > _stale_thresh:
            d["status"] = "critical"
        if alpha_health_monitor is not None:
            ah_status = {}
            for sym in config.symbols:
                ah_status[sym] = alpha_health_monitor.get_status(sym)
            d["alpha_health"] = ah_status
        if regime_sizer is not None:
            d["regime_sizer"] = regime_sizer.get_status()
        if portfolio_allocator is not None:
            d["portfolio_allocator"] = portfolio_allocator.get_status()
        return d

    control_plane = OperatorControlPlane(SimpleNamespace())
    control_plane.runner = None

    health_server = HealthServer(
        port=config.health_port,
        status_fn=_health_status_fn,
        operator_fn=lambda: control_plane.runner.operator_status() if control_plane.runner is not None else {"error": "runner unavailable"},
        control_history_fn=lambda: [
            {
                "command": rec.command,
                "reason": rec.reason,
                "source": rec.source,
                "result": rec.result,
                "ts": rec.ts.isoformat(),
            }
            for rec in (control_plane.runner.control_history if control_plane.runner is not None else [])
        ],
        control_fn=lambda body: control_plane.execute(body).to_dict() if control_plane.runner is not None else {
            "accepted": False,
            "command": str(body.get("command", "")),
            "outcome": "rejected",
            "reason": str(body.get("reason", "")),
            "source": str(body.get("source", "operator")),
            "status": None,
            "detail": None,
            "error": "runner unavailable",
            "error_code": "runner_unavailable",
        },
        alerts_fn=lambda: control_plane.runner.execution_alert_history() if control_plane.runner is not None else [],
        ops_audit_fn=lambda: control_plane.runner.ops_audit_snapshot() if control_plane.runner is not None else {
            "operator": {"error": "runner unavailable"},
            "control_history": [],
            "execution_alerts": [],
            "model_alerts": [],
            "model_actions": [],
            "model_status": [],
            "timeline": [],
        },
        attribution_fn=lambda: (
            live_signal_tracker.get_status()
            if live_signal_tracker is not None
            else {"error": "attribution tracker unavailable"}
        ),
        host=config.health_host,
        auth_token=health_token,
    )

    report.record("health_server", True)
    return health_server, control_plane


@dataclass
class LiveRunner(OperatorControlMixin, OperatorObservabilityMixin):
    """Full live trading runner with reconciliation, kill switch, and margin monitoring.

    Use LiveRunner.build() to assemble the complete production stack.
    Call start() to begin trading (blocks until stop() or signal).
    """

    loop: EngineLoop
    coordinator: EngineCoordinator
    runtime: Any
    kill_switch: KillSwitch
    health: Optional[SystemHealthMonitor] = None
    reconcile_scheduler: Optional[Any] = None
    margin_monitor: Optional[MarginMonitor] = None
    shutdown_handler: Optional[GracefulShutdown] = None
    latency_tracker: Optional[LatencyTracker] = None
    alert_manager: Optional[AlertManager] = None
    health_server: Optional[Any] = None
    state_store: Optional[Any] = None
    event_log: Optional[Any] = None
    correlation_computer: Optional[Any] = None
    attribution_tracker: Optional[Any] = None
    correlation_gate: Optional[Any] = None
    risk_gate: Optional[Any] = None
    module_reloader: Optional[Any] = None
    decision_bridge: Optional[Any] = None
    user_stream: Optional[Any] = None
    order_state_machine: Optional[Any] = None
    timeout_tracker: Optional[Any] = None
    model_loader: Optional[Any] = None
    inference_bridge: Optional[Any] = None
    portfolio_aggregator: Optional[Any] = None
    data_scheduler: Optional[Any] = None
    freshness_monitor: Optional[Any] = None
    alpha_health_monitor: Optional[Any] = None
    ws_order_gateway: Optional[Any] = None
    ensemble_combiner: Optional[Any] = None  # Direction 13: multi-TF ensemble
    regime_sizer: Optional[Any] = None  # Direction 17: regime-aware sizing
    live_signal_tracker: Optional[Any] = None  # Direction 18: attribution feedback
    portfolio_allocator: Optional[Any] = None  # Direction 19: cross-asset allocator
    periodic_checkpointer: Optional[PeriodicCheckpointer] = None
    event_recorder: Optional[EventRecorder] = None
    _fills: List[Dict[str, Any]] = field(default_factory=list)
    _control_history: List[OperatorControlRecord] = field(default_factory=list)
    _running: bool = field(default=False, init=False)
    _stopped: bool = field(default=False, init=False)
    _reload_models_pending: bool = field(default=False, init=False)
    _user_stream_thread: Optional[Any] = field(default=None, init=False)
    _user_stream_failure_count: int = field(default=0, init=False)
    _last_user_stream_failure_at: Optional[datetime] = field(default=None, init=False)
    _last_user_stream_failure_kind: Optional[str] = field(default=None, init=False)
    _last_model_reload_status: Optional[Dict[str, Any]] = field(default=None, init=False)

    @staticmethod
    def _build_core_infra(
        config: LiveRunnerConfig,
        on_fill: Optional[Callable[[Any], None]],
        alert_sink: Optional[Any],
        fills: List[Dict[str, Any]],
    ) -> tuple:
        """Phase 1: structured logging, latency tracker, fill recorder, kill switch, alert sink."""
        # ── Auto-wire Telegram alerts from env vars ──────────
        if alert_sink is None:
            tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
            if tg_token and tg_chat:
                from monitoring.alerts.telegram import TelegramAlertSink
                from monitoring.alerts.base import CompositeAlertSink
                from monitoring.alerts.console import ConsoleAlertSink
                alert_sink = CompositeAlertSink(sinks=[
                    ConsoleAlertSink(),
                    TelegramAlertSink(tg_token, tg_chat),
                ])
                logger.info("Telegram alerts auto-wired (chat_id=%s)", tg_chat)

        # ── 0) Structured logging ─────────────────────────────
        if config.enable_structured_logging:
            from infra.logging.structured import setup_structured_logging
            setup_structured_logging(
                level=config.log_level,
                log_file=config.log_file,
            )

        # ── 1) LatencyTracker ─────────────────────────────────
        latency_tracker = LatencyTracker()

        def _record_fill(fill: Any) -> None:
            fills.append({
                "ts": str(getattr(fill, "ts", "")),
                "symbol": str(getattr(fill, "symbol", "")),
                "side": str(getattr(fill, "side", "")),
                "qty": str(getattr(fill, "qty", "")),
                "price": str(getattr(fill, "price", "")),
            })
            order_id = getattr(fill, "order_id", None)
            if order_id:
                latency_tracker.record_fill(str(order_id))
            if on_fill is not None:
                on_fill(fill)

        # ── 2) KillSwitch ────────────────────────────────────
        kill_switch = KillSwitch()

        return latency_tracker, _record_fill, kill_switch, alert_sink

    @staticmethod
    def _build_monitoring(
        config: LiveRunnerConfig,
        kill_switch: KillSwitch,
        metrics_exporter: Any,
    ) -> tuple:
        """Phase 2: health monitor, monitoring hook, alpha health, regime sizer, signal tracker."""
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

        # ── LiveSignalTracker (Direction 18: attribution feedback) ──
        from attribution.live_tracker import LiveSignalTracker
        live_signal_tracker = LiveSignalTracker(prometheus=metrics_exporter)
        if hook is not None:
            hook.live_signal_tracker = live_signal_tracker

        return health, hook, alpha_health_monitor, regime_sizer, live_signal_tracker

    @staticmethod
    def _build_portfolio_and_correlation(config: LiveRunnerConfig) -> tuple:
        """Phase 3: portfolio allocator, burn-in gate, correlation, attribution."""
        # ── Portfolio Allocator (Direction 19) ──
        portfolio_allocator = None
        if config.enable_portfolio_risk:
            from portfolio.live_allocator import LivePortfolioAllocator, LiveAllocatorConfig
            portfolio_allocator = LivePortfolioAllocator(
                config=LiveAllocatorConfig(
                    max_gross_leverage=config.max_gross_leverage,
                    max_net_leverage=config.max_net_leverage,
                    max_concentration=config.max_concentration,
                ),
            )
            logger.info(
                "Portfolio allocator enabled: gross=%.1f net=%.1f concentration=%.1f",
                config.max_gross_leverage, config.max_net_leverage,
                config.max_concentration,
            )

        # ── Burn-in Gate (Direction 14) ──
        if config.enable_burnin_gate and not config.testnet:
            from runner.preflight import BurninGate
            burnin_gate = BurninGate(report_path=config.burnin_report_path)
            burnin_check = burnin_gate.check(testnet=config.testnet)
            if not burnin_check.passed:
                raise RuntimeError(
                    f"Burn-in gate FAILED: {burnin_check.message}\n"
                    "Complete paper→shadow→testnet phases before production."
                )
            logger.info("Burn-in gate passed: %s", burnin_check.message)

        # ── CorrelationComputer (created early for on_snapshot) ──
        from risk.correlation_computer import CorrelationComputer
        correlation_computer = CorrelationComputer(window=60)

        def _update_correlation(snapshot: Any) -> None:
            markets = getattr(snapshot, "markets", {})
            for sym, mkt in markets.items():
                close = getattr(mkt, "close", None)
                if close is not None:
                    correlation_computer.update(sym, float(close))

        # ── AttributionTracker ───────────────────────────────
        from attribution.tracker import AttributionTracker
        attribution_tracker = AttributionTracker()

        # ── CorrelationGate ──────────────────────────────────
        from risk.correlation_gate import CorrelationGate, CorrelationGateConfig
        correlation_gate = CorrelationGate(
            computer=correlation_computer,
            config=CorrelationGateConfig(max_avg_correlation=config.max_avg_correlation),
        )

        return (
            portfolio_allocator, correlation_computer, _update_correlation,
            attribution_tracker, correlation_gate,
        )

    @staticmethod
    def _build_order_infra(config: LiveRunnerConfig, alpha_models: Any) -> tuple:
        """Phase 4: order state machine, timeout tracker, model registry."""
        # ── OrderStateMachine (order lifecycle tracking) ────
        from execution.state_machine.machine import OrderStateMachine
        order_state_machine = OrderStateMachine()

        # ── TimeoutTracker (stale order detection) ──────────
        from execution.safety.timeout_tracker import OrderTimeoutTracker
        timeout_tracker = OrderTimeoutTracker(
            timeout_sec=config.pending_order_timeout_sec,
        )

        # ── ModelRegistry auto-loading (Phase 1) ──────────
        model_loader_inst = None
        if config.model_registry_db and config.model_names:
            from research.model_registry.registry import ModelRegistry
            from research.model_registry.artifact import ArtifactStore
            from alpha.model_loader import ProductionModelLoader

            registry = ModelRegistry(config.model_registry_db)
            artifact_store = ArtifactStore(config.artifact_store_root or "artifacts")
            model_loader_inst = ProductionModelLoader(registry, artifact_store)
            loaded = model_loader_inst.load_production_models(config.model_names)
            if loaded:
                alpha_models = list(alpha_models or []) + loaded
                logger.info("Auto-loaded %d production model(s) from registry", len(loaded))

        return order_state_machine, timeout_tracker, model_loader_inst, alpha_models

    @staticmethod
    def _build_features_and_inference(
        config: LiveRunnerConfig,
        feature_computer: Any,
        alpha_models: Any,
        inference_bridges: Optional[Dict[str, Any]],
        unified_predictors: Optional[Dict[str, Any]],
        metrics_exporter: Any,
        hook: Optional[EngineMonitoringHook],
        report: _SubsystemReport,
        bear_model: Any,
        funding_rate_source: Any,
        oi_source: Any,
        ls_ratio_source: Any,
        spot_close_source: Any,
        fgi_source: Any,
        implied_vol_source: Any,
        put_call_ratio_source: Any,
        onchain_source: Any,
        liquidation_source: Any,
        mempool_source: Any,
        macro_source: Any,
        sentiment_source: Any,
    ) -> tuple:
        """Phase 5: feature hook, inference bridge, multi-TF ensemble, decision recording."""
        feat_hook = None
        inference_bridge = None
        if feature_computer is not None:
            if inference_bridges is not None:
                # Multi-symbol: per-symbol bridges already constructed
                inference_bridge = inference_bridges
            elif alpha_models:
                from alpha.inference.bridge import LiveInferenceBridge
                inference_bridge = LiveInferenceBridge(
                    models=list(alpha_models),
                    metrics_exporter=metrics_exporter,
                    min_hold_bars=config.min_hold_bars,
                    long_only_symbols=config.long_only_symbols,
                    deadzone=config.deadzone,
                    trend_follow=config.trend_follow,
                    trend_indicator=config.trend_indicator,
                    trend_threshold=config.trend_threshold,
                    max_hold=config.max_hold,
                    monthly_gate=config.monthly_gate,
                    monthly_gate_window=config.monthly_gate_window,
                    bear_model=bear_model,
                    bear_thresholds=config.bear_thresholds,
                    vol_target=config.vol_target,
                    vol_feature=config.vol_feature,
                    ensemble_weights=config.ensemble_weights,
                )

            # ── Multi-timeframe Ensemble (Direction 13) ──────────────
            if config.enable_multi_tf_ensemble and inference_bridge is not None:
                inference_bridge = _build_multi_tf_ensemble(
                    config, inference_bridge, metrics_exporter, report,
                )

            feat_hook = FeatureComputeHook(
                computer=feature_computer,
                inference_bridge=inference_bridge if unified_predictors is None else None,
                unified_predictor=unified_predictors,
                funding_rate_source=funding_rate_source,
                oi_source=oi_source,
                ls_ratio_source=ls_ratio_source,
                spot_close_source=spot_close_source,
                fgi_source=fgi_source,
                implied_vol_source=implied_vol_source,
                put_call_ratio_source=put_call_ratio_source,
                onchain_source=onchain_source,
                liquidation_source=liquidation_source,
                mempool_source=mempool_source,
                macro_source=macro_source,
                sentiment_source=sentiment_source,
            )

        # Wire inference_bridge to monitoring hook
        _first_bridge = None
        if inference_bridge is not None:
            if isinstance(inference_bridge, dict):
                _first_bridge = next(iter(inference_bridge.values()), None)
            else:
                _first_bridge = inference_bridge
        if hook is not None and _first_bridge is not None:
            hook.inference_bridge = _first_bridge

        # Decision recording (replay support)
        if config.enable_decision_recording and hook is not None:
            from decision.persistence.decision_store import DecisionStore
            hook.decision_store = DecisionStore(path=config.decision_recording_path)
            logger.info("Decision recording enabled: %s", config.decision_recording_path)

        return feat_hook, inference_bridge

    @staticmethod
    def _build_coordinator_and_pipeline(
        config: LiveRunnerConfig,
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
        report: _SubsystemReport,
    ) -> tuple:
        """Phase 6: coordinator, risk gate, portfolio aggregator, gate chain, emit handler."""
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

    @staticmethod
    def _build_execution(
        config: LiveRunnerConfig,
        venue_clients: Dict[str, Any],
        coordinator: EngineCoordinator,
        kill_switch: KillSwitch,
        _emit: Any,
        _record_fill: Callable,
        risk_gate: Any,
        report: _SubsystemReport,
    ) -> tuple:
        """Phase 7: venue client, preflight, WS order gateway, execution bridge."""
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
            exec_adapter = _FillRecordingAdapter(inner=kill_bridge, on_fill=_record_fill)
        exec_bridge = ExecutionBridge(adapter=exec_adapter, dispatcher_emit=_emit, risk_gate=risk_gate)
        coordinator.attach_execution_bridge(exec_bridge)

        return venue_client, ws_order_gateway

    @staticmethod
    def _build_decision(
        config: LiveRunnerConfig,
        decision_modules: Sequence[Any] | None,
        _emit: Any,
        coordinator: EngineCoordinator,
    ) -> tuple:
        """Phase 8: decision bridge with regime wrapping, module reloader, engine loop."""
        # ── 5) Decision bridge ────────────────────────────────
        modules = list(decision_modules or [])

        if config.enable_regime_gate and modules:
            gated_modules = []
            for mod in modules:
                gated = RegimeAwareDecisionModule(
                    inner=mod,
                    policy=RegimePolicy(),
                )
                gated_modules.append(gated)
            modules = gated_modules

        decision_bridge_inst = None
        if modules:
            decision_bridge_inst = DecisionBridge(
                dispatcher_emit=_emit, modules=modules,
            )
            coordinator.attach_decision_bridge(decision_bridge_inst)

        # ── 5a) ModuleReloader ───────────────────────────────
        from engine.module_reloader import ModuleReloader, ReloaderConfig
        module_reloader = ModuleReloader(
            config=ReloaderConfig(),
            on_reload=lambda trigger: logger.info("Module reload triggered: %s", trigger),
        )

        # ── 6) EngineLoop with guard ─────────────────────────
        guard = build_basic_guard(GuardConfig())
        loop = EngineLoop(coordinator=coordinator, guard=guard, cfg=LoopConfig())

        return decision_bridge_inst, module_reloader, loop

    @staticmethod
    def _build_market_data(
        config: LiveRunnerConfig,
        transport: Any,
        venue_client: Any,
        loop: EngineLoop,
    ) -> tuple:
        """Phase 9: market data runtime (WS + REST fallback)."""
        from execution.adapters.binance.kline_processor import KlineProcessor
        from execution.adapters.binance.ws_market_stream_um import (
            BinanceUmMarketStreamWsClient,
            MarketStreamConfig,
        )
        from execution.adapters.binance.market_data_runtime import BinanceMarketDataRuntime
        from execution.adapters.binance.urls import resolve_binance_urls

        if config.testnet:
            logger.warning("*** TESTNET MODE — NOT PRODUCTION ***")

        binance_urls = resolve_binance_urls(config.testnet)

        if transport is None:
            from execution.adapters.binance.transport_factory import create_ws_transport
            transport = create_ws_transport()

        ws_url = config.ws_base_url
        if config.testnet:
            ws_url = binance_urls.ws_market_stream

        streams = tuple(
            f"{sym.lower()}@kline_{config.kline_interval}"
            for sym in config.symbols
        )
        processor = KlineProcessor(source="binance.ws.kline")
        ws_client = BinanceUmMarketStreamWsClient(
            transport=transport,
            processor=processor,
            streams=streams,
            cfg=MarketStreamConfig(ws_base_url=ws_url),
        )
        from execution.adapters.binance.rest_kline_source import RestKlineSource
        rest_base = (
            getattr(venue_client, '_cfg', None) and venue_client._cfg.base_url
            or binance_urls.rest_base
        )
        rest_fallback = RestKlineSource(
            base_url=rest_base,
            source="binance.rest.kline",
        )
        runtime = BinanceMarketDataRuntime(
            ws_client=ws_client,
            rest_fallback=rest_fallback,
            symbols=config.symbols,
            kline_interval=config.kline_interval,
        )
        loop.attach_runtime(runtime)

        return runtime, binance_urls

    @staticmethod
    def _build_user_stream(
        config: LiveRunnerConfig,
        venue_client: Any,
        coordinator: EngineCoordinator,
        binance_urls: Any,
        user_stream_transport: Any,
        report: _SubsystemReport,
    ) -> Any:
        """Phase 10: user stream (private fill/order feed)."""
        if not config.shadow_mode:
            from execution.adapters.binance.rest import BinanceRestClient as _BRC2
            if isinstance(venue_client, _BRC2):
                try:
                    from execution.adapters.binance.listen_key_um import BinanceUmListenKeyClient
                    from execution.adapters.binance.listen_key_manager import (
                        BinanceUmListenKeyManager, ListenKeyManagerConfig,
                    )
                    from execution.adapters.binance.ws_user_stream_um import (
                        BinanceUmUserStreamWsClient, UserStreamWsConfig,
                    )
                    from execution.adapters.binance.user_stream_processor_um import (
                        BinanceUmUserStreamProcessor,
                    )
                    from execution.adapters.binance.mapper_fill import BinanceFillMapper
                    from execution.adapters.binance.mapper_order import BinanceOrderMapper
                    from execution.ingress.router import FillIngressRouter
                    from execution.ingress.order_router import OrderIngressRouter

                    class _TimeClock:
                        def now(self) -> float:
                            return time.time()

                    fill_router = FillIngressRouter(
                        coordinator=coordinator, default_actor="venue:binance",
                    )
                    order_router = OrderIngressRouter(
                        coordinator=coordinator, default_actor="venue:binance",
                    )
                    us_processor = BinanceUmUserStreamProcessor(
                        order_router=order_router,
                        fill_router=fill_router,
                        order_mapper=BinanceOrderMapper(),
                        fill_mapper=BinanceFillMapper(),
                        default_actor="venue:binance",
                    )
                    lk_client = BinanceUmListenKeyClient(rest=venue_client)
                    lk_mgr = BinanceUmListenKeyManager(
                        client=lk_client,
                        clock=_TimeClock(),
                        cfg=ListenKeyManagerConfig(validity_sec=3600, renew_margin_sec=300),
                    )

                    us_transport = user_stream_transport
                    if us_transport is None:
                        from execution.adapters.binance.transport_factory import create_ws_transport as _cwt
                        us_transport = _cwt()

                    user_stream_client = BinanceUmUserStreamWsClient(
                        transport=us_transport,
                        listen_key_mgr=lk_mgr,
                        processor=us_processor,
                        cfg=UserStreamWsConfig(
                            ws_base_url=binance_urls.ws_user_stream,
                        ),
                    )
                    logger.info(
                        "User stream wired (url_base=%s)", binance_urls.ws_user_stream,
                    )
                    return user_stream_client
                except Exception as e:
                    report.record("user_stream", False, str(e))
                    logger.warning("User stream setup failed — continuing without", exc_info=True)
                    return None
            else:
                return None
        else:
            return None

    @staticmethod
    def _build_persistence_and_recovery(
        config: LiveRunnerConfig,
        coordinator: EngineCoordinator,
        kill_switch: KillSwitch,
        inference_bridge: Any,
        feat_hook: Any,
        correlation_computer: Any,
        timeout_tracker: Any,
        decision_bridge_inst: Any,
        fetch_venue_state: Optional[Callable],
        fetch_margin: Optional[Callable],
        alert_sink: Any,
        health: Optional[SystemHealthMonitor],
        latency_tracker: LatencyTracker,
        alpha_health_monitor: Any,
        report: _SubsystemReport,
    ) -> tuple:
        """Phase 11: reconcile, margin, alerts, persistent stores, recovery, data scheduler."""
        # ── 8) ReconcileScheduler ────────────────────────────
        reconcile_scheduler = None
        if config.enable_reconcile and fetch_venue_state is not None:
            from execution.reconcile.controller import ReconcileController
            from execution.reconcile.scheduler import (
                ReconcileScheduler,
                ReconcileSchedulerConfig,
            )

            reconcile_scheduler = ReconcileScheduler(
                controller=ReconcileController(),
                get_local_state=lambda: coordinator.get_state_view(),
                fetch_venue_state=fetch_venue_state,
                cfg=ReconcileSchedulerConfig(
                    interval_sec=config.reconcile_interval_sec,
                    venue=config.venue,
                ),
                on_halt=lambda report: coordinator.stop(),
            )

        # ── 9) MarginMonitor (production) ────────────────────
        margin_monitor = None
        if fetch_margin is not None:
            margin_monitor = MarginMonitor(
                config=MarginConfig(
                    check_interval_sec=config.margin_check_interval_sec,
                    warning_margin_ratio=config.margin_warning_ratio,
                    critical_margin_ratio=config.margin_critical_ratio,
                ),
                fetch_margin=fetch_margin,
                kill_switch=kill_switch,
                alert_sink=alert_sink,
            )

        # ── 10) AlertManager + default rules ────────────────
        alert_manager = AlertManager(sink=alert_sink)
        _build_alert_rules(
            alert_manager, health, kill_switch, latency_tracker,
            alpha_health_monitor, correlation_computer, config, report,
        )

        # ── 11) Persistent stores (conditional) ─────────────
        state_store = None
        event_log = None
        if config.enable_persistent_stores:
            from execution.store.ack_store import SQLiteAckStore
            from execution.store.event_log import SQLiteEventLog
            from state.store import SqliteStateStore

            data_dir = config.data_dir
            SQLiteAckStore(path=os.path.join(data_dir, "ack_store.db"))
            event_log = SQLiteEventLog(path=os.path.join(data_dir, "event_log.db"))
            state_store = SqliteStateStore(
                path=os.path.join(data_dir, "state.db"),
            )

            # State restoration: restore from latest checkpoint
            for sym in config.symbols:
                checkpoint = state_store.latest(sym)
                if checkpoint is not None:
                    coordinator.restore_from_snapshot(checkpoint.snapshot)
                    logger.info(
                        "Restored state for %s from bar_index=%d",
                        sym, checkpoint.bar_index,
                    )
                    break  # One restore is enough (snapshot contains all symbols)

            # Recovery: restore all auxiliary state (atomic bundle with individual fallback)
            _exit_mgr = _find_module_attr(decision_bridge_inst, '_exit_mgr')
            _regime_gate = _find_module_attr(decision_bridge_inst, '_regime_gate')
            restore_results = restore_all_auxiliary_state(
                kill_switch=kill_switch,
                inference_bridge=inference_bridge,
                feature_hook=feat_hook,
                exit_manager=_exit_mgr,
                regime_gate=_regime_gate,
                correlation_computer=correlation_computer,
                timeout_tracker=timeout_tracker,
                data_dir=data_dir,
            )
            for comp_name, restored in restore_results.items():
                if restored:
                    logger.info("Recovery: %s state restored from checkpoint", comp_name)

        # ── 11a) Startup reconciliation with healing ─────────
        if config.reconcile_on_startup and fetch_venue_state is not None:
            try:
                venue_state = fetch_venue_state()
                # First: detect-only pass for logging
                local_view = coordinator.get_state_view()
                mismatches = _reconcile_startup(local_view, venue_state, config.symbols)
                for m in mismatches:
                    logger.warning("Startup reconciliation mismatch: %s", m)
                # Second: heal mismatches by updating local state
                if mismatches:
                    actions = reconcile_and_heal(coordinator, venue_state, config.symbols)
                    for a in actions:
                        logger.warning("Reconciliation action: %s", a)
            except Exception:
                logger.exception(
                    "Startup reconciliation failed — proceeding with local state"
                )

        # ── 11b) DataScheduler + FreshnessMonitor (Phase 3) ─
        data_scheduler = None
        freshness_monitor = None
        if config.enable_data_scheduler:
            try:
                from data.scheduler.data_scheduler import DataScheduler, DataSchedulerConfig
                from data.scheduler.freshness_monitor import FreshnessMonitor, FreshnessConfig

                data_scheduler = DataScheduler(DataSchedulerConfig(symbols=config.symbols))
                freshness_monitor = FreshnessMonitor(FreshnessConfig(
                    data_dir=config.data_files_dir,
                    symbols=config.symbols,
                    on_alert=lambda a: logger.warning(
                        "Data stale: %s age=%.1fh", a.source, a.age_hours,
                    ),
                ))
                logger.info("DataScheduler + FreshnessMonitor configured")
            except Exception as e:
                report.record("data_scheduler", False, str(e))
                logger.warning("DataScheduler setup failed — continuing without", exc_info=True)

        return (
            reconcile_scheduler, margin_monitor, alert_manager,
            state_store, event_log, data_scheduler, freshness_monitor,
        )

    @staticmethod
    def _build_shutdown(
        config: LiveRunnerConfig,
        state_store: Any,
        coordinator: EngineCoordinator,
        kill_switch: KillSwitch,
        inference_bridge: Any,
        feat_hook: Any,
        decision_bridge_inst: Any,
        correlation_computer: Any,
        timeout_tracker: Any,
        reconcile_scheduler: Any,
        venue_client: Any,
        health: Optional[SystemHealthMonitor],
        alpha_health_monitor: Any,
        regime_sizer: Any,
        portfolio_allocator: Any,
        live_signal_tracker: Any,
        event_log: Any,
        report: _SubsystemReport,
    ) -> tuple:
        """Phase 12: graceful shutdown, health server, periodic checkpointer, event recorder."""
        # ── 12) GracefulShutdown ─────────────────────────────
        shutdown_cfg = ShutdownConfig(
            pending_order_timeout_sec=config.pending_order_timeout_sec,
        )
        save_snapshot_fn = None
        if state_store is not None:
            def save_snapshot_fn(_path: str) -> None:
                data_dir_s = config.data_dir
                snapshot = coordinator.get_state_view().get("last_snapshot")
                if snapshot is not None:
                    state_store.save(snapshot)
                    logger.info("State snapshot saved on shutdown")
                # Persist all auxiliary state atomically
                save_all_auxiliary_state(
                    kill_switch=kill_switch,
                    inference_bridge=inference_bridge,
                    feature_hook=feat_hook,
                    exit_manager=_find_module_attr(decision_bridge_inst, '_exit_mgr'),
                    regime_gate=_find_module_attr(decision_bridge_inst, '_regime_gate'),
                    correlation_computer=correlation_computer,
                    timeout_tracker=timeout_tracker,
                    data_dir=data_dir_s,
                )

        # wait_pending: returns True when no active orders remain
        def _wait_pending() -> bool:
            return timeout_tracker.pending_count == 0

        # cancel_all: cancel all open orders on the venue
        def _cancel_all() -> None:
            if hasattr(venue_client, "cancel_all_orders"):
                for sym in config.symbols:
                    try:
                        venue_client.cancel_all_orders(sym)
                    except Exception:
                        logger.warning("cancel_all_orders failed for %s", sym, exc_info=True)

        # reconcile: run one reconciliation pass
        def _reconcile_once() -> None:
            if reconcile_scheduler is not None:
                try:
                    reconcile_scheduler.run_once()
                except Exception:
                    logger.warning("Shutdown reconciliation failed", exc_info=True)

        # Use a mutable container for late-binding cleanup: the runner
        # instance doesn't exist yet, so we capture a list and patch it
        # after construction (see below).
        _runner_ref: List[Any] = []

        def _cleanup() -> None:
            if _runner_ref:
                _runner_ref[0]._running = False

        shutdown_handler = GracefulShutdown(
            config=shutdown_cfg,
            stop_new_orders=lambda: kill_switch.trigger(
                scope=KillScope.GLOBAL,
                key="*",
                mode=KillMode.HARD_KILL,
                reason="graceful_shutdown",
                source="shutdown",
            ),
            wait_pending=_wait_pending,
            cancel_all=_cancel_all,
            reconcile=_reconcile_once,
            save_snapshot=save_snapshot_fn,
            cleanup=_cleanup,
        )

        # ── 13) Health HTTP endpoint (optional) ────────────────
        health_server, control_plane = _build_health_server(
            config, health, alpha_health_monitor, regime_sizer,
            portfolio_allocator, live_signal_tracker, report,
        )

        # ── 14) Recovery infrastructure ────────────────────────
        # Periodic checkpointer: saves state every 60s (not just on shutdown)
        periodic_checkpointer = None
        if state_store is not None:
            def _get_snapshot() -> Any:
                return coordinator.get_state_view().get("last_snapshot")

            periodic_checkpointer = PeriodicCheckpointer(
                state_store=state_store,
                get_snapshot=_get_snapshot,
                interval_sec=config.reconcile_interval_sec,  # reuse reconcile interval
            )

        # Event recorder: captures market/fill events to event_log
        event_recorder = None
        if event_log is not None:
            event_recorder = EventRecorder(event_log)

        return (
            shutdown_handler, health_server, control_plane,
            periodic_checkpointer, event_recorder, _runner_ref,
        )

    @classmethod
    def build(
        cls,
        config: LiveRunnerConfig,
        *,
        venue_clients: Dict[str, Any],
        decision_modules: Sequence[Any] | None = None,
        transport: Any = None,
        metrics_exporter: Any = None,
        fetch_venue_state: Optional[Callable[[], Dict[str, Any]]] = None,
        fetch_margin: Optional[Callable[[], float]] = None,
        on_fill: Optional[Callable[[Any], None]] = None,
        alert_sink: Optional[Any] = None,
        feature_computer: Any = None,
        alpha_models: Sequence[Any] | None = None,
        inference_bridges: Optional[Dict[str, Any]] = None,
        unified_predictors: Optional[Dict[str, Any]] = None,
        tick_processors: Optional[Dict[str, Any]] = None,
        user_stream_transport: Any = None,
        funding_rate_source: Any = None,
        oi_source: Any = None,
        ls_ratio_source: Any = None,
        spot_close_source: Any = None,
        fgi_source: Any = None,
        implied_vol_source: Any = None,
        put_call_ratio_source: Any = None,
        onchain_source: Any = None,
        liquidation_source: Any = None,
        mempool_source: Any = None,
        macro_source: Any = None,
        sentiment_source: Any = None,
        bear_model: Any = None,
    ) -> "LiveRunner":
        """Build the full production stack.

        Args:
            config: Runner configuration.
            venue_clients: Mapping of venue name to venue client object.
                           The client for config.venue is used as the execution adapter.
            decision_modules: Decision modules (strategy, risk, etc.).
            transport: WsTransport override (for testing).
            metrics_exporter: Optional PrometheusExporter.
            fetch_venue_state: Callable returning exchange state dict for reconciliation.
            fetch_margin: Callable returning current margin ratio (0.0-1.0).
            on_fill: Optional fill callback.
            alert_sink: Optional AlertSink for health/margin alerts.
        """
        symbol_default = config.symbols[0]
        fills: List[Dict[str, Any]] = []
        report = _SubsystemReport()

        # Phase 1: core infrastructure
        latency_tracker, _record_fill, kill_switch, alert_sink = cls._build_core_infra(
            config, on_fill, alert_sink, fills,
        )

        # Phase 2: monitoring
        health, hook, alpha_health_monitor, regime_sizer, live_signal_tracker = (
            cls._build_monitoring(config, kill_switch, metrics_exporter)
        )

        # Phase 3: portfolio and correlation
        (
            portfolio_allocator, correlation_computer, _update_correlation,
            attribution_tracker, correlation_gate,
        ) = cls._build_portfolio_and_correlation(config)

        # Phase 4: order infrastructure
        order_state_machine, timeout_tracker, model_loader_inst, alpha_models = (
            cls._build_order_infra(config, alpha_models)
        )

        # Phase 5: features and inference
        feat_hook, inference_bridge = cls._build_features_and_inference(
            config, feature_computer, alpha_models, inference_bridges,
            unified_predictors, metrics_exporter, hook, report,
            bear_model,
            funding_rate_source, oi_source, ls_ratio_source,
            spot_close_source, fgi_source, implied_vol_source,
            put_call_ratio_source, onchain_source, liquidation_source,
            mempool_source, macro_source, sentiment_source,
        )

        # Phase 6: coordinator and pipeline
        (
            coordinator, risk_gate, portfolio_aggregator,
            _emit_handler, _emit, _event_recorder_ref,
        ) = cls._build_coordinator_and_pipeline(
            config, symbol_default, hook, feat_hook, tick_processors,
            _update_correlation, correlation_gate, kill_switch,
            order_state_machine, timeout_tracker, attribution_tracker,
            live_signal_tracker, alpha_health_monitor, regime_sizer,
            portfolio_allocator, fetch_margin, report,
        )

        # Phase 7: execution
        venue_client, ws_order_gateway = cls._build_execution(
            config, venue_clients, coordinator, kill_switch,
            _emit, _record_fill, risk_gate, report,
        )

        # Phase 8: decision bridge and engine loop
        decision_bridge_inst, module_reloader, loop = cls._build_decision(
            config, decision_modules, _emit, coordinator,
        )

        # Phase 9: market data runtime
        runtime, binance_urls = cls._build_market_data(
            config, transport, venue_client, loop,
        )

        # Phase 10: user stream
        user_stream_client = cls._build_user_stream(
            config, venue_client, coordinator, binance_urls,
            user_stream_transport, report,
        )

        # Phase 11: persistence and recovery
        (
            reconcile_scheduler, margin_monitor, alert_manager,
            state_store, event_log, data_scheduler, freshness_monitor,
        ) = cls._build_persistence_and_recovery(
            config, coordinator, kill_switch, inference_bridge,
            feat_hook, correlation_computer, timeout_tracker,
            decision_bridge_inst, fetch_venue_state, fetch_margin,
            alert_sink, health, latency_tracker, alpha_health_monitor,
            report,
        )

        # Phase 12: shutdown, health server, checkpointer
        (
            shutdown_handler, health_server, control_plane,
            periodic_checkpointer, event_recorder, _runner_ref,
        ) = cls._build_shutdown(
            config, state_store, coordinator, kill_switch,
            inference_bridge, feat_hook, decision_bridge_inst,
            correlation_computer, timeout_tracker, reconcile_scheduler,
            venue_client, health, alpha_health_monitor, regime_sizer,
            portfolio_allocator, live_signal_tracker, event_log, report,
        )

        runner = cls(
            loop=loop,
            coordinator=coordinator,
            runtime=runtime,
            kill_switch=kill_switch,
            health=health,
            reconcile_scheduler=reconcile_scheduler,
            margin_monitor=margin_monitor,
            shutdown_handler=shutdown_handler,
            latency_tracker=latency_tracker,
            alert_manager=alert_manager,
            health_server=health_server,
            state_store=state_store,
            event_log=event_log,
            correlation_computer=correlation_computer,
            attribution_tracker=attribution_tracker,
            correlation_gate=correlation_gate,
            risk_gate=risk_gate,
            module_reloader=module_reloader,
            decision_bridge=decision_bridge_inst,
            user_stream=user_stream_client,
            order_state_machine=order_state_machine,
            timeout_tracker=timeout_tracker,
            model_loader=model_loader_inst,
            inference_bridge=inference_bridge,
            ensemble_combiner=inference_bridge if config.enable_multi_tf_ensemble else None,
            portfolio_aggregator=portfolio_aggregator,
            data_scheduler=data_scheduler,
            freshness_monitor=freshness_monitor,
            alpha_health_monitor=alpha_health_monitor,
            ws_order_gateway=ws_order_gateway,
            regime_sizer=regime_sizer,
            live_signal_tracker=live_signal_tracker,
            portfolio_allocator=portfolio_allocator,
            periodic_checkpointer=periodic_checkpointer,
            event_recorder=event_recorder,
            _fills=fills,
        )
        if control_plane is not None:
            control_plane.runner = runner
        # Patch late-binding reference so cleanup callback can stop the runner
        _runner_ref.append(runner)
        runner._config = config
        # Patch event recorder into pipeline output hook and emit handler
        if event_recorder is not None:
            _event_recorder_ref[0] = event_recorder
            _emit_handler._event_recorder = event_recorder
        report.log_summary()
        return runner

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        *,
        venue_clients: Dict[str, Any],
        decision_modules: Sequence[Any] | None = None,
        transport: Any = None,
        metrics_exporter: Any = None,
        fetch_venue_state: Optional[Callable[[], Dict[str, Any]]] = None,
        fetch_margin: Optional[Callable[[], float]] = None,
        on_fill: Optional[Callable[[Any], None]] = None,
        alert_sink: Optional[Any] = None,
        shadow_mode: bool = False,
        feature_computer: Any = None,
        inference_bridges: Optional[Dict[str, Any]] = None,
    ) -> "LiveRunner":
        """Build a LiveRunner from a YAML/JSON config file.

        Supports two config formats:
          - Flat: keys map 1:1 to LiveRunnerConfig fields (production.yaml)
          - Nested: trading.symbol, risk.*, execution.* sections (legacy)

        When feature_computer and inference_bridges are not provided,
        auto-discovers and loads models from models_v8/.
        """
        import dataclasses
        from infra.config.loader import load_config_secure, resolve_credentials

        raw = load_config_secure(config_path)

        # ── Detect config format: flat vs nested ──
        is_flat = "symbols" in raw or "venue" in raw
        is_nested = "trading" in raw

        if is_flat:
            # Flat format: keys map directly to LiveRunnerConfig fields
            config_fields = {f.name for f in dataclasses.fields(LiveRunnerConfig)}
            kwargs: Dict[str, Any] = {}
            for k, v in raw.items():
                if k in config_fields:
                    kwargs[k] = v
            # Ensure symbols is a tuple
            if "symbols" in kwargs:
                kwargs["symbols"] = tuple(kwargs["symbols"])
            # Apply shadow_mode override from CLI
            kwargs["shadow_mode"] = shadow_mode
            runner_config = LiveRunnerConfig(**kwargs)
        elif is_nested:
            # Legacy nested format: validate against nested schema
            from infra.config.schema import validate_trading_config
            errors = validate_trading_config(raw)
            if errors:
                raise ValueError(
                    f"Config validation failed ({config_path}):\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )
            trading = raw.get("trading", {})
            risk = raw.get("risk", {})
            monitoring = raw.get("monitoring", {})
            log_cfg = raw.get("logging", {})

            symbol = trading.get("symbol", "BTCUSDT")
            symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

            kwargs = {}
            if risk.get("max_leverage") is not None:
                kwargs["margin_warning_ratio"] = float(risk["max_leverage"])
            if risk.get("max_drawdown_pct") is not None:
                kwargs["margin_critical_ratio"] = float(risk["max_drawdown_pct"]) / 100.0
            if monitoring.get("health_check_interval") is not None:
                kwargs["health_stale_data_sec"] = float(monitoring["health_check_interval"])
            if monitoring.get("health_port") is not None:
                kwargs["health_port"] = int(monitoring["health_port"])
            if monitoring.get("health_host") is not None:
                kwargs["health_host"] = str(monitoring["health_host"])
            if monitoring.get("health_auth_token_env") is not None:
                kwargs["health_auth_token_env"] = str(monitoring["health_auth_token_env"])

            runner_config = LiveRunnerConfig(
                symbols=symbols,
                venue=trading.get("exchange", "binance"),
                enable_structured_logging=log_cfg.get("structured", True),
                log_level=log_cfg.get("level", "INFO"),
                log_file=log_cfg.get("file"),
                shadow_mode=shadow_mode,
                testnet=bool(trading.get("testnet", False)),
                **kwargs,
            )
        else:
            raise ValueError(
                f"Config format not recognized ({config_path}): "
                "expected flat keys (symbols, venue) or nested sections (trading, risk)"
            )

        resolve_credentials(raw)

        # ── Auto-discover and load models if not provided ──
        if feature_computer is None or inference_bridges is None:
            from runner.model_discovery import (
                discover_active_models,
                load_symbol_models,
                build_inference_bridge,
                build_feature_computer,
            )

            if feature_computer is None:
                feature_computer = build_feature_computer()

            if inference_bridges is None:
                active = discover_active_models()
                bridges: Dict[str, Any] = {}
                for sym in runner_config.symbols:
                    if sym in active:
                        info = active[sym]
                        models, weights = load_symbol_models(
                            sym, info["dir"], info["config"],
                        )
                        if models:
                            bridges[sym] = build_inference_bridge(
                                sym, models, info["config"], runner_config,
                                metrics_exporter=metrics_exporter,
                                ensemble_weights=weights,
                            )
                        else:
                            logger.warning("No models loaded for %s — no inference bridge", sym)
                    else:
                        logger.warning("No active model found for %s in models_v8/", sym)
                inference_bridges = bridges if bridges else None

        return cls.build(
            runner_config,
            venue_clients=venue_clients,
            decision_modules=decision_modules,
            transport=transport,
            metrics_exporter=metrics_exporter,
            fetch_venue_state=fetch_venue_state,
            fetch_margin=fetch_margin,
            on_fill=on_fill,
            alert_sink=alert_sink,
            feature_computer=feature_computer,
            inference_bridges=inference_bridges,
        )

    @staticmethod
    def _apply_perf_tuning() -> None:
        """Apply OS-level performance tuning for low-latency trading."""
        import os as _os
        try:
            nohz_cpus = set()
            try:
                with open("/sys/devices/system/cpu/nohz_full") as f:
                    for part in f.read().strip().split(","):
                        part = part.strip()
                        if not part or part == "(null)":
                            continue
                        if "-" in part:
                            lo, hi = part.split("-")
                            if lo.isdigit() and hi.isdigit():
                                nohz_cpus.update(range(int(lo), int(hi) + 1))
                        elif part.isdigit():
                            nohz_cpus.add(int(part))
            except FileNotFoundError:
                pass

            if nohz_cpus:
                _os.sched_setaffinity(0, nohz_cpus)
                logger.info("CPU affinity pinned to nohz_full cores: %s", nohz_cpus)

            _os.nice(-10)
            logger.info("Process priority raised (nice=-10)")
        except (OSError, PermissionError) as e:
            logger.warning("Performance tuning partially failed: %s", e)

    def _apply_attribution_feedback(self) -> None:
        """Apply attribution-based weight adjustments to ensemble combiner (Direction 18)."""
        tracker = self.live_signal_tracker
        if tracker is None or not hasattr(tracker, 'compute_weight_recommendations'):
            return

        recommendations = tracker.compute_weight_recommendations(
            alpha_health_monitor=self.alpha_health_monitor,
        )

        combiner = self.ensemble_combiner
        if combiner is None:
            return

        # Apply to all combiners (may be dict of per-symbol combiners)
        combiners = combiner.values() if isinstance(combiner, dict) else [combiner]
        for c in combiners:
            if hasattr(c, 'update_weight'):
                for origin, weight_mult in recommendations.items():
                    if weight_mult < 1.0:
                        c.update_weight(origin, weight_mult)
                        logger.info(
                            "Attribution feedback: %s weight -> %.2f",
                            origin, weight_mult,
                        )

    def start(self) -> None:
        """Start the live trading system. Blocks until stop() or signal."""
        self._stopped = False
        self._running = True

        self._apply_perf_tuning()

        # ── Systemd watchdog notify (Direction 20) ──
        _sd_notify_fn = None
        try:
            import socket
            _sd_addr = os.environ.get("NOTIFY_SOCKET")
            if _sd_addr:
                _sd_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                if _sd_addr.startswith("@"):
                    _sd_addr = "\0" + _sd_addr[1:]

                def _sd_notify_fn(msg: str) -> None:
                    try:
                        _sd_sock.sendto(msg.encode(), _sd_addr)
                    except Exception:
                        pass

                _sd_notify_fn("READY=1")
                logger.info("Systemd notify: READY=1")
        except Exception:
            pass

        if self.shutdown_handler is not None:
            self.shutdown_handler.install_handlers()

        self.coordinator.start()
        if self.health is not None:
            self.health.start()
        if self.reconcile_scheduler is not None:
            self.reconcile_scheduler.start()
        if self.margin_monitor is not None:
            self.margin_monitor.start()
        if self.alert_manager is not None:
            self.alert_manager.start_periodic()
        if self.health_server is not None:
            self.health_server.start()
        if self.module_reloader is not None:
            self.module_reloader.start()
        if self.data_scheduler is not None:
            self.data_scheduler.start()
        if self.freshness_monitor is not None:
            self.freshness_monitor.start()
        if self.periodic_checkpointer is not None:
            self.periodic_checkpointer.start()

        # SIGHUP: schedule model reload on next main loop iteration
        # Works with both ModelRegistry and direct model file reload
        import signal as _signal
        import threading as _threading
        def _sighup_handler(signum: int, frame: Any) -> None:
            logger.info("SIGHUP received — scheduling model reload")
            self._reload_models_pending = True
        try:
            if _threading.current_thread() is _threading.main_thread():
                _signal.signal(_signal.SIGHUP, _sighup_handler)
            else:
                logger.warning("Skipping LiveRunner SIGHUP handler: not running in main thread")
        except (OSError, AttributeError, ValueError):
            pass

        self.runtime.start()

        if self.user_stream is not None:
            import threading

            def _user_stream_loop() -> None:
                try:
                    self.user_stream.connect()
                    self._record_user_stream_connect()
                except Exception:
                    self._record_user_stream_failure(kind="connect")
                    logger.warning("User stream initial connect failed", exc_info=True)
                    return
                while self._running:
                    try:
                        self.user_stream.step()
                    except Exception:
                        self._record_user_stream_failure(kind="step")
                        logger.warning("User stream step error, reconnecting in 1s", exc_info=True)
                        time.sleep(1.0)
                        try:
                            self.user_stream.connect()
                            self._record_user_stream_connect()
                        except Exception:
                            self._record_user_stream_failure(kind="reconnect")
                            logger.warning("User stream reconnect failed", exc_info=True)

            t = threading.Thread(target=_user_stream_loop, daemon=True, name="user-stream")
            t.start()
            self._user_stream_thread = t
            logger.info("User stream thread started")

        self.loop.start_background()

        # ── Adaptive BTC config selector state ──
        _last_adaptive_check = 0.0
        _adaptive_selector = None
        cfg = getattr(self, "_config", None)
        if cfg is not None and cfg.adaptive_btc_enabled and "BTCUSDT" in cfg.symbols:
            try:
                from alpha.adaptive_config import AdaptiveConfigSelector
                _adaptive_selector = AdaptiveConfigSelector()
                logger.info("Adaptive BTC config selector enabled (interval=%dh)", cfg.adaptive_btc_interval_hours)
            except Exception:
                logger.warning("Adaptive config selector init failed", exc_info=True)

        logger.info("LiveRunner started. Press Ctrl+C to stop.")
        try:
            while self._running:
                time.sleep(1.0)
                # Check for timed-out orders
                if self.timeout_tracker is not None:
                    timed_out = self.timeout_tracker.check_timeouts()
                    if timed_out:
                        logger.warning("Timed out orders: %s", timed_out)
                        venue = str(getattr(getattr(self, "_config", None), "venue", ""))
                        timeout_sec = float(getattr(self.timeout_tracker, "timeout_sec", 0.0))
                        for order_id in timed_out:
                            try:
                                self._emit_execution_incident(
                                    timeout_to_alert(
                                        venue=venue,
                                        symbol="*",
                                        order_id=str(order_id),
                                        timeout_sec=timeout_sec,
                                    )
                                )
                            except Exception:
                                logger.exception("timeout alert emit failed for order=%s", order_id)
                if self._reload_models_pending:
                    self._reload_models_pending = False
                    self._handle_model_reload()
                # ── Adaptive BTC config check (periodic) ──
                if _adaptive_selector is not None:
                    now = time.time()
                    interval_sec = cfg.adaptive_btc_interval_hours * 3600
                    if now - _last_adaptive_check >= interval_sec:
                        _last_adaptive_check = now
                        self._run_adaptive_btc_check(_adaptive_selector)

                # ── Attribution feedback loop (Direction 18) ──
                if (self.live_signal_tracker is not None
                        and self.ensemble_combiner is not None):
                    now = time.time()
                    if not hasattr(self, '_last_attr_feedback'):
                        self._last_attr_feedback = now
                    if now - self._last_attr_feedback >= 3600:  # hourly
                        self._last_attr_feedback = now
                        self._apply_attribution_feedback()

                # ── Systemd watchdog notify (Direction 20) ──
                if _sd_notify_fn is not None:
                    _sd_notify_fn("WATCHDOG=1")

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop all subsystems gracefully."""
        if self._stopped:
            return
        self._stopped = True
        self._running = False

        logger.info("Stopping LiveRunner...")
        if self.user_stream is not None:
            try:
                self.user_stream.close()
            except Exception:
                logger.warning("User stream close error", exc_info=True)
            if self._user_stream_thread is not None:
                from infra.threading_utils import safe_join_thread

                safe_join_thread(self._user_stream_thread, timeout=5.0)
                self._user_stream_thread = None
        if self.periodic_checkpointer is not None:
            self.periodic_checkpointer.stop()
        if self.freshness_monitor is not None:
            self.freshness_monitor.stop()
        if self.data_scheduler is not None:
            self.data_scheduler.stop()
        if self.module_reloader is not None:
            self.module_reloader.stop()
        if self.health_server is not None:
            self.health_server.stop()
        if self.alert_manager is not None:
            self.alert_manager.stop()
        if self.margin_monitor is not None:
            self.margin_monitor.stop()
        if self.reconcile_scheduler is not None:
            self.reconcile_scheduler.stop()
        self.runtime.stop()
        self.loop.stop_background()
        self.coordinator.stop()
        if self.health is not None:
            self.health.stop()

        if self.ws_order_gateway is not None:
            try:
                self.ws_order_gateway.stop()
            except Exception:
                logger.warning("WS order gateway stop error", exc_info=True)

        logger.info("LiveRunner stopped. Total fills: %d", len(self._fills))

    def _run_adaptive_btc_check(self, selector: Any) -> None:
        """Run adaptive config selection for BTC and update inference bridge params."""
        try:
            import numpy as np
            import pandas as pd

            data_path = "data_files/BTCUSDT_1h.csv"
            df = pd.read_csv(data_path)
            if len(df) < 720:
                logger.warning("Adaptive BTC: insufficient data (%d rows)", len(df))
                return

            closes = df["close"].values.astype(np.float64)

            # Compute z-scores from close returns (simple approximation)
            returns = np.diff(np.log(closes))
            window = 720
            if len(returns) < window:
                return
            rolling_mean = pd.Series(returns).rolling(window).mean().values
            rolling_std = pd.Series(returns).rolling(window).std().values
            z_scores = np.where(rolling_std > 1e-10, (returns - rolling_mean) / rolling_std, 0.0)

            result = selector.select_robust(z_scores, closes[1:])

            if result.confidence != "high":
                logger.info(
                    "Adaptive BTC: confidence=%s (need 'high'), keeping fixed params. "
                    "sharpe=%.2f trades=%d",
                    result.confidence, result.sharpe, result.trades,
                )
                return

            # Update inference bridge params for BTCUSDT only
            bridge = self.inference_bridge
            if bridge is None:
                return
            if isinstance(bridge, dict):
                bridge = bridge.get("BTCUSDT")
            if bridge is None or not hasattr(bridge, "update_params"):
                return

            bridge.update_params(
                "BTCUSDT",
                deadzone=result.deadzone,
                min_hold=result.min_hold,
                max_hold=result.max_hold,
                long_only=result.long_only,
            )
            logger.info(
                "Adaptive BTC applied: deadzone=%.1f min_hold=%d max_hold=%d "
                "long_only=%s sharpe=%.2f confidence=%s",
                result.deadzone, result.min_hold, result.max_hold,
                result.long_only, result.sharpe, result.confidence,
            )
        except Exception:
            logger.warning("Adaptive BTC check failed", exc_info=True)

    def _handle_model_reload(self) -> None:
        """Handle SIGHUP model reload via ModelRegistry or direct file reload."""
        cfg = getattr(self, '_config', None)

        # Path 1: ModelRegistry-based reload
        if self.model_loader is not None:
            try:
                names = tuple(cfg.model_names) if cfg and cfg.model_names else ()
                new_models = self.model_loader.reload_if_changed(names)
                if new_models is not None and self.inference_bridge is not None:
                    self.inference_bridge.update_models(new_models)
                    self._record_model_reload(
                        outcome="reloaded",
                        model_names=names,
                        detail={"reloaded_count": len(new_models)},
                    )
                elif new_models is None:
                    self._record_model_reload(
                        outcome="noop",
                        model_names=names,
                        detail={"reloaded_count": 0},
                    )
            except Exception:
                names = tuple(cfg.model_names) if cfg and cfg.model_names else ()
                self._record_model_reload(
                    outcome="failed",
                    model_names=names,
                    detail=None,
                    error="model_hot_reload_failed",
                )
                logger.exception("Model hot-reload failed")
            return

        # Path 2: Direct file reload (for auto_retrain.py SIGHUP)
        if self.inference_bridge is not None:
            try:
                from alpha.models.lgbm_alpha import LGBMAlphaModel
                from pathlib import Path as _Path

                symbols = tuple(cfg.symbols) if cfg else ()
                models = []
                for sym in symbols:
                    model_dir = _Path(f"models_v8/{sym}_gate_v2")
                    if not model_dir.exists():
                        continue
                    pkl_files = sorted(model_dir.glob("*.pkl"))
                    for pkl in pkl_files:
                        m = LGBMAlphaModel(name=f"{sym}_{pkl.stem}")
                        m.load(pkl)
                        models.append(m)
                if models:
                    bridge = self.inference_bridge
                    if isinstance(bridge, dict):
                        for sym in symbols:
                            b = bridge.get(sym)
                            if b is not None:
                                sym_models = [m for m in models if sym in m.name]
                                if sym_models:
                                    b.update_models(sym_models)
                    else:
                        bridge.update_models(models)
                    self._record_model_reload(
                        outcome="reloaded",
                        model_names=symbols,
                        detail={"reloaded_count": len(models), "source": "file_reload"},
                    )
                    logger.info("Direct model reload: %d model(s) from disk", len(models))
                else:
                    self._record_model_reload(
                        outcome="noop",
                        model_names=symbols,
                        detail={"reloaded_count": 0, "source": "file_reload"},
                    )
            except Exception:
                symbols = tuple(cfg.symbols) if cfg else ()
                self._record_model_reload(
                    outcome="failed",
                    model_names=symbols,
                    detail=None,
                    error="direct_file_reload_failed",
                )
                logger.exception("Direct model file reload failed")

    # halt, reduce_only, resume, flush, shutdown, apply_control
    # → OperatorControlMixin (runner/operator_control.py)

    # operator_status_snapshot, operator_status, execution_alert_history,
    # model_alert_history, ops_timeline, ops_audit_snapshot, control_history,
    # fills, event_index, _record_control, _record_user_stream_connect,
    # _record_user_stream_failure, _stream_status, _incident_state,
    # _recommended_action, _last_incident, _record_model_reload,
    # _emit_execution_incident, _emit_control_alert
    # → OperatorObservabilityMixin (runner/observability.py)





def _reconcile_startup(
    local_view: Dict[str, Any],
    venue_state: Dict[str, Any],
    symbols: tuple[str, ...],
) -> List[str]:
    """Compare local state against exchange state. Returns list of mismatch descriptions."""
    mismatches: List[str] = []

    venue_positions = venue_state.get("positions", {})
    local_positions = local_view.get("positions", {})

    for sym in symbols:
        local_pos = local_positions.get(sym)
        venue_pos = venue_positions.get(sym)

        local_qty = float(getattr(local_pos, "qty", 0) if local_pos else 0)
        venue_qty = float(venue_pos.get("qty", 0) if isinstance(venue_pos, dict) else 0)

        if abs(local_qty - venue_qty) > 1e-8:
            mismatches.append(
                f"{sym} position: local={local_qty}, venue={venue_qty}"
            )

    local_account = local_view.get("account")
    local_balance = float(getattr(local_account, "balance", 0) if local_account else 0)
    venue_balance = float(venue_state.get("balance", 0))
    if abs(local_balance - venue_balance) > 0.01:
        mismatches.append(
            f"Balance: local={local_balance:.2f}, venue={venue_balance:.2f}"
        )

    return mismatches


class _FillRecordingAdapter:
    """Thin wrapper that intercepts fill events from send_order results."""

    def __init__(self, inner: Any, on_fill: Callable[[Any], None]) -> None:
        self._inner = inner
        self._on_fill = on_fill

    def send_order(self, order_event: Any) -> list:
        results = list(self._inner.send_order(order_event))
        for ev in results:
            et = getattr(getattr(ev, "event_type", None), "value", "")
            if "fill" in str(et).lower():
                self._on_fill(ev)
        return results


if __name__ == "__main__":
    import gc
    gc.set_threshold(50_000, 50, 10)

    # Pin to isolated CPU1 + mlock all memory
    try:
        os.sched_setaffinity(0, {1})
    except OSError:
        pass
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.mlockall(3)  # MCL_CURRENT | MCL_FUTURE
    except OSError:
        pass

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Live trading runner")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    parser.add_argument("--shadow", action="store_true", help="Shadow mode — simulate orders")
    args = parser.parse_args()

    # Venue clients must be constructed from config credentials
    from infra.config.loader import load_config_secure, resolve_credentials

    raw = load_config_secure(args.config)
    creds = resolve_credentials(raw)

    venue_clients: Dict[str, Any] = {}
    exchange = raw.get("venue", raw.get("trading", {}).get("exchange", "binance"))
    testnet = bool(raw.get("testnet", raw.get("trading", {}).get("testnet", False)))

    if exchange == "binance":
        from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
        from execution.adapters.binance.urls import resolve_binance_urls

        binance_urls = resolve_binance_urls(testnet)
        client = BinanceRestClient(
            cfg=BinanceRestConfig(
                base_url=binance_urls.rest_base,
                api_key=creds.get("api_key", ""),
                api_secret=creds.get("api_secret", ""),
            )
        )
        venue_clients["binance"] = client

    runner = LiveRunner.from_config(
        args.config,
        venue_clients=venue_clients,
        shadow_mode=getattr(args, 'shadow', False),
    )
    runner.start()
