# runner/live_runner.py
"""LiveRunner -- framework live trading runner.

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

This module is the framework live runtime truth source.
It is not the current default host trading service for Bybit directional alpha
or Bybit market making; see docs/runtime_truth.md.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from execution.latency.tracker import LatencyTracker
from execution.observability.incidents import timeout_to_alert  # noqa: F401
from monitoring.alerts.manager import AlertManager
from monitoring.health import SystemHealthMonitor
from risk.kill_switch import KillSwitch
from risk.margin_monitor import MarginMonitor

from runner.config import (
    LiveRunnerConfig,
    OperatorControlRecord,
)
from runner.observability import OperatorObservabilityMixin
from runner.operator_control import OperatorControlMixin
from runner.graceful_shutdown import GracefulShutdown
from runner.recovery import (
    EventRecorder,
    PeriodicCheckpointer,
)
from runner.live_runner_helpers import (
    build_config_from_file,
    auto_discover_models,
    handle_model_reload,
    run_adaptive_btc_check,
    apply_attribution_feedback,
)

# Phase builders (extracted to runner/builders/)
from runner.builders.core_infra_builder import build_core_infra as _build_core_infra
from runner.builders.monitoring_builder import build_monitoring as _build_monitoring
from runner.builders.portfolio_builder import build_portfolio_and_correlation as _build_portfolio_and_correlation
from runner.builders.order_infra_builder import build_order_infra as _build_order_infra
from runner.builders.features_builder import build_features_and_inference as _build_features_and_inference
from runner.builders.engine_builder import build_coordinator_and_pipeline as _build_coordinator_and_pipeline
from runner.builders.execution_builder import build_execution_phase as _build_execution
from runner.builders.decision_builder import build_decision as _build_decision
from runner.builders.market_data_builder import build_market_data as _build_market_data
from runner.builders.user_stream_builder import build_user_stream as _build_user_stream
from runner.builders.persistence import build_persistence_and_recovery as _persistence_builder
from runner.builders.shutdown import build_shutdown as _shutdown_builder
from runner.builders.rust_components_builder import build_rust_components as _build_rust_components

logger = logging.getLogger(__name__)


# -- Sub-builder helpers (used by builders) --
from runner.live_runner_builder_helpers import _find_module_attr, _SubsystemReport  # noqa: E402, F401


@dataclass
class LiveRunner(OperatorControlMixin, OperatorObservabilityMixin):
    """Full live trading runner with reconciliation, kill switch, and margin monitoring."""

    loop: Any  # EngineLoop
    coordinator: Any  # EngineCoordinator
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
    ensemble_combiner: Optional[Any] = None
    regime_sizer: Optional[Any] = None
    staged_risk: Optional[Any] = None
    live_signal_tracker: Optional[Any] = None
    portfolio_allocator: Optional[Any] = None
    periodic_checkpointer: Optional[PeriodicCheckpointer] = None
    event_recorder: Optional[EventRecorder] = None
    ack_store: Optional[Any] = None
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
    _lifecycle_lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

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
        """Build the full framework live stack."""
        symbol_default = config.symbols[0]
        fills: List[Dict[str, Any]] = []
        report = _SubsystemReport()

        latency_tracker, _record_fill, kill_switch, alert_sink = _build_core_infra(
            config, on_fill, alert_sink, fills,
        )
        rust = _build_rust_components(config, config.symbols)
        health, hook, alpha_health_monitor, regime_sizer, staged_risk, live_signal_tracker = (
            _build_monitoring(config, kill_switch, metrics_exporter)
        )
        (
            portfolio_allocator, correlation_computer, _update_correlation,
            attribution_tracker, correlation_gate,
        ) = _build_portfolio_and_correlation(config)
        order_state_machine, timeout_tracker, model_loader_inst, alpha_models = (
            _build_order_infra(config, alpha_models)
        )
        feat_hook, inference_bridge, _dominance_computer = _build_features_and_inference(
            config, feature_computer, alpha_models, inference_bridges,
            unified_predictors, metrics_exporter, hook, report,
            bear_model,
            funding_rate_source, oi_source, ls_ratio_source,
            spot_close_source, fgi_source, implied_vol_source,
            put_call_ratio_source, onchain_source, liquidation_source,
            mempool_source, macro_source, sentiment_source,
        )
        (
            coordinator, risk_gate, portfolio_aggregator,
            _emit_handler, _emit, _event_recorder_ref,
        ) = _build_coordinator_and_pipeline(
            config, symbol_default, hook, feat_hook, tick_processors,
            _update_correlation, correlation_gate, kill_switch,
            order_state_machine, timeout_tracker, attribution_tracker,
            live_signal_tracker, alpha_health_monitor, regime_sizer,
            staged_risk, portfolio_allocator, fetch_margin, report,
        )
        venue_client, ws_order_gateway = _build_execution(
            config, venue_clients, coordinator, kill_switch,
            _emit, _record_fill, risk_gate, report, _FillRecordingAdapter,
        )
        decision_bridge_inst, module_reloader, loop = _build_decision(
            config, decision_modules, _emit, coordinator,
        )
        runtime, binance_urls = _build_market_data(
            config, transport, venue_client, loop,
        )
        user_stream_client = _build_user_stream(
            config, venue_client, coordinator, binance_urls,
            user_stream_transport, report,
        )
        (
            reconcile_scheduler, margin_monitor, alert_manager,
            state_store, event_log, data_scheduler, freshness_monitor,
            ack_store,
        ) = _persistence_builder(
            config, coordinator, kill_switch, inference_bridge,
            feat_hook, correlation_computer, timeout_tracker,
            decision_bridge_inst, fetch_venue_state, fetch_margin,
            alert_sink, health, latency_tracker, alpha_health_monitor,
            report,
        )
        (
            shutdown_handler, health_server, control_plane,
            periodic_checkpointer, event_recorder, _runner_ref,
        ) = _shutdown_builder(
            config, state_store, coordinator, kill_switch,
            inference_bridge, feat_hook, decision_bridge_inst,
            correlation_computer, timeout_tracker, reconcile_scheduler,
            venue_client, health, alpha_health_monitor, regime_sizer,
            portfolio_allocator, live_signal_tracker, event_log, report,
        )

        runner = cls(
            loop=loop, coordinator=coordinator, runtime=runtime, kill_switch=kill_switch,
            health=health, reconcile_scheduler=reconcile_scheduler,
            margin_monitor=margin_monitor, shutdown_handler=shutdown_handler,
            latency_tracker=latency_tracker, alert_manager=alert_manager,
            health_server=health_server, state_store=state_store, event_log=event_log,
            correlation_computer=correlation_computer,
            attribution_tracker=attribution_tracker, correlation_gate=correlation_gate,
            risk_gate=risk_gate, module_reloader=module_reloader,
            decision_bridge=decision_bridge_inst, user_stream=user_stream_client,
            order_state_machine=order_state_machine, timeout_tracker=timeout_tracker,
            model_loader=model_loader_inst, inference_bridge=inference_bridge,
            ensemble_combiner=inference_bridge if config.enable_multi_tf_ensemble else None,
            portfolio_aggregator=portfolio_aggregator,
            data_scheduler=data_scheduler, freshness_monitor=freshness_monitor,
            alpha_health_monitor=alpha_health_monitor,
            ws_order_gateway=ws_order_gateway, regime_sizer=regime_sizer,
            staged_risk=staged_risk, live_signal_tracker=live_signal_tracker,
            portfolio_allocator=portfolio_allocator,
            periodic_checkpointer=periodic_checkpointer,
            event_recorder=event_recorder, ack_store=ack_store, _fills=fills,
        )
        if control_plane is not None:
            control_plane.runner = runner
        _runner_ref.append(runner)
        runner._config = config
        runner._rust_components = rust
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
        """Build a LiveRunner from a YAML/JSON config file."""
        runner_config, _raw = build_config_from_file(
            config_path, shadow_mode=shadow_mode,
        )

        if feature_computer is None or inference_bridges is None:
            feature_computer, inference_bridges = auto_discover_models(
                runner_config,
                feature_computer=feature_computer,
                inference_bridges=inference_bridges,
                metrics_exporter=metrics_exporter,
            )

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
        apply_attribution_feedback(self)

    def start(self) -> None:
        """Start the live trading system. Blocks until stop() or signal."""
        _sd_notify_fn = None
        try:
            with self._lifecycle_lock:
                self._stopped = False
                self._running = True
                self._apply_perf_tuning()
                _sd_notify_fn = _setup_systemd_notify()

                if self.shutdown_handler is not None:
                    self.shutdown_handler.install_handlers()
                self.coordinator.start()
                _start_optional(self.health)
                _start_optional(self.reconcile_scheduler)
                _start_optional(self.margin_monitor)
                if self.alert_manager is not None:
                    self.alert_manager.start_periodic()
                _start_optional(self.health_server)
                _start_optional(self.module_reloader)
                _start_optional(self.data_scheduler)
                _start_optional(self.freshness_monitor)
                _start_optional(self.periodic_checkpointer)

                _install_sighup(self)
                self.runtime.start()
                _start_user_stream(self)
                self.loop.start_background()

            # Adaptive BTC config selector state
            _last_adaptive_check = 0.0
            _adaptive_selector = None
            cfg = getattr(self, "_config", None)
            if cfg is not None and cfg.adaptive_btc_enabled and "BTCUSDT" in cfg.symbols:
                try:
                    from alpha.adaptive_config import AdaptiveConfigSelector
                    _adaptive_selector = AdaptiveConfigSelector()
                    logger.info("Adaptive BTC config selector enabled (interval=%dh)",
                                cfg.adaptive_btc_interval_hours)
                except Exception:
                    logger.warning("Adaptive config selector init failed", exc_info=True)

            logger.info("LiveRunner started. Press Ctrl+C to stop.")
            while self._running:
                time.sleep(1.0)
                if self.timeout_tracker is not None:
                    _check_timeouts(self)
                if self._reload_models_pending:
                    self._reload_models_pending = False
                    self._handle_model_reload()
                if _adaptive_selector is not None:
                    now = time.time()
                    interval_sec = cfg.adaptive_btc_interval_hours * 3600
                    if now - _last_adaptive_check >= interval_sec:
                        _last_adaptive_check = now
                        self._run_adaptive_btc_check(_adaptive_selector)
                if (self.live_signal_tracker is not None
                        and self.ensemble_combiner is not None):
                    now = time.time()
                    if not hasattr(self, '_last_attr_feedback'):
                        self._last_attr_feedback = now
                    if now - self._last_attr_feedback >= 3600:
                        self._last_attr_feedback = now
                        self._apply_attribution_feedback()
                if _sd_notify_fn is not None:
                    _sd_notify_fn("WATCHDOG=1")

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop all subsystems gracefully."""
        with self._lifecycle_lock:
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
            for sub in (self.periodic_checkpointer, self.freshness_monitor,
                        self.data_scheduler, self.module_reloader,
                        self.health_server, self.alert_manager,
                        self.margin_monitor, self.reconcile_scheduler):
                if sub is not None:
                    try:
                        sub.stop()
                    except Exception:
                        logger.warning("Subsystem stop error", exc_info=True)
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
        run_adaptive_btc_check(self, selector)

    def _handle_model_reload(self) -> None:
        handle_model_reload(self)

    # halt, reduce_only, resume, flush, shutdown, apply_control
    # -> OperatorControlMixin (runner/operator_control.py)

    # operator_status_snapshot, operator_status, execution_alert_history, ...
    # -> OperatorObservabilityMixin (runner/observability.py)


# -- Module-level helpers (extracted to runner/live_runner_module_helpers.py) --
from runner.live_runner_module_helpers import (  # noqa: E402, F401
    _setup_systemd_notify,
    _start_optional,
    _install_sighup,
    _start_user_stream,
    _check_timeouts,
    _reconcile_startup,
    _FillRecordingAdapter,
)


if __name__ == "__main__":
    from runner.live_runner_cli import main
    main()
