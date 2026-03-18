# runner/live_runner.py
"""LiveRunner — production live trading runner (唯一生产入口).

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
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from execution.latency.tracker import LatencyTracker
from execution.observability.incidents import timeout_to_alert
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


# ── Sub-builder helpers (used by builders) ─────────────────────────


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


@dataclass
class LiveRunner(OperatorControlMixin, OperatorObservabilityMixin):
    """Full live trading runner with reconciliation, kill switch, and margin monitoring.

    Use LiveRunner.build() to assemble the complete production stack.
    Call start() to begin trading (blocks until stop() or signal).
    """

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
    ensemble_combiner: Optional[Any] = None  # Direction 13: multi-TF ensemble
    regime_sizer: Optional[Any] = None  # Direction 17: regime-aware sizing
    staged_risk: Optional[Any] = None  # Staged risk manager (equity-based)
    live_signal_tracker: Optional[Any] = None  # Direction 18: attribution feedback
    portfolio_allocator: Optional[Any] = None  # Direction 19: cross-asset allocator
    periodic_checkpointer: Optional[PeriodicCheckpointer] = None
    event_recorder: Optional[EventRecorder] = None
    ack_store: Optional[Any] = None  # SQLiteAckStore when enable_persistent_stores=True
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
        latency_tracker, _record_fill, kill_switch, alert_sink = _build_core_infra(
            config, on_fill, alert_sink, fills,
        )

        # Phase 1.5: Rust hot-path components (optional)
        rust = _build_rust_components(config, config.symbols)

        # Phase 2: monitoring
        health, hook, alpha_health_monitor, regime_sizer, staged_risk, live_signal_tracker = (
            _build_monitoring(config, kill_switch, metrics_exporter)
        )

        # Phase 3: portfolio and correlation
        (
            portfolio_allocator, correlation_computer, _update_correlation,
            attribution_tracker, correlation_gate,
        ) = _build_portfolio_and_correlation(config)

        # Phase 4: order infrastructure
        order_state_machine, timeout_tracker, model_loader_inst, alpha_models = (
            _build_order_infra(config, alpha_models)
        )

        # Phase 5: features and inference
        feat_hook, inference_bridge, _dominance_computer = _build_features_and_inference(
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
        ) = _build_coordinator_and_pipeline(
            config, symbol_default, hook, feat_hook, tick_processors,
            _update_correlation, correlation_gate, kill_switch,
            order_state_machine, timeout_tracker, attribution_tracker,
            live_signal_tracker, alpha_health_monitor, regime_sizer,
            staged_risk, portfolio_allocator, fetch_margin, report,
        )

        # Phase 7: execution
        venue_client, ws_order_gateway = _build_execution(
            config, venue_clients, coordinator, kill_switch,
            _emit, _record_fill, risk_gate, report, _FillRecordingAdapter,
        )

        # Phase 8: decision bridge and engine loop
        decision_bridge_inst, module_reloader, loop = _build_decision(
            config, decision_modules, _emit, coordinator,
        )

        # Phase 9: market data runtime
        runtime, binance_urls = _build_market_data(
            config, transport, venue_client, loop,
        )

        # Phase 10: user stream
        user_stream_client = _build_user_stream(
            config, venue_client, coordinator, binance_urls,
            user_stream_transport, report,
        )

        # Phase 11: persistence and recovery
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

        # Phase 12: shutdown, health server, checkpointer
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
            staged_risk=staged_risk,
            live_signal_tracker=live_signal_tracker,
            portfolio_allocator=portfolio_allocator,
            periodic_checkpointer=periodic_checkpointer,
            event_recorder=event_recorder,
            ack_store=ack_store,
            _fills=fills,
        )
        if control_plane is not None:
            control_plane.runner = runner
        # Patch late-binding reference so cleanup callback can stop the runner
        _runner_ref.append(runner)
        runner._config = config
        runner._rust_components = rust
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
            if risk.get("max_position_notional") is not None:
                kwargs["max_position_notional"] = float(risk["max_position_notional"])
            if risk.get("max_order_notional") is not None:
                kwargs["max_order_notional"] = float(risk["max_order_notional"])
            if risk.get("max_leverage") is not None:
                leverage = float(risk["max_leverage"])
                kwargs["max_gross_leverage"] = leverage
                kwargs["max_net_leverage"] = leverage
            if risk.get("max_drawdown_pct") is not None:
                # Legacy nested configs only provided a single drawdown stop.
                # Map it conservatively into the ordered warning/reduce/kill ladder.
                dd_kill = float(risk["max_drawdown_pct"])
                kwargs["dd_warning_pct"] = dd_kill * 0.5
                kwargs["dd_reduce_pct"] = dd_kill * 0.75
                kwargs["dd_kill_pct"] = dd_kill
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
        _sd_notify_fn = None
        try:
            with self._lifecycle_lock:
                self._stopped = False
                self._running = True

                self._apply_perf_tuning()

                # ── Systemd watchdog notify (Direction 20) ──
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
                            except Exception as e:
                                logger.error(
                                    "Failed to send systemd notify '%s': %s",
                                    msg, e, exc_info=True,
                                )

                        _sd_notify_fn("READY=1")
                        logger.info("Systemd notify: READY=1")
                except Exception as e:
                    logger.error("Failed to initialize systemd watchdog: %s", e, exc_info=True)

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

                def _sighup_handler(signum: int, frame: Any) -> None:
                    logger.info("SIGHUP received — scheduling model reload")
                    self._reload_models_pending = True

                try:
                    if threading.current_thread() is threading.main_thread():
                        _signal.signal(_signal.SIGHUP, _sighup_handler)
                    else:
                        logger.warning("Skipping LiveRunner SIGHUP handler: not running in main thread")
                except (OSError, AttributeError, ValueError) as e:
                    logger.warning("Failed to install SIGHUP handler: %s", e)

                self.runtime.start()

                if self.user_stream is not None:
                    def _user_stream_loop() -> None:
                        try:
                            self.user_stream.connect()
                            self._record_user_stream_connect()
                        except Exception:
                            self._record_user_stream_failure(kind="connect")
                            logger.warning("User stream initial connect failed", exc_info=True)
                            return
                        _backoff = 1.0
                        _MAX_BACKOFF = 60.0
                        while self._running:
                            try:
                                self.user_stream.step()
                                _backoff = 1.0  # reset on success
                            except Exception:
                                self._record_user_stream_failure(kind="step")
                                logger.warning(
                                    "User stream step error, reconnecting in %.0fs",
                                    _backoff, exc_info=True,
                                )
                                time.sleep(_backoff)
                                try:
                                    self.user_stream.connect()
                                    self._record_user_stream_connect()
                                    _backoff = 1.0  # reset on successful reconnect
                                except Exception:
                                    self._record_user_stream_failure(kind="reconnect")
                                    logger.warning("User stream reconnect failed", exc_info=True)
                                    _backoff = min(_backoff * 2, _MAX_BACKOFF)

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
                    logger.info(
                        "Adaptive BTC config selector enabled (interval=%dh)",
                        cfg.adaptive_btc_interval_hours,
                    )
                except Exception:
                    logger.warning("Adaptive config selector init failed", exc_info=True)

            logger.info("LiveRunner started. Press Ctrl+C to stop.")
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
    except OSError as e:
        logger.debug("Could not pin to CPU1: %s", e)
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.mlockall(3)  # MCL_CURRENT | MCL_FUTURE
    except OSError as e:
        logger.debug("Could not lock memory pages: %s", e)

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
