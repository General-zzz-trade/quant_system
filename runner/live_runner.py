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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.loop import EngineLoop, LoopConfig
from engine.guards import build_basic_guard, GuardConfig

from decision.regime_bridge import RegimeAwareDecisionModule
from decision.regime_policy import RegimePolicy

from engine.feature_hook import FeatureComputeHook
from execution.latency.tracker import LatencyTracker

from monitoring.alerts.base import Severity
from monitoring.alerts.manager import AlertManager, AlertRule
from monitoring.engine_hook import EngineMonitoringHook
from monitoring.health import SystemHealthMonitor, HealthConfig

from risk.kill_switch import KillMode, KillScope, KillSwitch
from risk.kill_switch_bridge import KillSwitchBridge
from risk.margin_monitor import MarginConfig, MarginMonitor

from runner.graceful_shutdown import GracefulShutdown, ShutdownConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LiveRunnerConfig:
    symbols: tuple[str, ...] = ("BTCUSDT",)
    currency: str = "USDT"
    ws_base_url: str = "wss://fstream.binance.com/stream"
    kline_interval: str = "1m"
    enable_regime_gate: bool = True
    enable_monitoring: bool = True
    enable_reconcile: bool = True
    reconcile_interval_sec: float = 60.0
    health_stale_data_sec: float = 120.0
    venue: str = "binance"
    # Margin monitoring
    margin_check_interval_sec: float = 30.0
    margin_warning_ratio: float = 0.15
    margin_critical_ratio: float = 0.08
    # Shutdown
    pending_order_timeout_sec: float = 30.0
    # Production infrastructure
    data_dir: str = "data/live"
    enable_persistent_stores: bool = False
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    # Pre-flight checks
    enable_preflight: bool = True
    preflight_min_balance: float = 0.0
    # Startup reconciliation
    reconcile_on_startup: bool = True
    # Shadow mode — simulate orders without executing
    shadow_mode: bool = False
    # Latency SLA
    latency_p99_threshold_ms: float = 5000.0
    # Correlation risk
    max_avg_correlation: float = 0.7
    # Health HTTP endpoint
    health_port: Optional[int] = None


@dataclass
class LiveRunner:
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
    correlation_computer: Optional[Any] = None
    attribution_tracker: Optional[Any] = None
    correlation_gate: Optional[Any] = None
    risk_gate: Optional[Any] = None
    module_reloader: Optional[Any] = None
    decision_bridge: Optional[Any] = None
    _fills: List[Dict[str, Any]] = field(default_factory=list)
    _running: bool = field(default=False, init=False)

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

        # ── 3) Coordinator with monitoring hook ───────────────
        health: Optional[SystemHealthMonitor] = None
        hook: Optional[EngineMonitoringHook] = None

        if config.enable_monitoring:
            health = SystemHealthMonitor(
                config=HealthConfig(stale_data_sec=config.health_stale_data_sec),
            )
            hook = EngineMonitoringHook(health=health, metrics=metrics_exporter)

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

        # ── RiskGate (pre-execution size/notional checks) ────
        from execution.safety.risk_gate import RiskGate, RiskGateConfig
        risk_gate = RiskGate(
            config=RiskGateConfig(),
            get_positions=lambda: coordinator.get_state_view().get("positions", {}),
            is_killed=lambda: kill_switch.is_killed() is not None,
        )

        # ── Feature computation + ML inference hook ──────────
        feat_hook = None
        if feature_computer is not None:
            inference_bridge = None
            if alpha_models:
                from alpha.inference.bridge import LiveInferenceBridge
                inference_bridge = LiveInferenceBridge(models=list(alpha_models))
            feat_hook = FeatureComputeHook(
                computer=feature_computer, inference_bridge=inference_bridge,
            )

        coord_cfg = CoordinatorConfig(
            symbol_default=symbol_default,
            symbols=config.symbols,
            currency=config.currency,
            on_pipeline_output=hook,
            on_snapshot=_update_correlation,
            feature_hook=feat_hook,
        )
        coordinator = EngineCoordinator(cfg=coord_cfg)

        def _emit(ev: Any) -> None:
            # Attribution: track all events
            attribution_tracker.on_event(ev)

            # Correlation gate: check ORDER events
            et = getattr(ev, "event_type", None)
            et_str = (str(et.value) if hasattr(et, "value") else str(et)).upper() if et else ""
            if et_str == "ORDER":
                # Gate 1: Correlation check
                view = coordinator.get_state_view()
                positions = view.get("positions", {})
                existing = [s for s, p in positions.items() if float(getattr(p, "qty", 0)) != 0]
                sym = getattr(ev, "symbol", "")
                decision = correlation_gate.should_allow(sym, existing)
                if not decision.ok:
                    msg = decision.violations[0].message if decision.violations else "blocked"
                    logger.warning("CorrelationGate REJECTED order for %s: %s", sym, msg)
                    return

                # Gate 2: Risk size/notional check
                risk_result = risk_gate.check(ev)
                if not risk_result.allowed:
                    logger.warning("RiskGate REJECTED order for %s: %s", sym, risk_result.reason)
                    return

            coordinator.emit(ev, actor="live")

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
                close = getattr(m, "close", None) if m else None
                return _Dec(str(close)) if close is not None else None

            exec_adapter = ShadowExecutionAdapter(price_source=_shadow_price)
            logger.warning("SHADOW MODE — orders will be simulated, not executed")
        else:
            exec_adapter = _FillRecordingAdapter(inner=kill_bridge, on_fill=_record_fill)
        exec_bridge = ExecutionBridge(adapter=exec_adapter, dispatcher_emit=_emit)
        coordinator.attach_execution_bridge(exec_bridge)

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

        # ── 7) Market data runtime ───────────────────────────
        if config.venue == "bitget":
            from execution.adapters.bitget.kline_processor import (
                BitgetKlineProcessor,
            )
            from execution.adapters.bitget.ws_client import (
                BitgetWsMarketStreamClient,
                BitgetWsConfig,
            )
            from execution.adapters.bitget.market_data_runtime import (
                BitgetMarketDataRuntime,
            )

            bitget_processor = BitgetKlineProcessor(source="bitget.ws.kline")
            bitget_runtime = BitgetMarketDataRuntime(
                ws_client=BitgetWsMarketStreamClient(symbols=[], channel=""),
                symbols=config.symbols,
                kline_interval=config.kline_interval,
            )

            def _bitget_on_kline(raw: Any, _p=bitget_processor, _r=bitget_runtime) -> None:
                event = _p.process(raw)
                if event is not None:
                    _r.enqueue(event)

            bitget_ws = BitgetWsMarketStreamClient(
                symbols=config.symbols,
                channel=f"candle{config.kline_interval}",
                cfg=BitgetWsConfig(),
                on_kline=_bitget_on_kline,
            )
            bitget_runtime.ws_client = bitget_ws
            runtime = bitget_runtime
            loop.attach_runtime(runtime)
        else:
            from execution.adapters.binance.kline_processor import KlineProcessor
            from execution.adapters.binance.ws_market_stream_um import (
                BinanceUmMarketStreamWsClient,
                MarketStreamConfig,
            )
            from execution.adapters.binance.market_data_runtime import BinanceMarketDataRuntime

            if transport is None:
                try:
                    from execution.adapters.binance.ws_transport_websocket_client import (
                        WebsocketClientTransport,
                    )
                    transport = WebsocketClientTransport()
                except ImportError:
                    raise RuntimeError(
                        "websocket-client not installed. Run: pip install websocket-client"
                    )

            streams = tuple(
                f"{sym.lower()}@kline_{config.kline_interval}"
                for sym in config.symbols
            )
            processor = KlineProcessor(source="binance.ws.kline")
            ws_client = BinanceUmMarketStreamWsClient(
                transport=transport,
                processor=processor,
                streams=streams,
                cfg=MarketStreamConfig(ws_base_url=config.ws_base_url),
            )
            from execution.adapters.binance.rest_kline_source import RestKlineSource
            rest_fallback = RestKlineSource(
                base_url=getattr(venue_client, '_cfg', None) and venue_client._cfg.base_url or "https://fapi.binance.com",
                source="binance.rest.kline",
            )
            runtime = BinanceMarketDataRuntime(
                ws_client=ws_client,
                rest_fallback=rest_fallback,
                symbols=config.symbols,
                kline_interval=config.kline_interval,
            )
            loop.attach_runtime(runtime)

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

        # Default rule: stale market data
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

        # Default rule: high drawdown (>15%)
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

        # Default rule: kill switch triggered
        def _kill_switch_condition(ks=kill_switch) -> bool:
            return ks.is_killed() is not None

        alert_manager.add_rule(AlertRule(
            name="kill_switch_triggered",
            condition=_kill_switch_condition,
            severity=Severity.CRITICAL,
            message_template="Kill switch has been triggered — trading halted",
            cooldown_sec=60.0,
        ))

        # Default rule: latency SLA breach
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

        # Default rule: high portfolio correlation (uses correlation_computer created earlier)
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

        # ── 11) Persistent stores (conditional) ─────────────
        state_store = None
        if config.enable_persistent_stores:
            from execution.store.ack_store import SQLiteAckStore
            from execution.store.event_log import SQLiteEventLog
            from state.store import SqliteStateStore

            data_dir = config.data_dir
            SQLiteAckStore(path=os.path.join(data_dir, "ack_store.db"))
            SQLiteEventLog(path=os.path.join(data_dir, "event_log.db"))
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

        # ── 11a) Startup reconciliation ──────────────────────
        if config.reconcile_on_startup and fetch_venue_state is not None:
            try:
                venue_state = fetch_venue_state()
                local_view = coordinator.get_state_view()
                mismatches = _reconcile_startup(local_view, venue_state, config.symbols)
                for m in mismatches:
                    logger.warning("Startup reconciliation mismatch: %s", m)
                if mismatches:
                    logger.warning(
                        "Found %d mismatches — local state may be stale. "
                        "Consider manual review.", len(mismatches),
                    )
            except Exception:
                logger.exception(
                    "Startup reconciliation failed — proceeding with local state"
                )

        # ── 12) GracefulShutdown ─────────────────────────────
        shutdown_cfg = ShutdownConfig(
            pending_order_timeout_sec=config.pending_order_timeout_sec,
        )
        save_snapshot_fn = None
        if state_store is not None:
            def save_snapshot_fn(_path: str) -> None:
                snapshot = coordinator.get_state_view().get("last_snapshot")
                if snapshot is not None:
                    state_store.save(snapshot)
                    logger.info("State snapshot saved on shutdown")

        shutdown_handler = GracefulShutdown(
            config=shutdown_cfg,
            stop_new_orders=lambda: kill_switch.trigger(
                scope=KillScope.GLOBAL,
                key="*",
                mode=KillMode.HARD_KILL,
                reason="graceful_shutdown",
                source="shutdown",
            ),
            save_snapshot=save_snapshot_fn,
        )

        # ── 13) Health HTTP endpoint (optional) ────────────────
        health_server = None
        if config.health_port is not None and health is not None:
            from monitoring.health_server import HealthServer
            from dataclasses import asdict

            _stale_thresh = config.health_stale_data_sec

            def _health_status_fn() -> Dict[str, Any]:
                st = health.get_status()
                d = asdict(st)
                age = st.data_age_sec
                if age is not None and age > _stale_thresh:
                    d["status"] = "critical"
                return d

            health_server = HealthServer(
                port=config.health_port,
                status_fn=_health_status_fn,
            )

        return cls(
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
            correlation_computer=correlation_computer,
            attribution_tracker=attribution_tracker,
            correlation_gate=correlation_gate,
            risk_gate=risk_gate,
            module_reloader=module_reloader,
            decision_bridge=decision_bridge_inst,
            _fills=fills,
        )

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
    ) -> "LiveRunner":
        """Build a LiveRunner from a YAML/JSON config file.

        Loads the config securely (rejects plaintext secrets), validates
        against the trading config schema, maps config sections to
        LiveRunnerConfig fields, and delegates to build().
        """
        from infra.config.loader import load_config_secure, resolve_credentials
        from infra.config.schema import validate_trading_config

        raw = load_config_secure(config_path)

        errors = validate_trading_config(raw)
        if errors:
            raise ValueError(
                f"Config validation failed ({config_path}):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        trading = raw.get("trading", {})
        risk = raw.get("risk", {})
        execution = raw.get("execution", {})
        monitoring = raw.get("monitoring", {})
        log_cfg = raw.get("logging", {})

        symbol = trading.get("symbol", "BTCUSDT")
        symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

        kwargs: Dict[str, Any] = {}
        if risk.get("max_leverage") is not None:
            kwargs["margin_warning_ratio"] = float(risk["max_leverage"])
        if risk.get("max_drawdown_pct") is not None:
            kwargs["margin_critical_ratio"] = float(risk["max_drawdown_pct"]) / 100.0
        if risk.get("max_position_notional") is not None:
            pass  # no direct LiveRunnerConfig field; handled by decision modules
        if execution.get("fee_bps") is not None:
            pass  # no direct LiveRunnerConfig field; handled by execution layer
        if execution.get("slippage_bps") is not None:
            pass  # no direct LiveRunnerConfig field; handled by execution layer
        if monitoring.get("health_check_interval") is not None:
            kwargs["health_stale_data_sec"] = float(monitoring["health_check_interval"])

        runner_config = LiveRunnerConfig(
            symbols=symbols,
            venue=trading.get("exchange", "binance"),
            enable_structured_logging=log_cfg.get("structured", True),
            log_level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("file"),
            shadow_mode=shadow_mode,
            **kwargs,
        )

        resolve_credentials(raw)

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
        )

    def start(self) -> None:
        """Start the live trading system. Blocks until stop() or signal."""
        self._running = True

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
        self.runtime.start()
        self.loop.start_background()

        logger.info("LiveRunner started. Press Ctrl+C to stop.")
        try:
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop all subsystems gracefully."""
        if not self._running:
            return
        self._running = False

        logger.info("Stopping LiveRunner...")
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

        logger.info("LiveRunner stopped. Total fills: %d", len(self._fills))

    @property
    def fills(self) -> List[Dict[str, Any]]:
        return list(self._fills)

    @property
    def event_index(self) -> int:
        return self.coordinator.get_state_view().get("event_index", 0)


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

    local_balance = float(local_view.get("balance", 0))
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
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Live trading runner")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    parser.add_argument("--shadow", action="store_true", help="Shadow mode — simulate orders")
    args = parser.parse_args()

    # Venue clients must be constructed from config credentials
    from infra.config.loader import load_config_secure, resolve_credentials

    raw = load_config_secure(args.config)
    resolve_credentials(raw)

    venue_clients: Dict[str, Any] = {}
    exchange = raw.get("trading", {}).get("exchange", "binance")

    if exchange == "binance":
        from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig

        creds = raw.get("credentials", {})
        client = BinanceRestClient(
            cfg=BinanceRestConfig(
                api_key=creds.get("api_key", ""),
                api_secret=creds.get("api_secret", ""),
            )
        )
        venue_clients["binance"] = client
    elif exchange == "bitget":
        from execution.adapters.bitget.rest import BitgetRestClient, BitgetRestConfig
        from execution.adapters.bitget.rest_client import BitgetFuturesRestClient
        from execution.adapters.bitget.order_gateway import BitgetFuturesOrderGateway
        from execution.adapters.bitget.venue_client_futures import BitgetFuturesFullVenueClient

        creds = raw.get("credentials", {})
        rest = BitgetRestClient(
            cfg=BitgetRestConfig(
                api_key=creds.get("api_key", ""),
                api_secret=creds.get("api_secret", ""),
                passphrase=creds.get("passphrase", ""),
            )
        )
        rest_client = BitgetFuturesRestClient(client=rest)
        gw = BitgetFuturesOrderGateway(rest=rest)
        venue_clients["bitget"] = BitgetFuturesFullVenueClient(
            rest_client=rest_client, order_gateway=gw,
        )

    runner = LiveRunner.from_config(
        args.config,
        venue_clients=venue_clients,
        shadow_mode=getattr(args, 'shadow', False),
    )
    runner.start()
