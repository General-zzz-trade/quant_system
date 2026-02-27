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
from typing import Any, Callable, Dict, List, Optional, Sequence

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.loop import EngineLoop, LoopConfig
from engine.guards import build_basic_guard, GuardConfig

from decision.regime_bridge import RegimeAwareDecisionModule
from decision.regime_policy import RegimePolicy

from execution.latency.tracker import LatencyTracker

from monitoring.alerts.manager import AlertManager
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
    state_store: Optional[Any] = None
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

        coord_cfg = CoordinatorConfig(
            symbol_default=symbol_default,
            symbols=config.symbols,
            currency=config.currency,
            on_pipeline_output=hook,
        )
        coordinator = EngineCoordinator(cfg=coord_cfg)

        def _emit(ev: Any) -> None:
            coordinator.emit(ev, actor="live")

        # ── 4) Execution adapter: KillSwitchBridge (production) ──
        venue_client = venue_clients.get(config.venue)
        if venue_client is None:
            raise ValueError(
                f"No venue client for '{config.venue}'. "
                f"Available: {list(venue_clients.keys())}"
            )

        kill_bridge = KillSwitchBridge(
            inner=venue_client,
            kill_switch=kill_switch,
            cancel_fn=getattr(venue_client, "cancel_all_orders", None),
        )

        # Wrap with fill recording: intercept results from send_order
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

        if modules:
            decision_bridge = DecisionBridge(
                dispatcher_emit=_emit, modules=modules,
            )
            coordinator.attach_decision_bridge(decision_bridge)

        # ── 6) EngineLoop with guard ─────────────────────────
        guard = build_basic_guard(GuardConfig())
        loop = EngineLoop(coordinator=coordinator, guard=guard, cfg=LoopConfig())

        # ── 7) Market data runtime ───────────────────────────
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
        runtime = BinanceMarketDataRuntime(ws_client=ws_client)
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

        # ── 10) AlertManager ────────────────────────────────
        alert_manager = AlertManager(sink=alert_sink)

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
            state_store=state_store,
            _fills=fills,
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
