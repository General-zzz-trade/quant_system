# runner/live_runner.py
"""LiveRunner — full production live trading runner.

Assembles:
  - EngineCoordinator + EngineLoop
  - ExecutionBridge (production, via venue_clients)
  - KillSwitch (execution gate)
  - ReconcileScheduler (periodic position/balance reconciliation)
  - MarginMonitor (margin ratio alerting)
  - GracefulShutdown (SIGTERM/SIGINT handling)
  - SystemHealthMonitor (stale data / drawdown alerts)

Usage:
    runner = LiveRunner.build(config, venue_clients={"binance": client}, ...)
    runner.start()  # blocks until stop() or signal
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.loop import EngineLoop, LoopConfig
from engine.guards import build_basic_guard, GuardConfig

from decision.regime_bridge import RegimeAwareDecisionModule
from decision.regime_policy import RegimePolicy

from monitoring.engine_hook import EngineMonitoringHook
from monitoring.health import SystemHealthMonitor, HealthConfig

from risk.kill_switch import KillMode, KillScope, KillSwitch

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


class _MarginMonitor:
    """Lightweight margin ratio monitor that runs on a background thread.

    Periodically calls fetch_margin() and fires alerts when the ratio
    breaches warning/critical thresholds.  If critical, triggers the kill switch.
    """

    def __init__(
        self,
        *,
        fetch_margin: Callable[[], float],
        kill_switch: KillSwitch,
        warning_ratio: float = 0.15,
        critical_ratio: float = 0.08,
        interval_sec: float = 30.0,
        on_alert: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self._fetch_margin = fetch_margin
        self._kill_switch = kill_switch
        self._warning_ratio = warning_ratio
        self._critical_ratio = critical_ratio
        self._interval_sec = interval_sec
        self._on_alert = on_alert
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="margin-monitor", daemon=True,
        )
        self._thread.start()
        logger.info("MarginMonitor started (interval=%ss)", self._interval_sec)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._interval_sec * 2)
            self._thread = None
        logger.info("MarginMonitor stopped")

    def check_once(self) -> Optional[float]:
        try:
            ratio = self._fetch_margin()
        except Exception:
            logger.exception("Failed to fetch margin ratio")
            return None

        if ratio <= self._critical_ratio:
            logger.error(
                "CRITICAL margin ratio %.4f <= %.4f — triggering kill switch",
                ratio, self._critical_ratio,
            )
            self._kill_switch.trigger(
                scope=KillScope.GLOBAL,
                key="*",
                mode=KillMode.HARD_KILL,
                reason=f"margin_critical:{ratio:.4f}",
                source="margin_monitor",
            )
            if self._on_alert:
                self._on_alert("critical", ratio)
        elif ratio <= self._warning_ratio:
            logger.warning(
                "Margin ratio %.4f <= %.4f (warning threshold)",
                ratio, self._warning_ratio,
            )
            if self._on_alert:
                self._on_alert("warning", ratio)

        return ratio

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._interval_sec)
            if self._running:
                self.check_once()


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
    margin_monitor: Optional[_MarginMonitor] = None
    shutdown_handler: Optional[GracefulShutdown] = None
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

        def _record_fill(fill: Any) -> None:
            fills.append({
                "ts": str(getattr(fill, "ts", "")),
                "symbol": str(getattr(fill, "symbol", "")),
                "side": str(getattr(fill, "side", "")),
                "qty": str(getattr(fill, "qty", "")),
                "price": str(getattr(fill, "price", "")),
            })
            if on_fill is not None:
                on_fill(fill)

        # 1) KillSwitch
        kill_switch = KillSwitch()

        # 2) Coordinator with monitoring hook
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

        # 3) Execution adapter from venue_clients
        venue_client = venue_clients.get(config.venue)
        if venue_client is None:
            raise ValueError(
                f"No venue client for '{config.venue}'. "
                f"Available: {list(venue_clients.keys())}"
            )

        # Wrap venue client as ExecutionAdapter (must have send_order)
        # Then wrap with kill switch gate
        exec_adapter = _KillSwitchAdapter(
            inner=venue_client,
            kill_switch=kill_switch,
            on_fill=_record_fill,
        )
        exec_bridge = ExecutionBridge(adapter=exec_adapter, dispatcher_emit=_emit)
        coordinator.attach_execution_bridge(exec_bridge)

        # 4) Decision bridge
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

        # 5) EngineLoop with guard
        guard = build_basic_guard(GuardConfig())
        loop = EngineLoop(coordinator=coordinator, guard=guard, cfg=LoopConfig())

        # 6) Market data runtime
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

        # 7) ReconcileScheduler (if enabled and fetch_venue_state provided)
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

        # 8) MarginMonitor (if fetch_margin provided)
        margin_monitor = None
        if fetch_margin is not None:
            margin_monitor = _MarginMonitor(
                fetch_margin=fetch_margin,
                kill_switch=kill_switch,
                warning_ratio=config.margin_warning_ratio,
                critical_ratio=config.margin_critical_ratio,
                interval_sec=config.margin_check_interval_sec,
            )

        # 9) GracefulShutdown
        shutdown_cfg = ShutdownConfig(
            pending_order_timeout_sec=config.pending_order_timeout_sec,
        )
        shutdown_handler = GracefulShutdown(
            config=shutdown_cfg,
            stop_new_orders=lambda: kill_switch.trigger(
                scope=KillScope.GLOBAL,
                key="*",
                mode=KillMode.HARD_KILL,
                reason="graceful_shutdown",
                source="shutdown",
            ),
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


class _KillSwitchAdapter:
    """Wraps a venue client with kill switch gating.

    Before forwarding send_order to the inner adapter, checks the kill switch.
    If killed, returns empty (order blocked).  Also invokes on_fill for each
    result event that looks like a fill.
    """

    def __init__(
        self,
        *,
        inner: Any,
        kill_switch: KillSwitch,
        on_fill: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._inner = inner
        self._kill_switch = kill_switch
        self._on_fill = on_fill

    def send_order(self, order_event: Any) -> list:
        symbol = getattr(order_event, "symbol", None) or ""
        strategy_id = getattr(order_event, "strategy_id", None)
        reduce_only = getattr(order_event, "reduce_only", False)

        allowed, record = self._kill_switch.allow_order(
            symbol=str(symbol),
            strategy_id=strategy_id,
            reduce_only=bool(reduce_only),
        )
        if not allowed:
            logger.warning(
                "Order blocked by kill switch: symbol=%s scope=%s mode=%s reason=%s",
                symbol,
                record.scope.value if record else "?",
                record.mode.value if record else "?",
                record.reason if record else "",
            )
            return []

        results = list(self._inner.send_order(order_event))

        if self._on_fill is not None:
            for ev in results:
                event_type = getattr(ev, "event_type", None)
                et_value = getattr(event_type, "value", str(event_type)) if event_type else ""
                if "fill" in str(et_value).lower():
                    self._on_fill(ev)

        return results
