# runner/live_paper_runner.py
"""Live Paper Trading Runner — assembles the full production stack with simulated execution.

Connects:
  BinanceMarketDataRuntime → EngineLoop → StatePipeline
    → DecisionBridge(RegimeAwareDecisionModule) → IntentEvent/OrderEvent
      → ExecutionBridge(BacktestExecutionAdapter) → FillEvent → Pipeline
    + EngineMonitoringHook
    + ReconcileScheduler (optional)
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.loop import EngineLoop, LoopConfig
from engine.guards import build_basic_guard, GuardConfig
from engine.pipeline import PipelineOutput

from execution.adapters.binance.kline_processor import KlineProcessor
from execution.adapters.binance.ws_market_stream_um import (
    BinanceUmMarketStreamWsClient,
    MarketStreamConfig,
)
from execution.adapters.binance.market_data_runtime import BinanceMarketDataRuntime

from decision.regime_bridge import RegimeAwareDecisionModule
from decision.regime_policy import RegimePolicy

from monitoring.engine_hook import EngineMonitoringHook
from monitoring.health import SystemHealthMonitor, HealthConfig

from runner.backtest_runner import BacktestExecutionAdapter

logger = logging.getLogger(__name__)


@dataclass
class LivePaperConfig:
    """Configuration for live paper trading."""
    symbols: tuple[str, ...] = ("BTCUSDT",)
    starting_balance: float = 10000.0
    currency: str = "USDT"
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    log_interval_sec: float = 60.0
    ws_base_url: str = "wss://fstream.binance.com/stream"
    kline_interval: str = "1m"
    enable_regime_gate: bool = True
    enable_monitoring: bool = True
    health_stale_data_sec: float = 120.0


@dataclass
class LivePaperRunner:
    """Assembles and runs the full production stack with simulated execution.

    Usage:
        runner = LivePaperRunner.build(config, decision_modules=[my_module])
        runner.start()  # blocks until Ctrl+C or stop()
    """

    loop: EngineLoop
    coordinator: EngineCoordinator
    runtime: BinanceMarketDataRuntime
    health: Optional[SystemHealthMonitor]
    _on_fill: Optional[Callable[[Any], None]] = None
    _fills: List[Dict[str, Any]] = field(default_factory=list)
    _bar_count: int = field(default=0, init=False)
    _running: bool = field(default=False, init=False)

    @classmethod
    def build(
        cls,
        config: LivePaperConfig,
        *,
        decision_modules: Sequence[Any] | None = None,
        transport: Any = None,
        metrics_exporter: Any = None,
        on_fill: Callable[[Any], None] | None = None,
    ) -> "LivePaperRunner":
        """Build the full production stack.

        Args:
            config: Runner configuration.
            decision_modules: Decision modules to use. If not provided,
                defaults to empty list (rebalance module can be added).
            transport: WsTransport override for testing.
            metrics_exporter: Optional PrometheusExporter for metrics.
            on_fill: Optional callback for each fill event.
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

        # 1) Coordinator
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
            starting_balance=config.starting_balance,
            on_pipeline_output=hook,
        )
        coordinator = EngineCoordinator(cfg=coord_cfg)

        # 2) Emit helper (goes through coordinator)
        def _emit(ev: Any) -> None:
            coordinator.emit(ev, actor="paper")

        # 3) Execution bridge (paper adapter)
        def _price(sym: str) -> Optional[Decimal]:
            view = coordinator.get_state_view()
            markets = view.get("markets", {})
            m = markets.get(sym)
            if m is None:
                return None
            return getattr(m, "close", None) or getattr(m, "last_price", None)

        def _ts() -> Optional[datetime]:
            return datetime.now(timezone.utc)

        exec_adapter = BacktestExecutionAdapter(
            price_source=_price,
            ts_source=_ts,
            fee_bps=Decimal(str(config.fee_bps)),
            slippage_bps=Decimal(str(config.slippage_bps)),
            source="paper",
            on_fill=_record_fill,
        )
        exec_bridge = ExecutionBridge(adapter=exec_adapter, dispatcher_emit=_emit)
        coordinator.attach_execution_bridge(exec_bridge)

        # 4) Decision bridge
        modules = list(decision_modules or [])

        if config.enable_regime_gate and modules:
            # Wrap each module in regime-aware gate
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

        # Attach runtime to loop (events → inbox → single-thread processing)
        loop.attach_runtime(runtime)

        return cls(
            loop=loop,
            coordinator=coordinator,
            runtime=runtime,
            health=health,
            _on_fill=_record_fill,
            _fills=fills,
        )

    def start(self) -> None:
        """Start the live paper trading system. Blocks until stop() is called."""
        self._running = True
        logger.info("Starting live paper trading system...")

        # Start subsystems
        self.coordinator.start()
        if self.health is not None:
            self.health.start()
        self.runtime.start()
        self.loop.start_background()

        logger.info("System running. Press Ctrl+C to stop.")

        last_log = time.monotonic()
        try:
            while self._running:
                time.sleep(1.0)
                now = time.monotonic()
                if now - last_log >= 60.0:
                    self._log_status()
                    last_log = now
        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop all subsystems gracefully."""
        if not self._running:
            return
        self._running = False

        logger.info("Stopping live paper trading system...")
        self.runtime.stop()
        self.loop.stop_background()
        self.coordinator.stop()
        if self.health is not None:
            self.health.stop()

        self._log_status()
        logger.info("System stopped. Total fills: %d", len(self._fills))

    def _log_status(self) -> None:
        view = self.coordinator.get_state_view()
        event_index = view.get("event_index", 0)
        phase = view.get("phase", "unknown")
        logger.info(
            "STATUS  phase=%s  event_index=%d  fills=%d",
            phase, event_index, len(self._fills),
        )

    @property
    def fills(self) -> List[Dict[str, Any]]:
        return list(self._fills)

    @property
    def event_index(self) -> int:
        return self.coordinator.get_state_view().get("event_index", 0)
