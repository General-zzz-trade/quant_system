"""RunnerLoop — main event loop: bar dispatch + timeout checks.

Wraps existing EngineLoop/WsRuntime. Dispatches bars through
TradingEngine → RiskManager → BinanceExecutor pipeline.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RunnerLoop:
    """1-second poll loop: dispatch bars, check timeouts."""

    def __init__(
        self,
        engine: Any,
        risk: Any,
        orders: Any,
        executor: Any,
        on_signal: Callable | None = None,
    ) -> None:
        self._engine = engine
        self._risk = risk
        self._orders = orders
        self._executor = executor
        self._on_signal = on_signal
        self._running = False

    def on_bar(self, symbol: str, bar: dict) -> None:
        """Process one bar through the full pipeline.

        Engine → risk check → submit order if signal passes.
        """
        prediction = self._engine.on_bar(symbol, bar)
        if prediction is None:
            return

        if self._on_signal:
            self._on_signal(symbol, prediction, bar)

    def poll(self) -> None:
        """Called every 1 second: check order timeouts."""
        timed_out = self._orders.check_timeouts()
        for order_id in timed_out:
            logger.warning("Order timed out: %s", order_id)

    def start(self, ws_runtime: Any = None) -> None:
        """Enter event loop. If ws_runtime provided, use it. Otherwise poll loop."""
        self._running = True
        logger.info("RunnerLoop started")

        if ws_runtime is not None:
            # Use WS runtime event loop (production path)
            try:
                ws_runtime.run()
            except KeyboardInterrupt:
                pass
        else:
            # Simple poll loop (testing / non-WS path)
            while self._running:
                self.poll()
                time.sleep(1)

    def stop(self) -> None:
        """Stop the event loop."""
        self._running = False
        logger.info("RunnerLoop stopped")
