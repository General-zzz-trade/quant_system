# execution/adapters/binance/market_data_runtime.py
"""BinanceMarketDataRuntime — RuntimeLike adapter for live market data via WebSocket."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from data.quality.live_validator import LiveBarValidator
from execution.adapters.binance.rest_kline_source import RestKlineSource
from execution.adapters.binance.ws_market_stream_um import BinanceUmMarketStreamWsClient

logger = logging.getLogger(__name__)


@dataclass
class BinanceMarketDataRuntime:
    """RuntimeLike adapter: WS market stream → handler callbacks.

    Implements the RuntimeLike protocol (subscribe/unsubscribe) expected by
    EngineLoop.attach_runtime() and EngineCoordinator.attach_runtime().

    Usage:
        runtime = BinanceMarketDataRuntime(ws_client=client)
        loop.attach_runtime(runtime)
        runtime.start()
    """

    ws_client: BinanceUmMarketStreamWsClient
    quality_gate: Optional[LiveBarValidator] = None
    rest_fallback: Optional[RestKlineSource] = None
    symbols: tuple[str, ...] = ()
    kline_interval: str = "1m"
    rest_cooldown_s: float = 5.0
    _handlers: List[Callable[[Any], None]] = field(default_factory=list, init=False)
    _running: bool = field(default=False, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _last_event_ts: float = field(default=0.0, init=False)

    def subscribe(self, handler: Callable[[Any], None]) -> None:
        if handler not in self._handlers:
            self._handlers.append(handler)

    def unsubscribe(self, handler: Callable[[Any], None]) -> None:
        try:
            self._handlers.remove(handler)
        except ValueError:
            pass

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="market-data-runtime", daemon=True,
        )
        self._thread.start()
        logger.info("BinanceMarketDataRuntime started")

    def stop(self) -> None:
        self._running = False
        try:
            self.ws_client.close()
        except Exception as e:
            logger.error("Failed to close market data WS client: %s", e, exc_info=True)
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
            self._thread = None
        logger.info("BinanceMarketDataRuntime stopped")

    def _run_loop(self) -> None:
        while self._running:
            try:
                event = self.ws_client.step()
                if event is not None:
                    self._last_event_ts = time.monotonic()
                    if self.quality_gate and not self.quality_gate.validate(event):
                        continue
                    self._dispatch(event)
            except Exception:
                if not self._running:
                    break
                self._try_rest_fallback()
                logger.exception("WS step error, continuing")

    def _dispatch(self, event: Any) -> None:
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("Handler error in market data runtime")

    def _try_rest_fallback(self) -> None:
        if self.rest_fallback is None or not self.symbols:
            return
        now = time.monotonic()
        if self._last_event_ts > 0 and (now - self._last_event_ts) < self.rest_cooldown_s:
            return
        logger.info("Attempting REST kline fallback for %d symbols", len(self.symbols))
        for sym in self.symbols:
            try:
                events = self.rest_fallback.fetch_as_events(
                    sym, interval=self.kline_interval, limit=2,
                )
                for ev in events:
                    self._dispatch(ev)
            except Exception:
                logger.exception("REST fallback failed for %s", sym)
