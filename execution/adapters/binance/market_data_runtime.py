# execution/adapters/binance/market_data_runtime.py
"""BinanceMarketDataRuntime — RuntimeLike adapter for live market data via WebSocket."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

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
    _handlers: List[Callable[[Any], None]] = field(default_factory=list, init=False)
    _running: bool = field(default=False, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)

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
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("BinanceMarketDataRuntime stopped")

    def _run_loop(self) -> None:
        while self._running:
            try:
                event = self.ws_client.step()
                if event is not None:
                    for handler in self._handlers:
                        try:
                            handler(event)
                        except Exception:
                            logger.exception("Handler error in market data runtime")
            except Exception:
                if not self._running:
                    break
                logger.exception("WS step error, continuing")
