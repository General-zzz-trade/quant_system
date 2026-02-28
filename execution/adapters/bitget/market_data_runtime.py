# execution/adapters/bitget/market_data_runtime.py
"""BitgetMarketDataRuntime -- RuntimeLike adapter for Bitget market data feed."""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

from execution.adapters.bitget.ws_client import BitgetWsMarketStreamClient

logger = logging.getLogger(__name__)


@dataclass
class BitgetMarketDataRuntime:
    """RuntimeLike adapter: Bitget WS market stream -> handler callbacks.

    Implements the RuntimeLike protocol (subscribe/unsubscribe/start/stop)
    expected by EngineLoop.attach_runtime() and EngineCoordinator.attach_runtime().

    The WS client pushes events via on_kline into an internal queue.
    A dispatcher thread drains the queue and calls registered handlers.
    """

    ws_client: BitgetWsMarketStreamClient
    symbols: Sequence[str] = ()
    kline_interval: str = "1m"

    _handlers: List[Callable[[Any], None]] = field(default_factory=list, init=False)
    _queue: deque = field(default_factory=deque, init=False)
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
            target=self._dispatch_loop, name="bitget-market-runtime", daemon=True,
        )
        self._thread.start()
        self.ws_client.start()
        logger.info("BitgetMarketDataRuntime started")

    def stop(self) -> None:
        self._running = False
        self.ws_client.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("BitgetMarketDataRuntime stopped")

    def enqueue(self, event: Any) -> None:
        """Called by the WS client's on_kline callback to push processed events."""
        self._queue.append(event)

    def _dispatch_loop(self) -> None:
        import time
        while self._running:
            try:
                event = self._queue.popleft()
            except IndexError:
                time.sleep(0.01)
                continue
            self._dispatch(event)

    def _dispatch(self, event: Any) -> None:
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("Handler error in BitgetMarketDataRuntime")
