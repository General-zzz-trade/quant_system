"""TickCollector — buffers incoming ticks and flushes to a TickStore."""
from __future__ import annotations

import logging
import threading
import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from data.backends.base import Tick, TickStore

logger = logging.getLogger(__name__)

FLUSH_SIZE = 1000
FLUSH_INTERVAL_S = 5.0


class TickCollector:
    """Subscribes to websocket aggTrade stream and writes ticks to a TickStore.

    Ticks are buffered and flushed every ``FLUSH_SIZE`` ticks or every
    ``FLUSH_INTERVAL_S`` seconds, whichever comes first.
    """

    def __init__(
        self,
        store: TickStore,
        symbols: Sequence[str],
        ws_callback: Callable[[Callable[[Dict[str, Any]], None]], None],
        *,
        flush_size: int = FLUSH_SIZE,
        flush_interval: float = FLUSH_INTERVAL_S,
    ) -> None:
        self._store = store
        self._symbols = set(s.upper() for s in symbols)
        self._ws_callback = ws_callback
        self._flush_size = flush_size
        self._flush_interval = flush_interval

        self._buffer: List[Tick] = []
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None
        self._last_active: Optional[float] = None

    # -- Collector protocol --------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ws_callback(self._on_trade)
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="tick-flush"
        )
        self._flush_thread.start()
        logger.info("TickCollector started for %s", sorted(self._symbols))

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None
        self._flush()
        logger.info("TickCollector stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_active_ts(self) -> Optional[float]:
        return self._last_active

    # -- Internal ------------------------------------------------------------

    def _on_trade(self, msg: Dict[str, Any]) -> None:
        symbol = str(msg.get("s", "")).upper()
        if self._symbols and symbol not in self._symbols:
            return

        from datetime import datetime, timezone

        ts_ms = msg.get("T") or msg.get("t")
        ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)

        tick = Tick(
            ts=ts,
            symbol=symbol,
            price=Decimal(str(msg.get("p", "0"))),
            qty=Decimal(str(msg.get("q", "0"))),
            side="sell" if msg.get("m") else "buy",
            trade_id=str(msg.get("a", "")),
        )

        with self._lock:
            self._buffer.append(tick)
            self._last_active = time.monotonic()
            if len(self._buffer) >= self._flush_size:
                self._flush_locked()

    def _flush_loop(self) -> None:
        while self._running:
            self._stop_event.wait(timeout=self._flush_interval)
            if not self._running:
                break
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()

        by_symbol: Dict[str, List[Tick]] = {}
        for tick in batch:
            by_symbol.setdefault(tick.symbol, []).append(tick)

        for symbol, ticks in by_symbol.items():
            try:
                self._store.write_ticks(symbol, ticks)
            except Exception:
                logger.exception("Failed to flush %d ticks for %s", len(ticks), symbol)

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)
