"""Orderbook collector — buffers L2 orderbook snapshots from exchange WS feeds."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Optional

from features.microstructure.orderbook import OrderbookSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OrderbookCollectorConfig:
    """Configuration for orderbook collection."""

    symbols: tuple[str, ...]
    depth: int = 20
    update_interval_ms: int = 100
    buffer_size: int = 1000


class OrderbookCollector:
    """Collect L2 orderbook snapshots from exchange WS.

    Parses incoming orderbook messages into OrderbookSnapshot objects,
    buffers them, and optionally invokes a callback per snapshot.
    """

    def __init__(
        self,
        *,
        config: OrderbookCollectorConfig,
        on_snapshot: Optional[Callable[[OrderbookSnapshot], None]] = None,
    ) -> None:
        self._config = config
        self._on_snapshot = on_snapshot
        self._buffer: list[OrderbookSnapshot] = []
        self._running = False
        self._last_active_ts: Optional[float] = None
        self._lock = threading.Lock()

    def on_message(self, msg: dict) -> None:
        """Handle incoming orderbook update message.

        Expected format:
            {
                "symbol": "BTCUSDT",
                "bids": [["40000.00", "1.5"], ...],
                "asks": [["40001.00", "2.0"], ...],
                "ts": 1700000000000,  # optional, epoch ms
            }
        """
        try:
            symbol = msg.get("symbol", "")
            if symbol and symbol not in self._config.symbols:
                return

            raw_bids = msg.get("bids", [])
            raw_asks = msg.get("asks", [])

            bids = tuple(
                (Decimal(str(b[0])), Decimal(str(b[1])))
                for b in raw_bids[:self._config.depth]
            )
            asks = tuple(
                (Decimal(str(a[0])), Decimal(str(a[1])))
                for a in raw_asks[:self._config.depth]
            )

            ts_ms = msg.get("ts")
            if ts_ms is not None:
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            snapshot = OrderbookSnapshot(
                ts=ts,
                symbol=symbol,
                bids=bids,
                asks=asks,
            )

            with self._lock:
                self._last_active_ts = time.monotonic()
                if len(self._buffer) >= self._config.buffer_size:
                    self._buffer.pop(0)
                self._buffer.append(snapshot)

            if self._on_snapshot is not None:
                self._on_snapshot(snapshot)

        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("Failed to parse orderbook message: %s", exc)

    def start(self) -> None:
        """Mark the collector as running."""
        self._running = True
        self._last_active_ts = time.monotonic()
        logger.info(
            "OrderbookCollector started for %s", self._config.symbols,
        )

    def stop(self) -> None:
        """Mark the collector as stopped."""
        self._running = False
        logger.info("OrderbookCollector stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_active_ts(self) -> Optional[float]:
        return self._last_active_ts

    def drain_buffer(self) -> list[OrderbookSnapshot]:
        """Get and clear buffered snapshots."""
        with self._lock:
            snapshots = list(self._buffer)
            self._buffer.clear()
        return snapshots

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)
