# execution/safety/timeout_tracker.py
"""OrderTimeoutTracker — detects and cancels orders that exceed their timeout."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderTimeoutTracker:
    """Tracks pending orders and cancels those that exceed timeout.

    Thread-safe: on_submit/on_fill may be called from different threads
    than check_timeouts.

    Usage:
        tracker = OrderTimeoutTracker(timeout_sec=30.0, cancel_fn=bridge.cancel)
        tracker.on_submit(order_id, cmd)
        tracker.on_fill(order_id)    # removes from tracking
        tracker.check_timeouts()      # call periodically
    """

    timeout_sec: float = 30.0
    cancel_fn: Optional[Callable[[Any], Any]] = None
    clock_fn: Callable[[], float] = field(default_factory=lambda: time.monotonic)

    _pending: Dict[str, tuple[float, Any]] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def on_submit(self, order_id: str, cmd: Any = None) -> None:
        """Record an order submission."""
        with self._lock:
            self._pending[order_id] = (self.clock_fn(), cmd)

    def on_fill(self, order_id: str) -> None:
        """Remove a filled/cancelled order from tracking."""
        with self._lock:
            self._pending.pop(order_id, None)

    def on_cancel(self, order_id: str) -> None:
        """Remove a cancelled order from tracking."""
        with self._lock:
            self._pending.pop(order_id, None)

    def check_timeouts(self) -> list[str]:
        """Check for timed-out orders and cancel them. Returns list of timed-out order IDs."""
        with self._lock:
            if not self._pending:
                return []
            now = self.clock_fn()
            timed_out = []
            to_remove = []

            for order_id, (submit_ts, cmd) in self._pending.items():
                if now - submit_ts > self.timeout_sec:
                    timed_out.append(order_id)
                    to_remove.append((order_id, cmd))
                    logger.warning(
                        "Order %s timed out after %.1fs", order_id, now - submit_ts,
                    )

            for order_id, _ in to_remove:
                self._pending.pop(order_id, None)

        # Call cancel_fn outside the lock to avoid deadlocks
        for order_id, cmd in to_remove:
            if self.cancel_fn is not None and cmd is not None:
                try:
                    self.cancel_fn(cmd)
                except Exception:
                    logger.exception("Failed to cancel timed-out order %s", order_id)

        return timed_out

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def pending_order_ids(self) -> list[str]:
        with self._lock:
            return list(self._pending.keys())
