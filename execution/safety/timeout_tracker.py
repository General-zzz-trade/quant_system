# execution/safety/timeout_tracker.py
"""OrderTimeoutTracker — detects and cancels orders that exceed their timeout."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_CANCEL_CMD_FIELDS = (
    "venue",
    "symbol",
    "order_id",
    "client_order_id",
    "orig_client_order_id",
    "binance_order_id",
    "command_id",
    "idempotency_key",
    "request_id",
)


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

    def checkpoint(self) -> dict:
        """Serialize pending orders for persistence.

        Stores elapsed seconds since submission (monotonic-based) rather than
        wall-clock timestamps. This avoids NTP clock skew causing incorrect
        timeout calculations after restore.
        """
        mono_now = self.clock_fn()
        with self._lock:
            entries = {}
            for order_id, (submit_ts, cmd) in self._pending.items():
                elapsed = mono_now - submit_ts
                entry = {"elapsed_sec": elapsed}
                serialized_cmd = _serialize_cancel_cmd(cmd)
                if serialized_cmd is not None:
                    entry["cancel_cmd"] = serialized_cmd
                entries[order_id] = entry
            return {"pending": entries, "timeout_sec": self.timeout_sec}

    def restore(self, data: dict) -> None:
        """Restore pending orders from checkpoint.

        Reconstructs monotonic submit times from stored elapsed durations.
        Backward compatible: accepts both 'elapsed_sec' (new) and
        'wall_submit_ts' (legacy) checkpoint formats.
        """
        import time as _time
        mono_now = self.clock_fn()
        with self._lock:
            self._pending.clear()
            for order_id, entry in data.get("pending", {}).items():
                if "elapsed_sec" in entry:
                    # New format: elapsed seconds since submission
                    elapsed = entry["elapsed_sec"]
                else:
                    # Legacy format: wall-clock submit timestamp
                    wall_now = _time.time()
                    wall_submit = entry.get("wall_submit_ts", wall_now)
                    elapsed = wall_now - wall_submit
                # Clamp negative elapsed (corrupted checkpoint or time travel)
                elapsed = max(0.0, elapsed)
                mono_submit = mono_now - elapsed
                cancel_cmd = _deserialize_cancel_cmd(entry.get("cancel_cmd"))
                self._pending[order_id] = (mono_submit, cancel_cmd)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def pending_order_ids(self) -> list[str]:
        with self._lock:
            return list(self._pending.keys())


def _serialize_cancel_cmd(cmd: Any) -> Any:
    if cmd is None:
        return None
    if isinstance(cmd, (str, int, float, bool)):
        return {"__raw__": cmd}
    if isinstance(cmd, dict):
        data = {
            key: value
            for key, value in cmd.items()
            if key in _CANCEL_CMD_FIELDS and value is not None
        }
        return data or None

    data: Dict[str, Any] = {}
    for field_name in _CANCEL_CMD_FIELDS:
        value = getattr(cmd, field_name, None)
        if value is not None:
            data[field_name] = value
    return data or None


def _deserialize_cancel_cmd(data: Any) -> Any:
    if not data:
        return None
    if isinstance(data, dict) and "__raw__" in data:
        return data["__raw__"]
    if isinstance(data, dict):
        return SimpleNamespace(**data)
    return None
