"""Thread-safe pipeline metrics for production monitoring."""
from __future__ import annotations

import threading


class PipelineMetrics:
    """Tracks key pipeline counters in a thread-safe manner.

    Usage:
        metrics = PipelineMetrics()
        metrics.inc_bars_processed()
        metrics.inc_orders_submitted()
        print(metrics.to_dict())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.bars_processed: int = 0
        self.signals_generated: int = 0
        self.orders_submitted: int = 0
        self.orders_filled: int = 0
        self.orders_rejected: int = 0
        self.gate_blocks: int = 0
        self.errors: int = 0

    # -- increment helpers ------------------------------------------------

    def inc_bars_processed(self, n: int = 1) -> None:
        with self._lock:
            self.bars_processed += n

    def inc_signals_generated(self, n: int = 1) -> None:
        with self._lock:
            self.signals_generated += n

    def inc_orders_submitted(self, n: int = 1) -> None:
        with self._lock:
            self.orders_submitted += n

    def inc_orders_filled(self, n: int = 1) -> None:
        with self._lock:
            self.orders_filled += n

    def inc_orders_rejected(self, n: int = 1) -> None:
        with self._lock:
            self.orders_rejected += n

    def inc_gate_blocks(self, n: int = 1) -> None:
        with self._lock:
            self.gate_blocks += n

    def inc_errors(self, n: int = 1) -> None:
        with self._lock:
            self.errors += n

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of all counters."""
        with self._lock:
            return {
                "bars_processed": self.bars_processed,
                "signals_generated": self.signals_generated,
                "orders_submitted": self.orders_submitted,
                "orders_filled": self.orders_filled,
                "orders_rejected": self.orders_rejected,
                "gate_blocks": self.gate_blocks,
                "errors": self.errors,
            }

    def reset(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self.bars_processed = 0
            self.signals_generated = 0
            self.orders_submitted = 0
            self.orders_filled = 0
            self.orders_rejected = 0
            self.gate_blocks = 0
            self.errors = 0
