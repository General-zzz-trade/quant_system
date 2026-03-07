# tests_unit/test_margin_monitor_thread_safety.py
from __future__ import annotations

import threading
import time

import pytest

from risk.margin_monitor import MarginConfig, MarginMonitor


def _make_monitor(interval: float = 0.05):
    return MarginMonitor(
        config=MarginConfig(check_interval_sec=interval),
        fetch_margin=lambda: {"margin_ratio": 0.5},
    )


class TestMarginMonitorThreadSafety:
    def test_double_start_single_thread(self):
        monitor = _make_monitor()
        monitor.start()
        try:
            thread1 = monitor._thread
            monitor.start()  # should be no-op
            assert monitor._thread is thread1
        finally:
            monitor.stop()

    def test_stop_without_start(self):
        monitor = _make_monitor()
        monitor.stop()  # should not raise

    def test_concurrent_start_stop(self):
        """Multiple threads calling start/stop concurrently should not crash."""
        monitor = _make_monitor(interval=0.02)
        errors = []

        def _start_stop():
            try:
                monitor.start()
                time.sleep(0.05)
                monitor.stop()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_start_stop) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert errors == [], f"Errors during concurrent start/stop: {errors}"

    def test_start_stop_cycle(self):
        monitor = _make_monitor(interval=0.02)
        for _ in range(3):
            monitor.start()
            time.sleep(0.05)
            monitor.stop()
            assert monitor._thread is None

    def test_lock_exists(self):
        monitor = _make_monitor()
        assert hasattr(monitor, "_lock")
        assert isinstance(monitor._lock, type(threading.Lock()))
