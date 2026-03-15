"""Tests for OrderManager."""
import time

from runner.order_manager import OrderManager


class TestOrderManagerLifecycle:
    def test_submit_increments_open_count(self):
        om = OrderManager(timeout_sec=30.0)
        om.submit("order-1", "BTCUSDT")
        assert om.open_count == 1

    def test_on_fill_decrements_open_count(self):
        om = OrderManager(timeout_sec=30.0)
        om.submit("order-1", "BTCUSDT")
        om.on_fill("order-1")
        assert om.open_count == 0

    def test_on_cancel_decrements_open_count(self):
        om = OrderManager(timeout_sec=30.0)
        om.submit("order-1", "BTCUSDT")
        om.on_cancel("order-1")
        assert om.open_count == 0

    def test_multiple_orders(self):
        om = OrderManager(timeout_sec=30.0)
        om.submit("o1", "BTCUSDT")
        om.submit("o2", "ETHUSDT")
        assert om.open_count == 2
        om.on_fill("o1")
        assert om.open_count == 1


class TestOrderManagerTimeout:
    def test_no_timeouts_when_fresh(self):
        om = OrderManager(timeout_sec=30.0)
        om.submit("order-1", "BTCUSDT")
        assert om.check_timeouts() == []

    def test_detects_stale_order(self):
        om = OrderManager(timeout_sec=0.01)
        om.submit("order-1", "BTCUSDT")
        time.sleep(0.02)
        timed_out = om.check_timeouts()
        assert "order-1" in timed_out

    def test_filled_order_not_timed_out(self):
        om = OrderManager(timeout_sec=0.01)
        om.submit("order-1", "BTCUSDT")
        om.on_fill("order-1")
        time.sleep(0.02)
        assert om.check_timeouts() == []


class TestOrderManagerDuplicates:
    def test_duplicate_submit_ignored(self):
        om = OrderManager(timeout_sec=30.0)
        om.submit("order-1", "BTCUSDT")
        om.submit("order-1", "BTCUSDT")
        assert om.open_count == 1

    def test_fill_unknown_order_ignored(self):
        om = OrderManager(timeout_sec=30.0)
        om.on_fill("nonexistent")  # should not raise
        assert om.open_count == 0
