"""Tests for BinanceExecutor — mocked venue client."""
from unittest.mock import MagicMock

from runner.binance_executor import BinanceExecutor


class TestBinanceExecutorSend:
    def test_send_delegates_to_client(self):
        client = MagicMock()
        client.send_order.return_value = {"order_id": "123"}
        ks = MagicMock()
        ks.is_killed.return_value = False
        exe = BinanceExecutor(venue_client=client, kill_switch=ks)
        result = exe.send(MagicMock())
        client.send_order.assert_called_once()
        assert result["order_id"] == "123"

    def test_send_blocked_when_killed(self):
        client = MagicMock()
        ks = MagicMock()
        ks.is_killed.return_value = True
        exe = BinanceExecutor(venue_client=client, kill_switch=ks)
        result = exe.send(MagicMock())
        client.send_order.assert_not_called()
        assert result["status"] == "blocked_kill_switch"


class TestBinanceExecutorShadow:
    def test_shadow_mode_logs_but_does_not_send(self):
        client = MagicMock()
        ks = MagicMock()
        ks.is_killed.return_value = False
        exe = BinanceExecutor(venue_client=client, kill_switch=ks, shadow_mode=True)
        result = exe.send(MagicMock())
        client.send_order.assert_not_called()
        assert result["status"] == "shadow"


class TestBinanceExecutorFillCallback:
    def test_on_fill_invokes_callback(self):
        client = MagicMock()
        ks = MagicMock()
        fills = []
        exe = BinanceExecutor(venue_client=client, kill_switch=ks, on_fill=fills.append)
        exe._handle_fill({"order_id": "1", "qty": 0.1})
        assert len(fills) == 1
        assert fills[0]["order_id"] == "1"

    def test_no_callback_doesnt_crash(self):
        client = MagicMock()
        ks = MagicMock()
        exe = BinanceExecutor(venue_client=client, kill_switch=ks)
        exe._handle_fill({"order_id": "1"})  # should not raise
