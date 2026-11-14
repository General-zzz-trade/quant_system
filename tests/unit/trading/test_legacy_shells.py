"""The compatibility shells supply what the application layer takes by injection (step-9 audit F7)."""
from __future__ import annotations

import inspect
import time
from decimal import Decimal as D
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.protective_stops import ProtectiveStopManager
from shared.confirmed_execution import ConfirmedExecutor


class _Venue:
    def send_market_order(self, symbol, side, qty, *, reduce_only=False, client_order_id=""):
        self.client_id = client_order_id
        return {"status": "submitted", "orderId": "7"}

    def get_order(self, symbol, *, order_id="", client_order_id=""):
        return SimpleNamespace(order_id="7", client_order_id=self.client_id, symbol="BTCUSDT", side="buy",
                               state="filled", filled_qty="1")

    def get_recent_fills(self, *, symbol=None, order_id="", since_ms=0, limit=100):
        return (SimpleNamespace(order_id="7", symbol="BTCUSDT", side="buy", trade_id="t1", qty="1", price="100",
                                fee="0", fee_asset="USDT"),)


def test_the_executor_shell_really_waits_between_placement_and_lookup(tmp_path):
    executor = ConfirmedExecutor(_Venue(), tmp_path / "orders.sqlite", "binance-testnet-ut", wait_seconds=0.5,
                                 owner="o", entry_order_type="market")
    with patch("shared.confirmed_execution.time.sleep") as sleep:
        assert len(executor.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("1"))) == 1
    sleep.assert_called_once_with(0.5)


def test_the_shells_default_to_the_wall_clock():
    for shell in (ConfirmedExecutor, ProtectiveStopManager):
        assert inspect.signature(shell.__init__).parameters["clock"].default is time.time


@pytest.mark.parametrize("testnet,key", [(True, "test-key"), (False, "main-key")])
def test_from_env_takes_the_keys_of_the_chosen_environment(monkeypatch, testnet, key):
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "test-secret")
    monkeypatch.setenv("BINANCE_API_KEY", "main-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "main-secret")
    config = BinanceConfig.from_env(testnet=testnet)
    assert (config.api_key, config.api_secret, config.testnet) == (key, key.replace("key", "secret"), testnet)


def test_from_env_refuses_to_fall_back_to_the_other_environment(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "main-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "main-secret")
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        BinanceConfig.from_env(testnet=True)
