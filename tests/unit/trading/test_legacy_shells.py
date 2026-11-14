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


def test_the_parts_moved_in_step_10_are_the_same_objects_at_their_old_addresses():
    from data.quality import causal_bars
    from runner import binance_main
    from trading.adapters.files.event_log import JsonlEventLog
    from trading.application.equity import EquitySource
    from trading.application.risk.readiness import EntryReadinessGate
    from trading.domain import bars
    assert causal_bars.Bar is bars.Bar and binance_main.bar_from_kline is bars.bar_from_kline
    assert binance_main.EventJournal is JsonlEventLog and binance_main.EntryReadinessGate is EntryReadinessGate
    assert issubclass(binance_main.EquitySource, EquitySource)
    assert inspect.signature(binance_main.EquitySource.__init__).parameters["clock"].default is time.time


def test_a_kline_row_becomes_a_closed_hourly_bar():
    from trading.domain.bars import Bar, bar_from_kline
    row = {"start": 7_200_000, "open": "1", "high": "3", "low": "0.5", "close": "2", "volume": "10"}
    bar = bar_from_kline(row)
    assert bar == Bar(7_200_000, D("1"), D("3"), D("0.5"), D("2"), D("10")) and bar.available_at == 10_800_000
    assert bar.extras is None and "extras" not in bar.record()
    full = bar_from_kline({**row, "quote_volume": "20", "trades": 3, "taker_buy_volume": "4",
                           "taker_buy_quote_volume": "8"})
    assert full.extras == {"quote_volume": D("20"), "trades": D("3"), "taker_buy_volume": D("4"),
                           "taker_buy_quote_volume": D("8")} and full == bar
    with pytest.raises(ValueError, match="hour-aligned"):
        bar_from_kline({**row, "start": 7_200_001})
