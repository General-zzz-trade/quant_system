"""The strategy capital cap (owner, 2026-11-16): the strategy shares the main account and sizes from
min(account equity, cap + its own cumulative realized PnL after fees). The owner chose that the base grows with
the strategy's profits (and shrinks with its losses); it never exceeds the account's equity, and anything unknown
sizes to zero. Execution on mainnet refuses to start without a cap.
"""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from trading.application.equity import CappedEquity, EquitySource
from trading.infrastructure.config import load_config, require_strategy_capital


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def _account(value):
    source = EquitySource(Clock())
    source.update(D(value))
    return source


@pytest.mark.parametrize("account,pnl,expected", [
    ("10000", "0", "300"),        # the cap, not the whole account
    ("10000", "45", "345"),       # profits grow the base (owner's choice)
    ("10000", "-120", "180"),     # losses shrink it
    ("250", "45", "250"),         # never more than the account holds
    ("10000", "-400", "0"),       # a base below zero sizes to zero
])
def test_the_base_is_the_cap_plus_the_strategy_s_own_pnl_within_the_account(account, pnl, expected):
    assert CappedEquity(_account(account), D("300"), lambda: D(pnl)).current() == D(expected)


def test_unknown_account_equity_or_an_unreadable_ledger_sizes_to_zero():
    stale = EquitySource(Clock())                                     # never updated: unknown
    assert CappedEquity(stale, D("300"), lambda: D("0")).current() == 0
    assert CappedEquity(_account("10000"), D("300"), lambda: None).current() == 0

    def broken():
        raise OSError("journal unreadable")
    assert CappedEquity(_account("10000"), D("300"), broken).current() == 0


ENV = {"RISK_SCOPE": "binance-ut", "BINANCE_TESTNET": "0", "TELEGRAM_BOT_TOKEN": "1:t", "TELEGRAM_CHAT_ID": "1",
       "HEARTBEAT_URL": "https://hc.example.invalid/x"}


def _load(**over):
    env = {k: v for k, v in dict(ENV, **over).items() if v is not None}
    return load_config(env, execution=True, stop_pct=D("0.08"))[0]


def test_the_cap_is_read_once_and_recorded():
    config = _load(STRATEGY_CAPITAL_USDT="300")
    assert config.strategy_capital == D("300")
    assert json.loads(json.dumps(config.record()))["strategy_capital"] == "300"


@pytest.mark.parametrize("raw", ["0", "-5", "nan", "inf", "three hundred", " "])
def test_a_cap_that_is_not_a_positive_amount_is_refused(raw):
    if raw.strip():
        with pytest.raises(RuntimeError, match="STRATEGY_CAPITAL_USDT"):
            _load(STRATEGY_CAPITAL_USDT=raw)
    else:
        assert _load(STRATEGY_CAPITAL_USDT=raw).strategy_capital is None


def test_mainnet_execution_needs_a_cap_and_the_testnet_does_not():
    with pytest.raises(RuntimeError, match="STRATEGY_CAPITAL_USDT"):
        require_strategy_capital(_load())
    require_strategy_capital(_load(STRATEGY_CAPITAL_USDT="300"))
    testnet = _load(BINANCE_TESTNET="1", RISK_SCOPE="binance-testnet-ut")
    require_strategy_capital(testnet)
    assert testnet.strategy_capital is None


def test_paper_reads_no_cap():
    config = load_config({"STRATEGY_CAPITAL_USDT": "300"}, execution=False, stop_pct=D("0.08"))[0]
    assert config.strategy_capital is None
    require_strategy_capital(config)


def test_the_strategy_s_pnl_is_its_owners_realized_pnl_after_usdt_fees():
    from trading.infrastructure.target import strategy_realized_pnl

    class Ledger:
        rows = {"target.BTCUSDT": {"position": {"realized_pnl": "50", "fees": {"USDT": "2", "BNB": "1"}}},
                "target.ETHUSDT": {"position": {"realized_pnl": "-10.5", "fees": {}}}}

        def checkpoint(self, owner):
            return self.rows.get(owner)
    assert strategy_realized_pnl(Ledger(), ["target.BTCUSDT", "target.ETHUSDT", "target.SOLUSDT"]) == D("37.5")

    class Unreadable:
        def checkpoint(self, owner):
            raise OSError("database is locked")
    with pytest.raises(OSError):
        strategy_realized_pnl(Unreadable(), ["target.BTCUSDT"])           # the cap then sizes to zero
