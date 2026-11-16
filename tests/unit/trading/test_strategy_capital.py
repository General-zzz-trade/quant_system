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


# -- funding in the base (tech lead 2026-11-16) -------------------------------------------------------------------
class Venue:
    """Funding income per symbol; records (symbol, since_ms, now_ms) of every request."""

    def __init__(self, income):
        self.income, self.calls, self.down = dict(income), [], False

    def get_funding_income(self, symbol, *, since_ms, now_ms):
        self.calls.append((symbol, since_ms, now_ms))
        if self.down:
            raise OSError("HTTP 503")
        return self.income[symbol]


def _funding(venue, first_fills, clock):
    from trading.application.equity import StrategyFunding
    return StrategyFunding(venue, ["BTCUSDT", "ETHUSDT"], first_fills.get, clock)


def test_funding_counts_from_each_symbol_s_first_fill_and_a_symbol_never_traded_adds_nothing():
    venue = Venue({"BTCUSDT": D("-3.25"), "ETHUSDT": D("1.5")})
    assert _funding(venue, {"BTCUSDT": 500.5}, Clock()).current() == D("-3.25")
    assert venue.calls == [("BTCUSDT", 500_500, 1_000_000)]   # no request for a symbol never traded
    venue.calls.clear()
    assert _funding(venue, {"BTCUSDT": 500.5, "ETHUSDT": 400.0}, Clock()).current() == D("-1.75")
    assert venue.calls == [("BTCUSDT", 500_500, 1_000_000), ("ETHUSDT", 400_000, 1_000_000)]


def test_funding_is_asked_at_most_every_ten_minutes():
    venue, clock = Venue({"BTCUSDT": D("-1"), "ETHUSDT": D("0")}), Clock()
    funding = _funding(venue, {"BTCUSDT": 10.0}, clock)
    funding.current()
    clock.now += 599
    venue.income["BTCUSDT"] = D("-2")
    assert funding.current() == D("-1") and len(venue.calls) == 1    # the cached total, no request
    clock.now += 1
    assert funding.current() == D("-2") and len(venue.calls) == 2


def test_a_failed_refresh_keeps_the_last_total_for_two_hours_then_the_base_sizes_to_zero():
    venue, clock = Venue({"BTCUSDT": D("-4"), "ETHUSDT": D("0")}), Clock()
    funding = _funding(venue, {"BTCUSDT": 10.0}, clock)
    account = EquitySource(clock)
    account.update(D("10000"))
    capped = CappedEquity(account, D("300"), lambda: D("20") + funding.current())
    start = clock.now
    assert capped.current() == D("316")
    venue.down = True
    for at in (600, 1200, 7200):                                      # failures are tried again, never raised
        clock.now = start + at
        assert funding.current() == D("-4")
    assert len(venue.calls) == 4
    clock.now = start + 7201                                          # older than two hours: unknown
    account.update(D("10000"))
    with pytest.raises(RuntimeError, match="funding"):
        funding.current()
    assert capped.current() == 0
    venue.down, clock.now = False, start + 7800                       # the next refresh is due again
    account.update(D("10000"))
    assert capped.current() == D("316")


@pytest.mark.parametrize("broken", ["venue", "ledger"])
def test_funding_never_read_is_unknown_and_sizes_the_base_to_zero(broken):
    from trading.application.equity import StrategyFunding
    venue = Venue({"BTCUSDT": D("5")})
    venue.down = broken == "venue"

    def first_fill(symbol):
        if broken == "ledger":
            raise OSError("database is locked")
        return 10.0
    funding = StrategyFunding(venue, ["BTCUSDT"], first_fill, Clock())
    with pytest.raises(RuntimeError, match="funding"):
        funding.current()
    assert CappedEquity(_account("10000"), D("300"), lambda: D("0") + funding.current()).current() == 0


def test_the_first_fill_is_the_earliest_order_of_the_owner_with_a_fill_or_a_manual_ledger_position(tmp_path):
    from tests.support.state_db import state_db
    from trading.adapters.sqlite.order_journal import SqliteOrderJournal
    journal = SqliteOrderJournal(state_db(tmp_path / "state.sqlite"), "scope")
    btc, eth, sol = "target.BTCUSDT", "target.ETHUSDT", "target.SOLUSDT"

    def order(intent, symbol, owner, at, *, filled):
        journal.reserve(intent_id=intent, symbol=symbol, client_id=intent, side="buy", qty="1", reducing=False,
                        owner=owner, parent=None, order_type="market", clock=lambda: at)
        journal.update_status(intent, "filled" if filled else "rejected", settle=True)
        if filled:
            journal.insert_fill(fill_id=f"f-{intent}", intent_id=intent, owner=owner, payload={})
    assert journal.first_fill_at(btc) is None
    order("e1", "ETHUSDT", eth, 10.0, filled=True)                  # another owner
    order("b1", "BTCUSDT", btc, 50.0, filled=False)                 # never filled
    assert journal.first_fill_at(btc) is None
    order("b2", "BTCUSDT", btc, 100.0, filled=True)
    order("b3", "BTCUSDT", btc, 200.0, filled=True)
    assert journal.first_fill_at(btc) == 100.0 and journal.first_fill_at(eth) == 10.0
    journal.set_position_manually(sol, {"qty": "1"}, venue={}, reason="ledger := venue", at=70.0)
    assert journal.first_fill_at(sol) == 70.0                       # a position put into the ledger by a person
    journal.insert_fill(fill_id="f-x", intent_id="no-such-order", owner=btc, payload={})
    with pytest.raises(RuntimeError, match="creation time"):        # a fill of unknown time: unknown funding
        journal.first_fill_at(btc)
