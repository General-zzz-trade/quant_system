"""The paper-only parts of the composition (trading/infrastructure/paper.py, S10 part 3), one by one and without a
runtime: the divergence events, the pauses that report them, per-symbol simulated equity, the simulated exchange as
research configures it, and the paper directory's state database."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from lab.strategy import StrategySpec
from trading.adapters.simulated.bar_broker import SYMBOL_RULES, broker_config
from trading.adapters.sqlite import schema
from trading.domain.events import EventKind
from trading.infrastructure.paper import (PAPER_STATE, DivergencePauses, PaperDivergenceLog, SymbolVenueEquity,
                                          paper_exchange, paper_state_db)


class Events:
    def __init__(self):
        self.rows = []

    def write(self, kind, **fields):
        self.rows.append((kind, fields))


def _divergences(events):
    return [f for k, f in events.rows if k is EventKind.PAPER_DIVERGENCE]


def test_an_entry_refused_for_good_is_one_divergence_and_a_deferred_one_is_none():
    events = Events()
    log = PaperDivergenceLog(events)
    log.write(EventKind.BLOCKED, symbol="BTCUSDT", order_id="o1", reason="max_orders_per_second", retry=True)
    log.write(EventKind.BLOCKED, symbol="BTCUSDT", order_id="o2", reason="kill switch active")
    log.write(EventKind.INTENT, symbol="BTCUSDT", order_id="o3")
    assert [k for k, _ in events.rows] == [EventKind.BLOCKED, EventKind.BLOCKED, EventKind.PAPER_DIVERGENCE,
                                          EventKind.INTENT]
    assert _divergences(events) == [{"cause": "entry_refused", "symbol": "BTCUSDT", "order_id": "o2",
                                     "reason": "kill switch active"}]


@pytest.mark.parametrize("kind,divergence", [("stop_triggered", True), ("stop_breached", True),
                                             ("stop_missing", False), ("unresolved_order", False),
                                             ("market_data", False)])
def test_only_a_stop_that_went_off_or_was_breached_is_a_divergence_issue(kind, divergence):
    events = Events()
    PaperDivergenceLog(events).write(EventKind.ISSUE, symbol="ETHUSDT", issue="text", issue_kind=kind)
    expected = [{"cause": kind, "symbol": "ETHUSDT", "issue": "text"}] if divergence else []
    assert _divergences(events) == expected


class Pauses:
    def __init__(self):
        self.paused = {}

    def pause(self, symbol, cause, triggers=()):
        new = symbol not in self.paused
        self.paused.setdefault(symbol, cause)
        return new

    def reason(self, symbol):
        return self.paused.get(symbol)

    def covered(self, symbol):
        return {"t1"} if symbol in self.paused else set()


def test_a_new_pause_is_one_divergence_and_a_repeated_one_is_none():
    events, inner = Events(), Pauses()
    pauses = DivergencePauses(inner, PaperDivergenceLog(events))
    assert pauses.pause("ETHUSDT", "protective stop triggered", triggers=["t1"]) is True
    assert pauses.pause("ETHUSDT", "protective stop triggered", triggers=["t2"]) is False
    assert _divergences(events) == [{"cause": "pause", "symbol": "ETHUSDT", "reason": "protective stop triggered"}]
    assert pauses.reason("ETHUSDT") == "protective stop triggered" and pauses.covered("ETHUSDT") == {"t1"}
    assert pauses.reason("BTCUSDT") is None and pauses.covered("BTCUSDT") == set()


def _spec(execution="maker", ttl=4):
    return StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, execution=execution, ttl_bars=ttl)


def test_the_simulated_exchange_is_configured_exactly_as_research_configures_its_brokers():
    spec = _spec()
    venue = paper_exchange(spec, ["BTCUSDT", "ETHUSDT"])
    for symbol in ("BTCUSDT", "ETHUSDT"):
        rules = SYMBOL_RULES[symbol]
        assert venue.broker(symbol).config == broker_config(spec, rules)
        f = venue.filters(symbol)
        assert (f.step_size, f.min_qty, f.min_notional) == (rules.step, rules.min_qty, rules.min_notional)
    with pytest.raises(RuntimeError):
        venue.broker("SOLUSDT")


def test_a_symbol_sizes_from_its_own_simulated_account_and_never_from_a_negative_one():
    venue = paper_exchange(_spec(), ["BTCUSDT", "ETHUSDT"])
    btc, eth = SymbolVenueEquity(venue, "BTCUSDT"), SymbolVenueEquity(venue, "ETHUSDT")
    assert btc.current() == eth.current() == D("5000")
    venue.broker("BTCUSDT").fees = D("6000")                   # the BTC account is gone, ETH's is not
    assert btc.current() == D(0) and eth.current() == D("5000")


def test_the_first_start_of_a_paper_record_creates_its_state_database(tmp_path):
    path, aside = paper_state_db(tmp_path / "paper", reset=False, stamp_ms=1)
    assert path == tmp_path / "paper" / schema.STATE_DB and aside is None
    assert schema.problems(path) == []
    assert paper_state_db(tmp_path / "paper", reset=False, stamp_ms=2) == (path, None)   # then it is used as is


@pytest.mark.parametrize("record", [PAPER_STATE, "paper_state.sqlite"])
def test_a_missing_database_beside_a_paper_record_is_refused_until_a_reset(tmp_path, record):
    (tmp_path / record).write_text("{}")
    with pytest.raises(RuntimeError, match="reset-paper-state"):
        paper_state_db(tmp_path, reset=False, stamp_ms=1)
    assert not (tmp_path / schema.STATE_DB).exists()
    path, aside = paper_state_db(tmp_path, reset=True, stamp_ms=1)
    assert path.exists() and aside is None


def test_a_reset_sets_the_old_database_aside_and_starts_a_new_one(tmp_path):
    path, _ = paper_state_db(tmp_path, reset=False, stamp_ms=1)
    for suffix in ("-wal", "-shm"):
        (tmp_path / f"{schema.STATE_DB}{suffix}").write_text("x")
    old = path.read_bytes()
    _, aside = paper_state_db(tmp_path, reset=True, stamp_ms=1700)
    assert aside == "state.1700.sqlite" and (tmp_path / aside).read_bytes() == old
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([schema.STATE_DB, "state.1700.sqlite",
                                                                 "state.1700.sqlite-shm", "state.1700.sqlite-wal"])
    assert schema.problems(path) == []
    _, again = paper_state_db(tmp_path, reset=True, stamp_ms=1700)    # a second reset in the same millisecond
    assert again == "state.1700.1.sqlite"
