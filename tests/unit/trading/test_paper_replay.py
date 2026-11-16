"""The market side by ``ModePolicy`` (ADR-0013, S10 part 3), with fakes for the market, the sessions and the venue.

Paper (``replay_all``, ``settle_new_bar_before_decision``, ``limiter_clock == "bar_time"``): the new bars of every
symbol are replayed in time order across the symbols. At each bar time the windows advance, the venue matches the
bar, the exit side settles it, the sessions decide at the bar's close, and an entry the per-second limit deferred is
asked again 5 s later in simulated time (S10 decision 2). Execution (``latest_only``) decides on the newest bar only,
on the runtime clock, and has no venue to match on.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from trading.application.market_poller import POLL_SECONDS, MarketPoller
from trading.domain.bars import HOUR
from trading.infrastructure.modes import EXECUTION, PAPER
from trading.infrastructure.paper import BarTimeClock

T0 = 1_700_006_400_000 // (8 * HOUR) * (8 * HOUR)      # a funding settlement time


def _row(start):
    return {"start": start, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0}


class Market:
    def __init__(self, last):
        self.rows = [_row(T0 + k * HOUR) for k in range(last + 1)]

    def get_funding_history(self, symbol, *, limit=1000):
        return [(t, 0.0001) for t in range(T0, T0 + 48 * HOUR, 8 * HOUR)]

    def get_klines(self, symbol, interval="60", limit=200, *, now_ms=None):
        return [r for r in self.rows if r["start"] + HOUR <= now_ms][-limit:]


class Session:
    def __init__(self, symbol, cursor, log):
        self.symbol, self.cursor, self.log = symbol, T0 + cursor * HOUR, log
        self.funding, self.bars = {}, deque()
        self.executor = SimpleNamespace(block_entries=lambda: log.append(("blocked", symbol)))

    def _hour(self, bar):
        return (bar.open_time - T0) // HOUR

    def advance(self, bar):
        assert bar.open_time == self.cursor + HOUR
        self.cursor = bar.open_time
        self.log.append(("advance", self.symbol, self._hour(bar)))

    def funding_rate(self, bar):
        return D("0.0001") if bar.open_time in self.funding else None

    def decide(self, bar, *, now_ms):
        self.log.append(("decide", self.symbol, self._hour(bar), now_ms - bar.available_at))

    def retry_deferred(self, now_ms):
        self.log.append(("retry", self.symbol, now_ms))

    def process_bar(self, bar, *, live, now_ms):
        self.advance(bar)
        self.log.append(("process", self.symbol, self._hour(bar), live))


class Venue:
    def __init__(self, log):
        self.log = log

    def on_closed_bar(self, symbol, bar, funding_rate):
        self.log.append(("match", symbol, (bar.open_time - T0) // HOUR, funding_rate))


class Events:
    def __init__(self):
        self.rows = []

    def write(self, kind, **fields):
        self.rows.append((kind, fields))


def _poller(policy, cursors, *, now_hours, gate=None, last=12):
    log = []
    sessions = {symbol: Session(symbol, cursor, log) for symbol, cursor in cursors.items()}
    clock = {"t": (T0 + now_hours * HOUR + 5_000) / 1000}
    poller = MarketPoller(Market(last), sessions, policy=policy, journal=Events(), clock=lambda: clock["t"],
                          market_errors=(OSError,), listeners=(Venue(log),),
                          settle=lambda: log.append(("settle",)), gate_clock=gate,
                          on_step=lambda group: log.append(("step", [(s, (b.open_time - T0) // HOUR)
                                                                     for s, b in group])))
    return poller, log, sessions


def test_paper_replays_every_new_bar_in_time_order_across_the_symbols():
    gate = BarTimeClock()
    poller, log, sessions = _poller(PAPER, {"BTCUSDT": 1, "ETHUSDT": 1}, now_hours=4, gate=gate)
    assert poller.poll() == 4
    expected = []
    for hour in (2, 3):
        close = T0 + (hour + 1) * HOUR
        expected += [("advance", "BTCUSDT", hour), ("advance", "ETHUSDT", hour),
                     ("match", "BTCUSDT", hour, None), ("match", "ETHUSDT", hour, None), ("settle",),
                     ("decide", "BTCUSDT", hour, 0), ("decide", "ETHUSDT", hour, 0),
                     ("retry", "BTCUSDT", close + 5_000), ("retry", "ETHUSDT", close + 5_000),
                     ("step", [("BTCUSDT", hour), ("ETHUSDT", hour)])]
    assert log == expected
    assert POLL_SECONDS == 5.0                           # the retry is one poll later, in simulated time


def test_the_venue_is_told_the_funding_settled_at_the_bar_s_open():
    poller, log, sessions = _poller(PAPER, {"BTCUSDT": 7, "ETHUSDT": 7}, now_hours=9, gate=BarTimeClock())
    poller.poll()                                        # bar 8 opens on a settlement, bar 9 has not closed
    assert [e for e in log if e[0] == "match"] == [("match", "BTCUSDT", 8, D("0.0001")),
                                                    ("match", "ETHUSDT", 8, D("0.0001"))]


def test_a_symbol_without_a_bar_at_a_time_is_left_out_of_that_step():
    poller, log, sessions = _poller(PAPER, {"BTCUSDT": 1, "ETHUSDT": 2}, now_hours=4, gate=BarTimeClock())
    poller.poll()
    assert [e for e in log if e[0] == "step"] == [("step", [("BTCUSDT", 2)]),
                                                  ("step", [("BTCUSDT", 3), ("ETHUSDT", 3)])]
    assert [e[1:] for e in log if e[0] == "match"][:1] == [("BTCUSDT", 2, None)]


def test_the_gate_clock_follows_the_bars_and_never_goes_back():
    gate = BarTimeClock()
    gate.at(T0 + 10 * HOUR)
    poller, log, sessions = _poller(PAPER, {"BTCUSDT": 1, "ETHUSDT": 1}, now_hours=3, gate=gate)
    poller.poll()                                        # bar 2 closes before the clock's time: it stays ahead
    assert [e[3] for e in log if e[0] == "decide"] == [T0 + 10 * HOUR - (T0 + 3 * HOUR)] * 2
    assert [e[2] for e in log if e[0] == "retry"] == [T0 + 10 * HOUR + 5_000] * 2
    assert gate.ms() == T0 + 10 * HOUR + 5_000 and gate() == gate.ms() / 1000


def test_without_a_gate_clock_a_replay_decides_on_the_runtime_clock():
    policy = dataclasses.replace(PAPER, limiter_clock="monotonic")
    poller, log, sessions = _poller(policy, {"BTCUSDT": 1, "ETHUSDT": 1}, now_hours=3)
    poller.poll()
    now = T0 + 3 * HOUR + 5_000
    assert [e[3] for e in log if e[0] == "decide"] == [now - (T0 + 3 * HOUR)] * 2
    assert not [e for e in log if e[0] == "retry"]       # execution retries on the next cycle instead


def test_settling_before_the_decision_is_the_policy_s_choice():
    policy = dataclasses.replace(PAPER, settle_new_bar_before_decision=False)
    poller, log, sessions = _poller(policy, {"BTCUSDT": 1, "ETHUSDT": 1}, now_hours=3, gate=BarTimeClock())
    poller.poll()
    assert ("settle",) not in log and [e[0] for e in log].count("match") == 2


def test_execution_decides_on_the_newest_bar_only_and_matches_nothing():
    poller, log, sessions = _poller(EXECUTION, {"BTCUSDT": 1, "ETHUSDT": 1}, now_hours=4)
    assert poller.poll() == 4
    assert log == [("advance", "BTCUSDT", 2), ("process", "BTCUSDT", 2, False),
                   ("advance", "BTCUSDT", 3), ("process", "BTCUSDT", 3, True),
                   ("advance", "ETHUSDT", 2), ("process", "ETHUSDT", 2, False),
                   ("advance", "ETHUSDT", 3), ("process", "ETHUSDT", 3, True)]


def test_paper_raises_a_failing_decision_and_execution_blocks_only_that_symbol():
    poller, log, sessions = _poller(PAPER, {"BTCUSDT": 1, "ETHUSDT": 1}, now_hours=3, gate=BarTimeClock())

    def broken(bar, *, now_ms):
        raise RuntimeError("decision failed")
    sessions["BTCUSDT"].decide = broken
    with pytest.raises(RuntimeError, match="decision failed"):
        poller.poll()
    poller, log, sessions = _poller(EXECUTION, {"BTCUSDT": 1, "ETHUSDT": 1}, now_hours=3)

    def broken_process(bar, *, live, now_ms):
        raise RuntimeError("decision failed")
    sessions["BTCUSDT"].process_bar = broken_process
    poller.poll()
    assert ("blocked", "BTCUSDT") in log and set(poller.decision_errors) == {"BTCUSDT"}
    assert ("process", "ETHUSDT", 2, True) in log


def test_bar_time_clock():
    gate = BarTimeClock()
    assert gate.ms() == 0
    gate.at(5_000)
    gate.at(4_000)
    assert gate.ms() == 5_000
    gate.advance(2.5)
    assert gate.ms() == 7_500 and gate() == 7.5

