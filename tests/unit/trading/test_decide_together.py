"""R3 of the runtime extension (docs/arch/runtime-lab-strategies-plan.md, owner 2026-11-16): decide together.

A strategy says whether it is cross-sectional. A per-symbol strategy (the overlay) decides each symbol as its bar
arrives, as before R3: a symbol that is late or failing holds back nobody else. A cross-sectional strategy decides
every symbol at one close with one call, once every symbol has that close's bar:

- while a symbol is behind, nobody decides (its decision depends on the missing bar: fail closed); the bar that
  completes the close makes the decision;
- once the leading close is too late to enter (``MAX_ENTRY_DELAY_MS``) and a symbol is still behind, that symbol's
  entries are blocked (DECISION_FAILED, once per close) in execution; paper raises, as it does for any decision that
  could not be made;
- a strategy that raises blocks every symbol's entries in execution and raises in paper;
- the exit side is not touched: the market side returns normally in execution.
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from trading.application.market_poller import MarketPoller
from trading.application.session import MAX_ENTRY_DELAY_MS
from trading.domain.bars import HOUR
from trading.domain.strategy.base import OverlayStrategy, PerSymbol
from trading.infrastructure.modes import EXECUTION, PAPER
from trading.infrastructure.paper import BarTimeClock

T0 = 1_700_006_400_000 // (8 * HOUR) * (8 * HOUR)      # a funding settlement time
SYMBOLS = ("AUSDT", "BUSDT", "CUSDT")


def _hour(ms):
    return (ms - T0) // HOUR


class Market:
    """Bars through ``last[symbol]`` (hours after T0); a settlement every 8 hours for every symbol."""

    def __init__(self, last):
        self.last = dict(last)

    def get_funding_history(self, symbol, *, limit=1000):
        return [(t, 0.0001) for t in range(T0, T0 + 48 * HOUR, 8 * HOUR)]

    def get_klines(self, symbol, interval="60", limit=200, *, now_ms=None):
        rows = [{"start": T0 + k * HOUR, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0}
                for k in range(self.last[symbol] + 1)]
        return [r for r in rows if r["start"] + HOUR <= now_ms][-limit:]


class Session:
    def __init__(self, symbol, cursor, log, strategy):
        self.symbol, self.cursor, self.log, self.strategy = symbol, T0 + cursor * HOUR, log, strategy
        self.funding, self.bars = {}, []
        self.executor = SimpleNamespace(block_entries=lambda: log.append(("blocked", symbol)))

    def advance(self, bar):
        assert bar.open_time == self.cursor + HOUR
        self.bars.append(bar)
        self.cursor = bar.open_time

    def window(self):
        return self.symbol, _hour(self.cursor)

    def funding_rate(self, bar):
        return None

    def decide(self, bar, *, now_ms, target=None):
        self.log.append(("decide", self.symbol, _hour(bar.open_time), target))

    def retry_deferred(self, now_ms):
        self.log.append(("retry", self.symbol))

    def process_bar(self, bar, *, live, now_ms):
        self.advance(bar)
        if live:
            self.decide(bar, now_ms=now_ms)


class Ranked:
    """A cross-sectional toy: one call for all symbols; the weight is the symbol's place in the sorted list."""
    name, cross_sectional = "ranked", True

    def __init__(self, log, fail=False):
        self.log, self.fail = log, fail

    def latest(self, windows):
        self.log.append(("latest", sorted(windows.values())))
        if self.fail:
            raise RuntimeError("ranking failed")
        return {symbol: (0.1 * k, True) for k, symbol in enumerate(sorted(windows))}


class Events:
    def __init__(self):
        self.rows = []

    def write(self, kind, **fields):
        self.rows.append((kind, fields))


def _poller(policy, last, *, now_hours, strategy=None, cursor=1, gate=None):
    log = []
    strategy = strategy if strategy is not None else Ranked(log)
    if isinstance(strategy, Ranked):
        strategy.log = log
    sessions = {s: Session(s, cursor, log, strategy) for s in SYMBOLS}
    clock = {"t": (T0 + now_hours * HOUR + 5_000) / 1000}
    market = Market(last)
    journal = Events()
    poller = MarketPoller(market, sessions, policy=policy, journal=journal, clock=lambda: clock["t"],
                          market_errors=(OSError,), gate_clock=gate)
    return SimpleNamespace(poller=poller, log=log, sessions=sessions, clock=clock, market=market, journal=journal)


def _all(last):
    return {s: last for s in SYMBOLS}


def _issues(world):
    return [f for kind, f in world.journal.rows if kind.value == "issue"]


def test_a_cross_sectional_strategy_decides_every_symbol_at_one_close_with_one_call():
    w = _poller(EXECUTION, _all(3), now_hours=4)
    assert w.poller.poll() == 6                                       # bars 2 and 3 of three symbols
    assert w.log == [("latest", [("AUSDT", 3), ("BUSDT", 3), ("CUSDT", 3)]),
                     ("decide", "AUSDT", 3, (0.0, True)), ("decide", "BUSDT", 3, (0.1, True)),
                     ("decide", "CUSDT", 3, (0.2, True))]


def test_without_a_new_bar_nothing_is_decided_again():
    w = _poller(EXECUTION, _all(3), now_hours=4, cursor=3)            # warmed up through bar 3
    assert w.poller.poll() == 0 and w.log == []
    w = _poller(EXECUTION, _all(3), now_hours=4)
    w.poller.poll()
    w.clock["t"] += 5
    w.poller.poll()                                                   # the same close is not decided twice
    assert [e[0] for e in w.log].count("latest") == 1


def test_while_a_symbol_is_behind_nobody_decides_and_its_bar_completes_the_close():
    w = _poller(EXECUTION, {"AUSDT": 3, "BUSDT": 3, "CUSDT": 2}, now_hours=4)
    w.poller.poll()
    assert w.log == [] and w.sessions["CUSDT"].cursor == T0 + 2 * HOUR
    w.market.last["CUSDT"] = 3
    w.clock["t"] += 5
    assert w.poller.poll() == 1
    assert w.log == [("latest", [("AUSDT", 3), ("BUSDT", 3), ("CUSDT", 3)]),
                     ("decide", "AUSDT", 3, (0.0, True)), ("decide", "BUSDT", 3, (0.1, True)),
                     ("decide", "CUSDT", 3, (0.2, True))]
    assert w.poller.decision_errors == {} and _issues(w) == []


def test_a_symbol_still_behind_when_the_close_is_too_late_has_its_entries_blocked_once():
    w = _poller(EXECUTION, {"AUSDT": 3, "BUSDT": 3, "CUSDT": 2}, now_hours=4)
    w.poller.poll()
    w.clock["t"] = (T0 + 4 * HOUR + MAX_ENTRY_DELAY_MS) / 1000       # exactly at the limit: still waiting
    w.poller.poll()
    assert _issues(w) == []
    w.clock["t"] += 1
    w.poller.poll()
    w.clock["t"] += 5
    w.poller.poll()
    issues = _issues(w)
    assert [(i["symbol"], i["issue_kind"]) for i in issues] == [("CUSDT", "decision_failed")]
    assert "CUSDT" in issues[0]["issue"] and "together" in issues[0]["issue"]
    assert set(w.poller.decision_errors) == {"CUSDT"} and w.log == [("blocked", "CUSDT")]
    w.market.last["CUSDT"] = 3                                        # the bar comes after all: a late decision
    w.poller.poll()                                                   # may still reduce (session.decide)
    assert [e[0] for e in w.log] == ["blocked", "latest", "decide", "decide", "decide"]
    assert w.poller.decision_errors == {}


def test_paper_raises_when_a_symbol_is_still_behind_a_late_close():
    gate = BarTimeClock()
    w = _poller(PAPER, {"AUSDT": 3, "BUSDT": 3, "CUSDT": 2}, now_hours=4, gate=gate)
    w.poller.poll()
    gate.at(T0 + 4 * HOUR + MAX_ENTRY_DELAY_MS + 1)
    with pytest.raises(RuntimeError, match="CUSDT"):
        w.poller.poll()


def test_a_symbol_whose_bar_fails_holds_back_the_cross_sectional_decision():
    w = _poller(EXECUTION, _all(3), now_hours=4)

    def broken(bar):
        raise RuntimeError("malformed bar")
    w.sessions["BUSDT"].advance = broken
    w.poller.poll()
    assert set(w.poller.decision_errors) == {"BUSDT"}
    assert not [e for e in w.log if e[0] in ("latest", "decide")]


def test_a_failing_cross_sectional_strategy_blocks_every_symbol_in_execution_and_raises_in_paper():
    w = _poller(EXECUTION, _all(3), now_hours=4, strategy=Ranked(None, fail=True))
    w.poller.poll()
    assert set(w.poller.decision_errors) == set(SYMBOLS)
    assert [e for e in w.log if e[0] == "blocked"] == [("blocked", s) for s in SYMBOLS]
    assert not [e for e in w.log if e[0] == "decide"]
    w = _poller(PAPER, _all(3), now_hours=4, strategy=Ranked(None, fail=True), gate=BarTimeClock())
    with pytest.raises(RuntimeError, match="ranking failed"):
        w.poller.poll()


def test_the_same_failure_at_the_next_close_is_not_written_again():
    w = _poller(EXECUTION, _all(3), now_hours=4, strategy=Ranked(None, fail=True))
    w.poller.poll()
    w.market.last = _all(4)
    w.clock["t"] += HOUR / 1000
    w.poller.poll()
    assert [e[0] for e in w.log].count("latest") == 2
    assert [i["symbol"] for i in _issues(w)] == list(SYMBOLS)              # written once, blocked at both closes
    assert [e for e in w.log if e[0] == "blocked"] == [("blocked", s) for s in SYMBOLS] * 2


def test_a_symbol_the_strategy_leaves_out_is_blocked_and_the_others_decide():
    class Partial(Ranked):
        def latest(self, windows):
            out = super().latest(windows)
            out.pop("BUSDT")
            return out
    w = _poller(EXECUTION, _all(3), now_hours=4, strategy=Partial(None))
    w.poller.poll()
    assert set(w.poller.decision_errors) == {"BUSDT"}
    assert [e[1] for e in w.log if e[0] == "decide"] == ["AUSDT", "CUSDT"]


def test_paper_decides_together_at_every_close_it_replays():
    w = _poller(PAPER, _all(3), now_hours=4, gate=BarTimeClock())
    w.poller.poll()
    closes = [e for e in w.log if e[0] == "latest"]
    assert closes == [("latest", [(s, h) for s in SYMBOLS]) for h in (2, 3)]
    assert [(e[1], e[2]) for e in w.log if e[0] == "decide"] == [(s, h) for h in (2, 3) for s in SYMBOLS]


def test_paper_waits_for_a_symbol_whose_bar_comes_a_poll_later():
    w = _poller(PAPER, {"AUSDT": 3, "BUSDT": 3, "CUSDT": 2}, now_hours=4, gate=BarTimeClock())
    w.poller.poll()
    assert [e[2] for e in w.log if e[0] == "decide" and e[1] == "AUSDT"] == [2]
    w.market.last["CUSDT"] = 3
    w.clock["t"] += 5
    before = len(w.log)
    w.poller.poll()
    assert [(e[1], e[2]) for e in w.log if e[0] == "decide"][3:] == [(s, 3) for s in SYMBOLS]
    assert [e for e in w.log[before:] if e[0] == "retry"] == [("retry", s) for s in SYMBOLS]   # all decided now


def test_a_per_symbol_strategy_still_decides_each_symbol_on_its_own():
    class Own(PerSymbol):
        name = "own"
    w = _poller(EXECUTION, {"AUSDT": 3, "BUSDT": 3, "CUSDT": 2}, now_hours=4, strategy=Own())
    w.poller.poll()
    assert w.log == [("decide", "AUSDT", 3, None), ("decide", "BUSDT", 3, None), ("decide", "CUSDT", 2, None)]
    assert Own.cross_sectional is False and OverlayStrategy.cross_sectional is False


def test_a_cross_sectional_strategy_must_be_the_one_every_symbol_uses():
    w = _poller(EXECUTION, _all(3), now_hours=4)
    w.sessions["CUSDT"].strategy = Ranked(w.log)
    with pytest.raises(ValueError, match="every symbol"):
        w.poller.poll()


def test_the_session_decides_on_a_given_target_without_asking_its_strategy():
    import numpy as np
    from trading.application.session import TargetSession
    from trading.domain.strategy.overlay import StrategySpec

    class Refuses(PerSymbol):
        name = "refuses"

        def window_hours(self):
            return 30

        def funding_window(self):
            return 4

        def symbol_history(self, window):
            raise AssertionError("the strategy was asked")

    class Parts:
        executor = consumer = stops = gate = flattener = equity = None
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
    session = TargetSession("AUSDT", spec, journal=None, rules=None, make_order=None, execution=Parts(),
                            strategy=Refuses())
    for k in range(40):
        t = T0 + k * HOUR
        session.bars.append(SimpleNamespace(open_time=t, available_at=t + HOUR, open=D(1), high=D(1), low=D(1),
                                            close=D(1), volume=D(1)))
    session.funding = {T0 + k * 8 * HOUR: 1e-4 for k in range(7)}          # two settle after the newest close
    session.decide(session.bars[-1], now_ms=session.bars[-1].available_at, target=(0.3, False))
    assert session.last_target == (0.3, False)
    window = session.window()
    assert len(window.hourly) == 40 and np.array_equal(window.hourly["available_at"].to_numpy()[-1:],
                                                       [T0 + 40 * HOUR])
    # the strategy's 4 settlements known at the close (T0 + 40 h, one settles then); none of the later ones
    assert list(window.funding.index) == [T0 + k * 8 * HOUR for k in range(2, 6)]
