"""R4 of the runtime extension (docs/arch/runtime-lab-strategies-plan.md, owner 2026-11-16): each symbol's own funding
interval, read from the exchange; a symbol whose interval is not known takes no entries.

Binance settles funding every 8, 4 or 1 hours depending on the symbol (``GET /fapi/v1/fundingInfo``; on 2026-11-16 it
listed every trading USDT perpetual: 466 rows at 4 h, 355 at 8 h, 3 at 1 h). The runtime waits for the settlement at
or before each close (up to ``FUNDING_WAIT_MS``) and refuses entries until it is published; both now use the symbol's
interval. The poller reads the intervals at warm-up and in the first poll of every hour; a failed read keeps the last
one for 2 hours (retried once a minute), then every interval is unknown. A symbol the exchange does not list is
unknown at once.
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from trading.adapters.binance.adapter import BinanceAdapter, BinanceConfig
from trading.application.market_poller import FUNDING_WAIT_MS, MarketPoller
from trading.domain.bars import HOUR
from trading.domain.strategy.base import PerSymbol
from trading.infrastructure.modes import EXECUTION

T0 = 1_700_006_400_000 // (8 * HOUR) * (8 * HOUR)      # a settlement time for every interval


# -- the adapter ---------------------------------------------------------------------------------------------------
class Client:
    def __init__(self, data):
        self.data, self.calls = data, []

    def request_public(self, *, method, path, params=None):
        self.calls.append((method, path))
        return self.data


def _adapter(data):
    client = Client(data)
    return BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=client), client


def test_the_adapter_reads_every_listed_symbol_s_interval_in_hours():
    adapter, client = _adapter([
        {"symbol": "BTCUSDT", "fundingIntervalHours": 8, "adjustedFundingRateCap": "0.02", "disclaimer": False},
        {"symbol": "DOGSUSDT", "fundingIntervalHours": 4}, {"symbol": "XUSDT", "fundingIntervalHours": 1}])
    assert adapter.get_funding_intervals() == {"BTCUSDT": 8, "DOGSUSDT": 4, "XUSDT": 1}
    assert client.calls == [("GET", "/fapi/v1/fundingInfo")]


@pytest.mark.parametrize("value", [0, -4, 5, 48, "4", 4.0, True, None])
def test_an_interval_that_cannot_be_used_leaves_the_symbol_out(value):
    adapter, _ = _adapter([{"symbol": "BTCUSDT", "fundingIntervalHours": value},
                           {"symbol": "ETHUSDT", "fundingIntervalHours": 8}, {"fundingIntervalHours": 8}])
    assert adapter.get_funding_intervals() == {"ETHUSDT": 8}


@pytest.mark.parametrize("data", [None, {}, "x", [None, 3]])
def test_an_answer_that_is_not_a_list_of_rows_gives_no_interval(data):
    adapter, _ = _adapter(data)
    assert adapter.get_funding_intervals() == {}


# -- the session ---------------------------------------------------------------------------------------------------
def _session(interval):
    from trading.application.session import TargetSession
    from trading.domain.strategy.overlay import StrategySpec

    class Parts:
        executor = consumer = stops = gate = flattener = equity = None
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
    session = TargetSession("AUSDT", spec, journal=None, rules=None, make_order=None, execution=Parts())
    session.funding_interval = interval
    return session


def _decide(session, close):
    bar = SimpleNamespace(open_time=close - HOUR, available_at=close, close=D(1))
    session.decide(bar, now_ms=close, target=(0.0, False))
    return session.entry_ready, session.entry_reason


def test_a_session_starts_with_an_unknown_interval_and_refuses_entries():
    session = _session(None)
    assert _session.__name__ and session.funding_interval is None
    session.funding = {T0: 1e-4, T0 + 4 * HOUR: 1e-4}
    assert _decide(session, T0 + 4 * HOUR) == (False, "funding interval not known")


def test_the_settlement_a_close_waits_for_follows_the_symbol_s_interval():
    four, eight = _session(4 * HOUR), _session(8 * HOUR)
    for session in (four, eight):
        session.funding = {T0: 1e-4}                     # the 04:00 settlement is not published yet
    assert _decide(four, T0 + 4 * HOUR) == (False, "funding settlement at or before this close not published yet")
    assert _decide(eight, T0 + 4 * HOUR) == (True, "")   # an 8-hour symbol has no settlement at 04:00
    four.funding[T0 + 4 * HOUR] = 1e-4
    assert _decide(four, T0 + 5 * HOUR) == (True, "")
    hourly = _session(HOUR)
    hourly.funding = {T0 + k * HOUR: 1e-4 for k in range(5)}
    assert _decide(hourly, T0 + 5 * HOUR)[0] is False and _decide(hourly, T0 + 4 * HOUR)[0] is True


# -- the poller ----------------------------------------------------------------------------------------------------
class Market:
    def __init__(self, last, settlements, intervals):
        self.last, self.settlements, self.intervals = last, settlements, intervals
        self.interval_reads, self.funding_limits, self.fail = 0, [], False

    def get_funding_intervals(self):
        self.interval_reads += 1
        if self.fail:
            raise OSError("down")
        return dict(self.intervals)

    def get_funding_history(self, symbol, *, limit=1000):
        self.funding_limits.append((symbol, limit))
        return [(t, 0.0001) for t in self.settlements[symbol]]

    def get_klines(self, symbol, interval="60", limit=200, *, now_ms=None):
        rows = [{"start": T0 + k * HOUR, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
                for k in range(self.last + 1)]
        return [r for r in rows if r["start"] + HOUR <= now_ms][-limit:]


class Session:
    strategy = PerSymbol()

    def __init__(self, symbol, cursor, log):
        self.symbol, self.cursor, self.log = symbol, T0 + cursor * HOUR, log
        self.funding, self.bars, self.funding_interval = {}, [], "never set"
        self.executor = SimpleNamespace(block_entries=lambda: None)

    def process_bar(self, bar, *, live, now_ms):
        assert bar.open_time == self.cursor + HOUR
        self.cursor = bar.open_time
        if live:
            self.log.append((self.symbol, (bar.open_time - T0) // HOUR))


class Events:
    def __init__(self):
        self.rows = []

    def write(self, kind, **fields):
        self.rows.append((kind.value, fields))


def _poller(*, cursor, now_ms, last, settlements, intervals):
    log, journal = [], Events()
    sessions = {s: Session(s, cursor, log) for s in ("FOURUSDT", "EIGHTUSDT")}
    market = Market(last, settlements, intervals)
    clock = {"ms": now_ms}
    poller = MarketPoller(market, sessions, policy=EXECUTION, journal=journal, clock=lambda: clock["ms"] / 1000,
                          market_errors=(OSError,))
    return SimpleNamespace(poller=poller, log=log, sessions=sessions, market=market, clock=clock, journal=journal)


def test_the_poller_waits_for_each_symbol_s_own_settlement():
    close = T0 + 4 * HOUR                                # bar 3 closes; only a 4-hour symbol settles now
    w = _poller(cursor=2, now_ms=close + 5_000, last=3, settlements={"FOURUSDT": [T0], "EIGHTUSDT": [T0]},
                intervals={"FOURUSDT": 4, "EIGHTUSDT": 8})
    w.poller.poll()
    assert w.log == [("EIGHTUSDT", 3)]                   # the 4-hour symbol waits for its 04:00 settlement
    assert w.sessions["FOURUSDT"].funding_interval == 4 * HOUR
    assert w.sessions["EIGHTUSDT"].funding_interval == 8 * HOUR
    w.market.settlements["FOURUSDT"] = [T0, close]
    w.clock["ms"] += 5_000
    w.poller.poll()
    assert w.log == [("EIGHTUSDT", 3), ("FOURUSDT", 3)]


def test_the_wait_ends_after_two_minutes_and_an_unknown_interval_does_not_wait():
    close = T0 + 4 * HOUR
    w = _poller(cursor=2, now_ms=close + FUNDING_WAIT_MS, last=3,
                settlements={"FOURUSDT": [T0], "EIGHTUSDT": [T0]}, intervals={"FOURUSDT": 4, "EIGHTUSDT": 8})
    w.poller.poll()
    assert sorted(w.log) == [("EIGHTUSDT", 3), ("FOURUSDT", 3)]
    w = _poller(cursor=2, now_ms=close + 5_000, last=3, settlements={"FOURUSDT": [T0], "EIGHTUSDT": [T0]},
                intervals={"EIGHTUSDT": 8})          # FOURUSDT not listed: unknown, its entries are refused anyway
    w.poller.poll()
    assert sorted(w.log) == [("EIGHTUSDT", 3), ("FOURUSDT", 3)] and w.sessions["FOURUSDT"].funding_interval is None


def test_the_funding_request_covers_the_missed_hours_at_the_symbol_s_interval():
    w = _poller(cursor=2, now_ms=T0 + 26 * HOUR + 5_000, last=25,
                settlements={"FOURUSDT": [T0], "EIGHTUSDT": [T0]}, intervals={"FOURUSDT": 1, "EIGHTUSDT": 8})
    w.poller.poll()                                      # 23 hours missed since bar 2
    assert dict(w.market.funding_limits) == {"FOURUSDT": 3 + 23, "EIGHTUSDT": 3 + 23 // 8}
    w = _poller(cursor=2, now_ms=T0 + 26 * HOUR + 5_000, last=25,
                settlements={"FOURUSDT": [T0], "EIGHTUSDT": [T0]}, intervals={"EIGHTUSDT": 8})
    w.poller.poll()
    assert dict(w.market.funding_limits)["FOURUSDT"] == 3 + 23        # unknown: as many as an hourly symbol


def test_the_intervals_are_read_in_the_first_poll_of_every_hour():
    w = _poller(cursor=3, now_ms=T0 + 4 * HOUR + 5_000, last=3, settlements={"FOURUSDT": [T0], "EIGHTUSDT": [T0]},
                intervals={"FOURUSDT": 4, "EIGHTUSDT": 8})
    w.poller.poll()
    w.clock["ms"] += 30 * 60_000
    w.poller.poll()
    assert w.market.interval_reads == 1
    w.market.intervals["EIGHTUSDT"] = 4                  # the exchange changed it
    w.clock["ms"] = T0 + 5 * HOUR + 1_000
    w.poller.poll()
    assert w.market.interval_reads == 2 and w.sessions["EIGHTUSDT"].funding_interval == 4 * HOUR


def test_a_failed_read_keeps_the_last_intervals_for_two_hours_then_none_is_known():
    w = _poller(cursor=3, now_ms=T0 + 4 * HOUR + 5_000, last=3, settlements={"FOURUSDT": [T0], "EIGHTUSDT": [T0]},
                intervals={"FOURUSDT": 4, "EIGHTUSDT": 8})
    w.poller.poll()
    read_at = w.clock["ms"]
    w.market.fail = True
    w.clock["ms"] = T0 + 5 * HOUR + 1_000
    w.poller.poll()
    w.clock["ms"] += 30_000
    w.poller.poll()                                      # retried once a minute, not every poll
    assert w.market.interval_reads == 2
    w.clock["ms"] += 30_000
    w.poller.poll()
    assert w.market.interval_reads == 3 and w.sessions["FOURUSDT"].funding_interval == 4 * HOUR
    w.clock["ms"] = read_at + 2 * HOUR
    w.poller.poll()
    assert w.sessions["FOURUSDT"].funding_interval == 4 * HOUR       # exactly 2 hours: still kept
    w.clock["ms"] += 60_000
    w.poller.poll()
    assert all(s.funding_interval is None for s in w.sessions.values())
    issues = [f for kind, f in w.journal.rows if kind == "issue"]
    assert [(i["symbol"], i["issue_kind"]) for i in issues] == [("FOURUSDT", "funding_interval_unknown"),
                                                               ("EIGHTUSDT", "funding_interval_unknown")]
    assert all("not read for 2 hours" in i["issue"] for i in issues)
    assert w.poller.symbol_issue("FOURUSDT") is not None                   # the exit side blocks and alerts it
    w.clock["ms"] += 60_000
    w.poller.poll()
    assert len([1 for kind, _ in w.journal.rows if kind == "issue"]) == 2      # said once per symbol
    w.market.fail = False
    w.clock["ms"] += 60_000
    w.poller.poll()
    assert w.sessions["FOURUSDT"].funding_interval == 4 * HOUR
    w.market.fail = True                                 # a second outage is treated like the first
    w.clock["ms"] += 2 * HOUR + 60_000
    w.poller.poll()
    w.clock["ms"] += 60_000
    w.poller.poll()
    assert all(s.funding_interval is None for s in w.sessions.values())
    assert len([1 for kind, _ in w.journal.rows if kind == "issue"]) == 4


def test_a_symbol_the_exchange_does_not_list_is_unknown_and_said_once_per_read():
    w = _poller(cursor=3, now_ms=T0 + 4 * HOUR + 5_000, last=3, settlements={"FOURUSDT": [T0], "EIGHTUSDT": [T0]},
                intervals={"EIGHTUSDT": 8})
    w.poller.poll()
    w.clock["ms"] += 10_000
    w.poller.poll()
    issues = [f for kind, f in w.journal.rows if kind == "issue"]
    assert [(i["symbol"], i["issue_kind"]) for i in issues] == [("FOURUSDT", "funding_interval_unknown")]
    assert w.sessions["FOURUSDT"].funding_interval is None
    assert w.poller.symbol_issue("FOURUSDT").kind.value == "funding_interval_unknown"
    assert w.poller.symbol_issue("EIGHTUSDT") is None
