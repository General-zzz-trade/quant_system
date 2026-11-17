"""The strategy's funding income kept in the state database (tech lead 2026-11-17: "累计资金费存进状态库").

Binance keeps the income history for three months only, so a strategy whose first fill is older could not know its
funding (the 88-day guard: the base sized to zero). Now every funding row read is kept, with how far each symbol's
rows are known whole (``funding_coverage``), and each refresh reads only from shortly before that point on.

- State database version 2 adds the two tables; ``schema.migrate`` brings a version-1 database there.
- The store keeps each row once (by Binance's transaction id), refuses to leave a gap in a symbol's coverage, and
  sums a symbol's rows from a time on.
- ``StrategyFunding`` with the store: the first read from the first fill, later reads from 8 hours before the
  coverage's end; the total is the store's. A gap longer than Binance's history (the reads fail) leaves the total
  unknown to the sizing base after two hours, as before; the loss stop's total is the store's, also after a restart.
"""
from __future__ import annotations

import sqlite3
from decimal import Decimal as D

import pytest

from tests.support.state_db import state_db
from trading.adapters.sqlite import schema
from trading.adapters.sqlite.funding_store import SqliteFundingStore
from trading.application.equity import FUNDING_MAX_AGE, StrategyFunding

SCOPE = "binance-testnet-ut"
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class Venue:
    """Funding rows (tran id, time ms, amount) per symbol; serves those inside a window, like Binance, and refuses a
    window that starts more than 88 days back. Records every request."""

    def __init__(self, rows):
        self.rows, self.calls, self.down = {s: list(r) for s, r in rows.items()}, [], False

    def get_funding_rows(self, symbol, *, since_ms, now_ms):
        self.calls.append((symbol, since_ms, now_ms))
        if self.down:
            raise OSError("HTTP 503")
        if since_ms < now_ms - 88 * DAY_MS:
            raise RuntimeError("Binance keeps income history for three months only")
        return [r for r in self.rows.get(symbol, []) if since_ms <= r[1] <= now_ms]


# -- the schema ---------------------------------------------------------------------------------------------------
def _version_1(path):
    """A state database as version 1 made it: every table but the funding ones."""
    state_db(path)
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE funding_rows")
        db.execute("DROP TABLE funding_coverage")
        db.execute("PRAGMA user_version=1")


def test_version_2_has_the_funding_tables(tmp_path):
    assert schema.VERSION == 2 and {"funding_rows", "funding_coverage"} <= set(schema.TABLES)
    assert schema.problems(state_db(tmp_path / schema.STATE_DB)) == []


def test_a_version_1_database_is_told_to_migrate_and_migrate_brings_it_to_version_2(tmp_path):
    path = tmp_path / schema.STATE_DB
    _version_1(path)
    found = schema.problems(path)
    assert len(found) == 1 and "version 1, expected 2" in found[0] and "migrate" in found[0]   # only the version
    assert schema.migrate(path) == 1
    assert schema.problems(path) == []
    assert schema.migrate(path) == 2                              # already there: nothing changes


def test_migrate_refuses_a_version_it_does_not_know(tmp_path):
    path = tmp_path / schema.STATE_DB
    state_db(path)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=7")
    with pytest.raises(ValueError, match="version 7"):
        schema.migrate(path)


# -- the store ----------------------------------------------------------------------------------------------------
def _store(tmp_path):
    return SqliteFundingStore(state_db(tmp_path / schema.STATE_DB), SCOPE)


def test_the_store_keeps_each_row_once_and_sums_from_a_time(tmp_path):
    store = _store(tmp_path)
    assert store.coverage("BTCUSDT") is None
    store.add("BTCUSDT", [("1", 100, D("-1.5")), ("2", 200, D("0.25"))], from_ms=50, through_ms=250)
    store.add("BTCUSDT", [("2", 200, D("0.25")), ("3", 300, D("2"))], from_ms=150, through_ms=350)   # overlap
    assert store.coverage("BTCUSDT") == (50, 350)
    assert store.total("BTCUSDT", since_ms=0) == D("0.75") and store.total("BTCUSDT", since_ms=200) == D("2.25")
    assert store.total("ETHUSDT", since_ms=0) == D(0)


def test_the_store_refuses_a_gap_and_a_reread_from_earlier_covers_it_all(tmp_path):
    store = _store(tmp_path)
    store.add("BTCUSDT", [], from_ms=100, through_ms=200)
    with pytest.raises(ValueError, match="gap"):
        store.add("BTCUSDT", [], from_ms=201, through_ms=300)
    assert store.coverage("BTCUSDT") == (100, 200)
    store.add("BTCUSDT", [("9", 60, D("1"))], from_ms=40, through_ms=400)    # read again from before: all of it
    assert store.coverage("BTCUSDT") == (40, 400) and store.total("BTCUSDT", since_ms=0) == D("1")


def test_rows_and_coverage_are_kept_by_scope(tmp_path):
    path = state_db(tmp_path / schema.STATE_DB)
    SqliteFundingStore(path, SCOPE).add("BTCUSDT", [("1", 100, D("1"))], from_ms=0, through_ms=200)
    other = SqliteFundingStore(path, "binance-testnet-other")
    assert other.coverage("BTCUSDT") is None and other.total("BTCUSDT", since_ms=0) == D(0)


# -- StrategyFunding with the store -------------------------------------------------------------------------------
def _funding(tmp_path, venue, first_fills, clock):
    store = SqliteFundingStore(tmp_path / schema.STATE_DB, SCOPE) if (tmp_path / schema.STATE_DB).exists() \
        else _store(tmp_path)
    return StrategyFunding(venue, ["BTCUSDT", "ETHUSDT"], first_fills.get, clock, store=store), store


def test_the_first_read_is_from_the_first_fill_and_later_ones_from_24_hours_before_the_coverage_end(tmp_path):
    clock = Clock(1_000_000.0)
    venue = Venue({"BTCUSDT": [("a", 600_000_000, D("-2")), ("b", 990_000_000, D("1"))]})
    funding, store = _funding(tmp_path, venue, {"BTCUSDT": 500_000.0}, clock)
    assert funding.current() == D("-1")
    assert venue.calls == [("BTCUSDT", 500_000_000, 1_000_000_000)]
    clock.now += 600
    venue.rows["BTCUSDT"].append(("c", 1_000_300_000, D("0.5")))
    assert funding.current() == D("-0.5")
    assert venue.calls[-1] == ("BTCUSDT", 1_000_000_000 - 24 * HOUR_MS, 1_000_600_000)
    assert store.coverage("BTCUSDT") == (500_000_000, 1_000_600_000)


def test_months_later_the_total_is_still_known_whole(tmp_path):
    """The point of the store: the first fill is far beyond Binance's three months, every refresh reads a short
    window, and the total keeps every row since the first fill."""
    clock = Clock(1_000_000.0)
    rows = [(str(k), 1_000_000_000 + k * 8 * HOUR_MS, D("-0.1")) for k in range(1, 400)]   # 133 days of settlements
    venue = Venue({"BTCUSDT": rows})
    funding, store = _funding(tmp_path, venue, {"BTCUSDT": 1_000_000.0}, clock)
    while clock.now * 1000 - 1_000_000_000 < 130 * DAY_MS:        # a refresh every 7 hours for 130 days
        clock.now += 7 * 3600
        funding.current()
    seen = [r for r in rows if r[1] <= clock.now * 1000]
    assert funding.current() == sum((r[2] for r in seen), D(0)) and len(seen) > 350
    assert all(now - since <= 31 * HOUR_MS for _, since, now in venue.calls[1:])   # never more than 24 h + 7 h back


def test_a_gap_longer_than_binance_s_history_is_unknown_to_the_base_and_to_the_loss_stop(tmp_path):
    """Audit of PR #147, finding 1: rows kept only up to more than 88 days ago can never be completed, so their total
    is partial; it must read as unknown (the loss stop blocks every entry and alerts), never as known."""
    clock = Clock(1_000_000.0)
    venue = Venue({"BTCUSDT": [("a", 1_000_100_000, D("-3"))]})
    funding, store = _funding(tmp_path, venue, {"BTCUSDT": 1_000_000.0}, clock)
    clock.now += 600
    assert funding.current() == D("-3")
    clock.now += 100 * 86_400                                       # the runtime was down for 100 days
    venue.rows["BTCUSDT"].append(("gap", int(clock.now * 1000) - 50 * DAY_MS, D("-299")))
    again, _ = _funding(tmp_path, venue, {"BTCUSDT": 1_000_000.0}, clock)   # the restart
    with pytest.raises(RuntimeError, match="funding"):
        again.current()
    with pytest.raises(RuntimeError, match="88 days"):
        again.last_known()                                          # not -3: the gap's -299 is unknowable
    with pytest.raises(RuntimeError, match="88 days"):
        funding.last_known()                                        # the process that ran on: the same
    assert store.coverage("BTCUSDT")[1] == 1_000_600_000            # no coverage across the gap


def test_a_restart_within_binance_s_history_reads_the_rest(tmp_path):
    clock = Clock(1_000_000.0)
    venue = Venue({"BTCUSDT": [("a", 1_000_100_000, D("-3"))]})
    funding, _ = _funding(tmp_path, venue, {"BTCUSDT": 1_000_000.0}, clock)
    clock.now += 600
    funding.current()
    clock.now += 30 * 86_400                                        # down for 30 days
    venue.rows["BTCUSDT"].append(("b", int(clock.now * 1000) - 10 * DAY_MS, D("-7")))
    again, _ = _funding(tmp_path, venue, {"BTCUSDT": 1_000_000.0}, clock)
    assert again.last_known() == D("-10") and again.current() == D("-10")


def test_after_a_restart_the_loss_stop_knows_the_stored_total_before_any_read(tmp_path):
    clock = Clock(1_000_000.0)
    venue = Venue({"BTCUSDT": [("a", 1_000_100_000, D("-3"))]})
    funding, _ = _funding(tmp_path, venue, {"BTCUSDT": 1_000_000.0}, clock)
    clock.now += 600
    funding.current()
    venue.down = True
    again, _ = _funding(tmp_path, venue, {"BTCUSDT": 1_000_000.0}, clock)   # a new process, the venue down
    assert again.last_known() == D("-3")
    with pytest.raises(RuntimeError, match="funding"):
        again.current()                                             # the base needs a fresh read


def test_a_symbol_whose_rows_do_not_reach_back_to_its_first_fill_is_unknown(tmp_path):
    clock = Clock(1_000_000.0)
    venue = Venue({})
    funding, store = _funding(tmp_path, venue, {"BTCUSDT": 1_000_000.0}, clock)
    store.add("BTCUSDT", [], from_ms=999_000_000_000, through_ms=1_000_000_000_000)   # covered from too late
    venue.down = True
    with pytest.raises(RuntimeError, match="funding"):
        funding.last_known()


def test_without_a_store_nothing_changes(tmp_path):
    from tests.unit.trading.test_strategy_capital import Venue as Old
    venue = Old({"BTCUSDT": D("-3.25"), "ETHUSDT": D("1.5")})
    funding = StrategyFunding(venue, ["BTCUSDT", "ETHUSDT"], {"BTCUSDT": 500.5}.get, Clock())
    assert funding.current() == D("-3.25") and venue.calls == [("BTCUSDT", 500_500, 1_000_000_000)]
    assert FUNDING_MAX_AGE == 7200


def test_kept_rows_that_start_after_the_first_fill_are_read_again_from_the_first_fill(tmp_path):
    """Mutation self-check survivor: a first fill earlier than the kept span (a person set the ledger's position)
    makes the next read start at the first fill, so the total covers it."""
    clock = Clock(1_000_000.0)
    venue = Venue({"BTCUSDT": [("early", 950_000_000, D("-5")), ("late", 999_000_000, D("1"))]})
    funding, store = _funding(tmp_path, venue, {"BTCUSDT": 900_000.0}, clock)
    store.add("BTCUSDT", [("late", 999_000_000, D("1"))], from_ms=990_000_000, through_ms=1_000_000_000)
    assert funding.current() == D("-4")
    assert venue.calls == [("BTCUSDT", 900_000_000, 1_000_000_000)]
    assert store.coverage("BTCUSDT") == (900_000_000, 1_000_000_000)


# -- the audit of PR #147 (2026-11-17) --------------------------------------------------------------------------
def test_the_same_transaction_id_on_two_symbols_is_kept_for_each(tmp_path):
    """Finding 8: rows are keyed by symbol too; nothing relies on Binance's ids being unique across symbols."""
    store = _store(tmp_path)
    store.add("BTCUSDT", [("7", 100, D("-1"))], from_ms=0, through_ms=200)
    store.add("ETHUSDT", [("7", 100, D("-2"))], from_ms=0, through_ms=200)
    assert store.total("BTCUSDT", since_ms=0) == D("-1") and store.total("ETHUSDT", since_ms=0) == D("-2")


def test_a_read_that_ends_before_the_kept_span_starts_is_a_gap(tmp_path):
    """Finding 6: [0, 10] before a span of [100, 200] must not mark [10, 100] as covered."""
    store = _store(tmp_path)
    store.add("BTCUSDT", [], from_ms=100, through_ms=200)
    with pytest.raises(ValueError, match="gap"):
        store.add("BTCUSDT", [], from_ms=0, through_ms=10)
    assert store.coverage("BTCUSDT") == (100, 200)


def test_a_migration_that_fails_halfway_changes_nothing(tmp_path):
    """Finding 4: the tables and the version change in one transaction."""
    path = tmp_path / schema.STATE_DB
    _version_1(path)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE funding_coverage (x INTEGER)")        # the second CREATE will fail
    with pytest.raises(sqlite3.OperationalError):
        schema.migrate(path)
    with sqlite3.connect(path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "funding_rows" not in tables


def test_migrate_never_creates_a_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        schema.migrate(tmp_path / "missing.sqlite")
    assert not (tmp_path / "missing.sqlite").exists()
