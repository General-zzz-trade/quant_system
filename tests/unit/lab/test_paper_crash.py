"""S10 decision 4: a crash never leaves the saved simulated exchange and the order journal at different instants.

A crash is simulated by copying the state database (SQLite's online backup: exactly what is committed) right after a
journal transaction commits, and starting a new paper process on the copy later. The paper state is written inside
the journal's own transactions, so every copy is one instant of the running process:

- after any commit, the restarted process finds the venue and the ledger in agreement (no ledger mismatch, no order
  at the venue that the journal does not know) and goes on trading without an error;
- after the commit that follows a bar's matching but precedes its decision, the restart decides that bar - it is not
  matched twice - and the run ends with exactly research's fills.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pytest

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.unit.binance.fake_venue import HOUR
from tests.unit.lab.test_target_runtime import LIVE, _advance, _world
from tests.unit.lab.test_target_runtime_resilience import _assert_matches_lab
from trading.adapters.simulated.exchange import SimulatedExchange
from trading.adapters.sqlite.paper_state import PaperOrderJournal
from trading.adapters.sqlite.schema import STATE_DB
from trading.domain.issues import IssueKind

INCONSISTENT = {IssueKind.LEDGER_MISMATCH, IssueKind.FOREIGN_HOLDING, IssueKind.UNDELIVERED_FILL}


@pytest.fixture(autouse=True)
def _dev_signing(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "RISK_SCOPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")


def _spec(execution="market", ttl=1):
    return StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution=execution,
                        ttl_bars=ttl)


def _paper(world, state_dir):
    return TargetRuntime(spec_path=world["spec_path"], state_dir=state_dir, mode="paper", market=world["adapter"],
                         clock=lambda: world["clock"]["now"])


def _backup(source, directory):
    directory.mkdir(parents=True)
    with sqlite3.connect(source) as src, sqlite3.connect(directory / STATE_DB) as dst:
        src.backup(dst)


class Commits:
    """Counts the paper journal's committed transactions and copies the database after the chosen ones."""

    def __init__(self, monkeypatch, state, root):
        self.n, self.copies, self.wanted = 0, {}, set()
        self.state, self.root = state, root
        original = PaperOrderJournal.connect

        @contextmanager
        def connect(journal):
            with original(journal) as db:
                yield db
            self.n += 1
            if self.n in self.wanted:
                self.copies[self.n] = self.root / f"crash-{self.n}"
                _backup(self.state / STATE_DB, self.copies[self.n])
        monkeypatch.setattr(PaperOrderJournal, "connect", connect)


def test_after_any_commit_a_restart_finds_the_venue_and_the_journal_in_agreement(tmp_path, monkeypatch):
    world = _world(tmp_path, _spec("maker", 4))
    state = tmp_path / "paper"
    commits = Commits(monkeypatch, state, tmp_path)
    runtime = _paper(world, state)
    runtime.warmup()
    commits.wanted = set(range(commits.n + 1, commits.n + 4000, 37))     # spread over the whole run
    end = world["warm"] + 120
    for i in range(world["warm"], end):
        _advance(world, runtime, i)
    traded = sum(len(s.sleeve.broker.fills) for s in runtime.sessions.values())
    assert traded > 0 and len(commits.copies) >= 20
    restarts = 0
    for n, copy in sorted(commits.copies.items())[::max(1, len(commits.copies) // 5)]:
        restarted = _paper(world, copy)
        restarted.warmup()                              # replays every bar since the copy's cursors
        _advance(world, restarted, end)
        for symbol in restarted.sessions:
            found = {issue.kind for issue in restarted.exits.last_issues[symbol]}
            assert not found & INCONSISTENT, (n, symbol, restarted.exits.last_issues[symbol])
        venue = restarted.account
        assert isinstance(venue, SimulatedExchange)
        for symbol, session in restarted.sessions.items():
            assert session.position_qty() == venue.broker(symbol).position.qty, (n, symbol)
        restarts += 1
    assert restarts >= 5


def test_a_crash_between_a_bar_s_matching_and_its_decision_decides_the_bar_once(tmp_path, monkeypatch):
    spec = _spec()
    world = _world(tmp_path, spec)
    state = tmp_path / "paper"
    commits = Commits(monkeypatch, state, tmp_path)
    runtime = _paper(world, state)
    runtime.warmup()
    crash_at = world["warm"] + 150
    crash_open, venue, after = world["rows"]["ETHUSDT"][crash_at]["start"], runtime.account, []
    match = venue.on_closed_bar

    def recording(symbol, bar, funding_rate):
        match(symbol, bar, funding_rate)
        if symbol == "ETHUSDT" and bar.open_time == crash_open:
            after.append(commits.n + 1)                # copy after the first commit that follows the matching
            commits.wanted.add(commits.n + 1)
    monkeypatch.setattr(venue, "on_closed_bar", recording)
    for i in range(world["warm"], crash_at + 1):
        _advance(world, runtime, i)
    (after,) = after
    assert after in commits.copies
    resume = crash_at + 3                                # down for two more bars
    world["clock"]["now"] = (world["rows"]["BTCUSDT"][resume - 1]["start"] + HOUR + 5_000) / 1000
    restarted = _paper(world, commits.copies[after])
    restarted.warmup()
    for i in range(resume, world["warm"] + LIVE):
        _advance(world, restarted, i)
    _assert_matches_lab(world, restarted, spec)


def test_the_restart_skips_only_bars_the_venue_has_matched(tmp_path):
    world = _world(tmp_path, _spec())
    state = tmp_path / "paper"
    runtime = _paper(world, state)
    runtime.warmup()
    for i in range(world["warm"], world["warm"] + 3):
        _advance(world, runtime, i)
    venue = runtime.account
    rows = world["rows"]["BTCUSDT"]
    from trading.domain.bars import bar_from_kline
    last, nxt = bar_from_kline(rows[world["warm"] + 2]), bar_from_kline(rows[world["warm"] + 3])
    assert venue.has_matched("BTCUSDT", last) and not venue.has_matched("BTCUSDT", nxt)
