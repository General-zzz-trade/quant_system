"""The judge with maker entries (the cost lever, owner 2026-11-17): a run's ``execution`` and ``ttl_bars`` reach every
replay, its cache and its registry rows; a run without them replays, caches and registers exactly as before.
Synthetic data only."""
from __future__ import annotations

import pytest

from lab import um_judge as judge
from tests.unit.lab.test_um_judge_runs import FULL, _replays, _rows, make, market  # noqa: F401  (fixture)


def test_the_run_s_execution_reaches_every_replay(market, tmp_path, monkeypatch):  # noqa: F811
    calls = _replays(monkeypatch)
    make(market, tmp_path, execution="maker", ttl_bars=2).run_config((FULL, (1,)))
    assert [(c["execution"], c["ttl_bars"]) for c in calls] == [("maker", 2)] * 2      # UP and CRASH
    del calls[:]
    make(market, tmp_path).run_config((FULL, (1,)))
    assert len(calls) == 2 and not any("execution" in c or "ttl_bars" in c for c in calls)


def test_maker_entries_change_the_result(market, tmp_path):  # noqa: F811
    taker = make(market, tmp_path).run_config((FULL, (1,)))["periods"][1]
    maker = make(market, tmp_path, execution="maker").run_config((FULL, (1,)))["periods"][1]
    assert not taker["daily"].equals(maker["daily"])


def test_a_maker_run_s_rows_and_cache_say_so(market, tmp_path):  # noqa: F811
    maker = make(market, tmp_path, execution="maker", ttl_bars=3)
    maker.run_all([(FULL, (0,))], processes=1)
    row = _rows(maker)[0]
    assert (row["execution"], row["ttl_bars"]) == ("maker", 3) and row["terms"].endswith(", maker ttl 3")
    with pytest.raises(ValueError, match="other settings"):
        make(market, tmp_path).load(FULL)                       # a market run never reads a maker cache
    plain = make(market, tmp_path / "plain")
    plain.run_all([(FULL, (0,))], processes=1)
    row = _rows(plain)[0]
    assert "execution" not in row and "ttl_bars" not in row and "maker" not in row["terms"]


@pytest.mark.parametrize("bad", [{"execution": "limit"}, {"execution": "maker", "ttl_bars": 0}])
def test_an_unknown_execution_or_lifetime_is_refused(market, tmp_path, bad):  # noqa: F811
    with pytest.raises(ValueError, match="execution"):
        make(market, tmp_path, **bad)


def test_the_default_is_market():
    assert (judge.Judge.execution, judge.Judge.ttl_bars) == ("market", 1)
