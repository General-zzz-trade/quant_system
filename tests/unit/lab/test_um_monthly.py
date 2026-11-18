"""The lab's monthly re-selection (lab/um_monthly.py; tech lead 2026-11-17: monthly, on a new snapshot).

- ``monthly_periods``: the holdout is the 6 months before the snapshot's end; before it, whole yearly folds as far back
  as leave at least a year of selection after the 30 days the list rule needs; period 0 is the rest from 2020-01-01.
  On the 2026-11-12 snapshot (end 2026-11-13) that is exactly the 2026-11-17 run's periods.
- ``plan`` writes windows.md once; ``screen`` and ``final`` refuse to run unless windows.md is committed and unchanged
  (the periods are registered before any result).
"""
from __future__ import annotations

import json
import subprocess
from datetime import date

import pytest

from lab import um_judge as judge
from lab import um_lab as lab
from lab import um_monthly as monthly
from tests.unit.lab.test_um_judge_runs import make, market  # noqa: F401  (a fixture)


def d(text):
    return lab.ms(text)


def test_the_2026_11_13_end_gives_the_2026_11_17_run_s_periods():
    folds = tuple((d(f"{y}-05-13"), d(f"{y + 1}-05-13")) for y in range(2021, 2026))
    assert monthly.monthly_periods(date(2026, 11, 13)) == judge.periods(d("2020-01-01"), folds, d("2026-11-13"))


def test_a_month_later_everything_moves_a_month():
    p = monthly.monthly_periods(date(2026, 12, 1))
    assert p[0] == (d("2020-01-31"), d("2021-06-01"))
    assert p[1:-1] == tuple((d(f"{y}-06-01"), d(f"{y + 1}-06-01")) for y in range(2021, 2026))
    assert p[-1] == (d("2026-06-01"), d("2026-12-01"))


def test_a_fold_is_added_only_when_a_year_of_selection_is_left_before_it():
    p = monthly.monthly_periods(date(2027, 7, 1))                 # 2021-01-01 would leave 11 months: not a fold
    assert p[0] == (d("2020-01-31"), d("2022-01-01")) and len(p) == 1 + 5 + 1
    q = monthly.monthly_periods(date(2027, 12, 1))                 # 2021-06-01 leaves 16 months: a sixth fold
    assert q[0] == (d("2020-01-31"), d("2021-06-01")) and len(q) == 1 + 6 + 1
    assert q[-1] == (d("2027-06-01"), d("2027-12-01"))


def test_the_end_comes_from_the_snapshot_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"window": ["2020-01-01", "2026-12-01"]}))
    assert monthly.snapshot_end(tmp_path) == date(2026, 12, 1)
    (tmp_path / "manifest.json").write_text(json.dumps({"window": ["2021-01-01", "2026-12-01"]}))
    with pytest.raises(ValueError, match="2020-01-01"):
        monthly.snapshot_end(tmp_path)


# -- registered before any result -----------------------------------------------------------------------------
def _repo(tmp_path):
    repo = tmp_path / "repo"
    run = repo / "research" / "lab" / "runs" / "2026-12-08"
    run.mkdir(parents=True)
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "manifest.json").write_text(json.dumps({"window": ["2020-01-01", "2026-12-01"]}))
    return repo, run, snap


def _commit(repo):
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "plan"], check=True)


def test_plan_writes_the_windows_once(tmp_path):
    repo, run, snap = _repo(tmp_path)
    monthly.plan(snap, run)
    text = (run / "windows.md").read_text()
    assert "2026-06-01" in text and "2026-12-01" in text and snap.name in text and "1092" in text
    with pytest.raises(FileExistsError):
        monthly.plan(snap, run)


def test_the_screen_refuses_windows_that_are_not_committed_or_changed(tmp_path):
    repo, run, snap = _repo(tmp_path)
    with pytest.raises(SystemExit, match="plan"):
        monthly.registered(snap, run)                              # no windows.md
    monthly.plan(snap, run)
    with pytest.raises(SystemExit, match="commit"):
        monthly.registered(snap, run)                              # written, not committed
    _commit(repo)
    monthly.registered(snap, run)                                  # committed: fine
    (run / "windows.md").write_text((run / "windows.md").read_text() + "\nedited\n")
    with pytest.raises(SystemExit, match="commit"):
        monthly.registered(snap, run)                              # changed after the commit


def test_the_screen_refuses_another_snapshot_than_the_plan_s(tmp_path):
    repo, run, snap = _repo(tmp_path)
    monthly.plan(snap, run)
    _commit(repo)
    other = tmp_path / "other"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"window": ["2020-01-01", "2026-12-01"]}))
    with pytest.raises(SystemExit, match="snapshot"):
        monthly.registered(other, run)


# -- the 2026-11-18 standards (owner, after the retrospective) ----------------------------------------------------
def test_the_line_applies_the_new_gates_after_the_screen(market, tmp_path):  # noqa: F811
    j = make(market, tmp_path)
    variants = (judge.LIVE_STOP,)
    screened = j.screen(variants=variants, processes=1)
    out = monthly.line(j, screened, variants=variants)
    assert set(out["benchmarks"]) == {"btc_hold", "btc_vol_target"}
    for name, row in out["ideas"].items():
        assert set(row["checks"]) == {"screen", "family_deflated_at_least_0.9", "regression_alpha", "portfolio_gain"}
        assert row["family_deflated"]["n_trials"] == len(screened["ideas"][name]["selection_sharpe"])
        assert row["survives"] == all(row["checks"].values())
        assert row["checks"]["screen"] == screened["ideas"][name]["survives"]
        assert 0.0 <= row["power"]["at_sharpe_0.5"] <= 1.0


def test_only_the_line_s_survivors_reach_the_final_check():
    screened = {"ideas": {"a": {"survives": True}, "b": {"survives": True}, "c": {"survives": False}}}
    line = {"ideas": {"a": {"survives": False}, "b": {"survives": True}, "c": {"survives": False}}}
    assert {n for n, r in monthly.with_line(screened, line)["ideas"].items() if r["survives"]} == {"b"}


def test_an_idea_survives_only_when_every_gate_passes():
    assert monthly.survives({"screen": True, "family_deflated_at_least_0.9": True, "regression_alpha": True,
                             "portfolio_gain": True})
    assert not monthly.survives({"screen": True, "family_deflated_at_least_0.9": False, "regression_alpha": True,
                                 "portfolio_gain": True})
