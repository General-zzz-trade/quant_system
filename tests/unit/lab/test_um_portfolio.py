"""lab/um_portfolio.py: the benchmark of holding BTC with position control and the fold-by-fold portfolio of ideas
(tech lead 2026-11-17, the next lab round). Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import um_judge as judge
from lab import um_lab as lab
from lab import um_portfolio as portfolio
from tests.unit.lab.test_um_judge import PERIODS
from tests.unit.lab.test_um_judge_runs import market  # noqa: F401  (a fixture)

DAY = lab.DAY
FOLDS = (1, 2, 3)


def _days(k, mean, phase=0.0, noise=1e-4):
    days = np.arange(PERIODS[k][0] // DAY, PERIODS[k][1] // DAY)
    return pd.Series(mean + noise * np.sin(days + phase), index=days)


def _idea(name, means, phase=0.0):
    periods = {k: {"daily": _days(k, m, phase), "trips": [], "entries": [], "stops": 0, "fills": 0}
               for k, m in enumerate(means)}
    return [(judge.Config(name, {"p": 1}), periods)]


BENCH = {k: _days(k, 0.0005, phase=2.0, noise=0.01) for k in range(len(PERIODS))}      # Sharpe about 1.3
RUNS = {"steady": _idea("steady", (0.002, 0.002, 0.002, 0.002, 0.002)),
        "late": _idea("late", (-0.003, 0.004, 0.004, 0.004, 0.004), phase=1.0),
        "loser": _idea("loser", (-0.001,) * 5, phase=3.0),
        "unrunnable": _idea("unrunnable", (0.003,) * 5, phase=4.0)}


def _run(runs=RUNS, k=5, bench=(BENCH,)):
    return portfolio.walk_forward_portfolio(runs, PERIODS, FOLDS, bench, eligible=lambda n: n != "unrunnable", k=k)


def test_combine_holds_equal_capital_without_rebalancing():
    a, b = pd.Series([0.1, 0.0], index=[0, 1]), pd.Series([0.0, 0.1], index=[0, 1])
    out = portfolio.combine([a, b])
    assert out.iloc[0] == pytest.approx(0.05) and out.iloc[1] == pytest.approx(1.1 / 1.05 - 1)
    assert portfolio.combine([a]).equals(a.astype(float)) or np.allclose(portfolio.combine([a]), a)
    assert portfolio.combine([]).empty


def test_members_are_chosen_from_the_periods_before_each_fold():
    out = _run()
    # fold 1 sees period 0 only: "late" lost there; fold 2 sees 0-1: "late" made money and beats the benchmark;
    # the higher Sharpe on the earlier periods first
    assert out["members"] == [["steady"], ["steady", "late"], ["steady", "late"]]
    assert "unrunnable" not in out["chosen"] and "loser" in out["chosen"]
    fold2 = portfolio.combine([RUNS["steady"][0][1][2]["daily"], RUNS["late"][0][1][2]["daily"]])
    assert out["daily"].loc[PERIODS[2][0] // DAY:PERIODS[2][1] // DAY - 1].equals(fold2)
    assert list(out["daily"].index) == list(range(PERIODS[1][0] // DAY, PERIODS[3][1] // DAY))


def test_nothing_from_a_fold_or_after_changes_its_members():
    base = _run()["members"]
    for fold in FOLDS:
        flipped = {n: [(c, {k: ({**p, "daily": -p["daily"] * 50} if k >= fold else p) for k, p in periods.items()})
                       for c, periods in r] for n, r in RUNS.items()}
        assert _run(flipped)["members"][fold - 1] == base[fold - 1]


def test_at_most_k_members_the_best_prior_sharpe_first():
    assert _run(k=1)["members"] == [["steady"], ["steady"], ["steady"]]
    without = {n: r for n, r in RUNS.items() if n != "steady"}
    assert _run(without, k=1)["members"] == [[], ["late"], ["late"]]


def test_a_benchmark_that_is_too_good_leaves_the_fold_flat():
    rich = {k: _days(k, 0.01, phase=5.0) for k in range(len(PERIODS))}
    out = _run(bench=(BENCH, rich))
    assert out["members"] == [[], [], []] and (out["daily"] == 0.0).all()


def test_the_benchmark_sizes_btc_from_completed_days_and_never_above_0_95(market):  # noqa: F811
    w = portfolio.vol_target_weights(market)
    # 30 returns need 31 closes: the first size is known at the close of day 30 (the last bar of that day decides)
    assert w.max() <= 0.95 and w.min() >= 0.0 and (w.iloc[: 30 * 24 + 23] == 0.0).all() and w.iloc[30 * 24 + 23] > 0
    assert (w.iloc[40 * 24:] > 0).all()
    close = lab.frame("BTCUSDT", market)["close"]
    t = int(close.index[60 * 24])
    later = close.copy()
    later.loc[later.index > t] *= 5.0
    import trading.domain.strategy.ideas as ideas
    day = later.groupby(later.index // DAY).last()
    size = (0.02 / day.pct_change().rolling(30).std()).clip(upper=0.95)
    again = ideas._hours(lab.frame("BTCUSDT", market), size)
    assert again.loc[:t].equals(w.loc[:t])                                        # later prices change nothing before


def test_the_benchmark_is_replayed_per_period_with_the_run_s_capital(market):  # noqa: F811
    out = portfolio.vol_target_hold(market, PERIODS[:2], capital="300")
    assert set(out) == {0, 1}
    for k, (start, end) in enumerate(PERIODS[:2]):
        assert list(out[k].index) == list(range(start // DAY, end // DAY))
    assert lab.summary(out[1])["total_return"] != 0.0
