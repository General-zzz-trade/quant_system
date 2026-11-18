"""The research gates of 2026-11-18 (lab/gates.py; owner, after the retrospective
docs/research/retrospective-2026-11-18.md): regression alpha with a Newey-West t, the risk-parity portfolio gain,
the family-deflated Sharpe at 0.9, the power of a design, and the sequential test of a forward look. Synthetic data."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lab import gates


def _days(n):
    return pd.RangeIndex(20_000, 20_000 + n)


def test_regression_alpha_recovers_a_known_intercept_and_beta():
    rng = np.random.default_rng(0)
    n = 3000
    bench = pd.Series(rng.normal(0.0004, 0.01, n), index=_days(n))
    noise = rng.normal(0.0, 0.004, n)
    strat = 0.0008 + 0.5 * bench + noise                            # 0.08 % a day of alpha, beta 0.5
    out = gates.regression_alpha(strat, pd.DataFrame({"hold": bench}))
    assert out["betas"]["hold"] == pytest.approx(0.5, abs=0.03)
    # the intercept's standard error: 0.004 / sqrt(3000) a day, about 0.027 a year; within three of them
    assert out["alpha_annual"] == pytest.approx(0.0008 * 365, abs=0.08)
    assert out["t_newey_west"] > 5 and out["passes"]
    beta_only = gates.regression_alpha(0.5 * bench + noise, pd.DataFrame({"hold": bench}))
    assert abs(beta_only["t_newey_west"]) < 2.5 and not beta_only["passes"] or beta_only["alpha_annual"] <= 0


def test_newey_west_widens_the_error_of_autocorrelated_residuals():
    rng = np.random.default_rng(1)
    n = 2000
    bench = pd.Series(rng.normal(0, 0.01, n), index=_days(n))
    e = np.zeros(n)
    shocks = rng.normal(0, 0.004, n)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + shocks[i]                           # persistent residuals
    strat = 0.0002 + 0.3 * bench + e
    nw = gates.regression_alpha(strat, pd.DataFrame({"hold": bench}))
    plain = gates.regression_alpha(strat, pd.DataFrame({"hold": bench}), lags=0)
    assert abs(nw["t_newey_west"]) < abs(plain["t_newey_west"])


def test_adding_an_uncorrelated_positive_strategy_at_risk_parity_raises_the_sharpe():
    rng = np.random.default_rng(2)
    n = 3000
    bench = pd.Series(rng.normal(0.0004, 0.01, n), index=_days(n))
    good = pd.Series(rng.normal(0.0004, 0.01, n), index=_days(n))
    out = gates.portfolio_gain(good, bench)
    assert out["sharpe_with"] > out["sharpe_benchmark"] and out["passes"]
    assert out["weights"]["strategy"] == pytest.approx(0.5, abs=0.05)          # equal risk
    bad = pd.Series(rng.normal(-0.0004, 0.01, n), index=_days(n))
    assert not gates.portfolio_gain(bad, bench)["passes"]
    wild = pd.Series(rng.normal(0.0008, 0.02, n), index=_days(n))              # twice the benchmark's spread
    assert gates.portfolio_gain(wild, bench)["weights"]["strategy"] == pytest.approx(1 / 3, abs=0.03)


def test_the_family_deflated_sharpe_counts_the_family_and_needs_0_9():
    rng = np.random.default_rng(3)
    daily = pd.Series(rng.normal(0.002, 0.01, 1500), index=_days(1500))     # a strong strategy
    few = gates.family_deflated(daily, [0.1, 0.3, -0.2, 0.5])
    many = gates.family_deflated(daily, list(rng.normal(0, 1.0, 500)))
    assert few["n_trials"] == 4 and few["probability"] > many["probability"]
    assert few["threshold"] == 0.9 and few["passes"] == (few["probability"] >= 0.9)
    weak = gates.family_deflated(pd.Series(rng.normal(0.0001, 0.01, 1500), index=_days(1500)), [0.1, 0.3, -0.2, 0.5])
    assert not weak["passes"]


def test_power_and_the_sample_a_design_needs():
    # a mean of 0.5 % per event with a spread of 3 % (effect size 1/6): about 285 events for 80 % power (two-sided 5 %)
    n = gates.samples_needed(0.005, 0.03)
    assert 270 <= n <= 300
    assert gates.power(0.005, 0.03, n) == pytest.approx(0.8, abs=0.02)
    assert gates.power(0.005, 0.03, 28) < 0.2                                 # 28 weekends: hopeless
    mde = gates.min_detectable(0.03, 28)
    assert mde == pytest.approx(0.03 * (1.959964 + 0.841621) / math.sqrt(28), rel=0.01)


def test_the_sequential_test_stops_early_on_a_clear_effect_and_continues_on_noise():
    rng = np.random.default_rng(4)
    clear = rng.normal(0.01, 0.02, 400)
    out = gates.sequential(clear, effect=0.01, sd=0.02)
    assert out["decision"] == "effect" and out["at"] < 100
    nothing = rng.normal(0.0, 0.02, 400)
    assert gates.sequential(nothing, effect=0.01, sd=0.02)["decision"] == "no effect"
    short = gates.sequential(rng.normal(0.003, 0.02, 5), effect=0.01, sd=0.02)
    assert short["decision"] == "continue" and short["at"] is None
