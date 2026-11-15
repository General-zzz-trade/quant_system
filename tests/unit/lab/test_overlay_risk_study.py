"""Overlay risk-management study: the predeclared variants and gate are pinned, the reference is the frozen
overlay exactly, and only development data is used. Synthetic data only; never the real snapshot."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from lab import overlay_risk_study as rs
from lab.data import DAY, funding_series, load_symbol
from lab.overlay_study import SLEEVE_BALANCE, SPEC
from lab.simulate import simulate
from lab.strategy import compute_targets
from lab.study import SYMBOLS
from tests.unit.lab.test_pair_study import T0, write_snapshot

REFERENCE = {"vol": "simple", "portfolio": False, "brake": False}


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("overlay_risk")
    market, _ = write_snapshot(root / "data", 330)
    bars, funding, hourly = {}, {}, {}
    for s in SYMBOLS:
        bars[s], funding[s], hourly[s] = load_symbol(market, s)
    return root, market, bars, funding, hourly


def test_the_variants_parameters_and_gate_are_the_predeclared_ones():
    # Changing any of these after results exist invalidates the study.
    assert rs.VARIANTS == {
        "ewma_vol": {"vol": "ewma", "portfolio": False, "brake": False},
        "portfolio_vol": {"vol": "simple", "portfolio": True, "brake": False},
        "drawdown_brake": {"vol": "simple", "portfolio": False, "brake": True},
        "combined": {"vol": "ewma", "portfolio": True, "brake": True},
    }
    assert (rs.EWMA_HALFLIFE_DAYS, rs.CORR_LOOKBACK_DAYS, rs.BRAKE) == (10, 30, (0.05, 0.15, 0.25))
    assert rs.GATE == {"max_drawdown_at_most_ratio_of_overlay": 0.85, "sharpe_shortfall_vs_overlay_max": 0.05,
                       "cost_x2_total_return_min_exclusive": 0.0, "fills_per_symbol_year_max": 150,
                       "years_drawdown_not_worse_min": 4}
    assert rs.FAMILY == "directional_btc_eth"


def test_the_reference_targets_are_the_frozen_overlay_exactly(snapshot):
    _, _, bars, funding, hourly = snapshot
    mine = rs.hourly_targets(hourly, REFERENCE)
    for s in SYMBOLS:
        frozen = compute_targets(hourly[s], funding_series(funding[s]), SPEC)
        assert np.array_equal(mine[s]["ready"].to_numpy(), frozen["ready"].to_numpy())
        assert np.allclose(mine[s]["target"].to_numpy(), frozen["target"].to_numpy(), rtol=0, atol=1e-12)


def test_the_joint_loop_without_the_brake_is_the_research_simulation_exactly(snapshot):
    _, _, bars, funding, hourly = snapshot
    start, end = T0 + 200 * DAY, T0 + 320 * DAY
    targets = rs.hourly_targets(hourly, REFERENCE)
    joint = rs.simulate_portfolio(bars, targets, funding, brake=False, start_ms=start, end_ms=end)
    for result in joint:
        alone = simulate(result.symbol, bars[result.symbol], targets[result.symbol], funding[result.symbol], SPEC,
                         start_ms=start, end_ms=end, starting_balance=SLEEVE_BALANCE)
        pd.testing.assert_frame_equal(result.curve, alone.curve)
        assert result.fills == alone.fills


def test_the_brake_is_full_to_5_percent_linear_to_15_percent_then_a_quarter():
    assert rs.brake_multiplier(0.0) == rs.brake_multiplier(0.05) == 1.0
    assert math.isclose(rs.brake_multiplier(0.10), 0.625)
    assert rs.brake_multiplier(0.15) == rs.brake_multiplier(0.6) == 0.25
    with pytest.raises(ValueError):
        rs.brake_multiplier(float("nan"))


def test_the_correlation_cap_never_scales_up_and_is_causal(snapshot):
    _, _, _, _, hourly = snapshot
    frames = {s: rs.daily_frame(hourly[s], "simple") for s in SYMBOLS}
    mult = rs.portfolio_multiplier(frames).dropna()
    assert len(mult) and (mult <= 1.0 + 1e-12).all() and (mult > 0).all()
    cut = len(hourly["BTCUSDT"]) - 24 * 40
    changed = {s: hourly[s].copy() for s in SYMBOLS}
    for s in SYMBOLS:
        changed[s].loc[cut:, ["open", "high", "low", "close"]] *= 3.0          # a different future
    later = rs.portfolio_multiplier({s: rs.daily_frame(changed[s], "simple") for s in SYMBOLS})
    known = mult.index[mult.index <= int(hourly["BTCUSDT"]["available_at"].iloc[cut - 1])]
    assert np.allclose(mult.loc[known], later.loc[known], equal_nan=True)


def test_ewma_targets_are_causal(snapshot):
    _, _, _, _, hourly = snapshot
    base = rs.daily_frame(hourly["BTCUSDT"], "ewma")
    cut = len(hourly["BTCUSDT"]) - 24 * 30
    changed = hourly["BTCUSDT"].copy()
    changed.loc[cut:, ["open", "high", "low", "close"]] *= 0.5
    later = rs.daily_frame(changed, "ewma")
    known = base["available_at"] <= int(hourly["BTCUSDT"]["available_at"].iloc[cut - 1])
    assert np.allclose(base.loc[known, "weight"], later.loc[known, "weight"], equal_nan=True)


def _checks(**over):
    variant = {"max_drawdown": 0.12, "sharpe": 0.70}
    overlay = {"max_drawdown": 0.16, "sharpe": 0.72}
    x2 = {"total_return": 0.3}
    fills = {"BTCUSDT": 70.0, "ETHUSDT": 60.0}
    yv = {2021: 0.1, 2022: 0.1, 2023: 0.1, 2024: 0.1, 2025: 0.2}
    yo = {2021: 0.1, 2022: 0.1, 2023: 0.1, 2024: 0.1, 2025: 0.1}
    for key, value in over.items():
        where, field = key.split("__")
        target = {"v": variant, "o": overlay, "x": x2, "f": fills, "yv": yv}[where]
        target[int(field) if where == "yv" else field] = value
    return rs.gate_checks(variant, overlay, x2, fills, yv, yo)


def test_the_gate_needs_every_check():
    assert all(c["ok"] for c in _checks().values())
    assert not _checks(v__max_drawdown=0.137)["max_drawdown_vs_overlay"]["ok"]       # 0.137 > 0.85 * 0.16
    assert not _checks(v__sharpe=0.66)["sharpe_vs_overlay"]["ok"]
    assert not _checks(x__total_return=0.0)["cost_x2_total_return"]["ok"]
    assert not _checks(f__ETHUSDT=151.0)["fills_per_symbol_year"]["ok"]
    assert not _checks(yv__2024=0.2)["years_drawdown_not_worse"]["ok"]               # 3 of 5


def test_selection_takes_the_passing_variant_with_the_highest_calmar():
    results = {"a": {"strategy": {"calmar": 2.0}, "gate": {"passed": False}},
               "b": {"strategy": {"calmar": 1.0}, "gate": {"passed": True}},
               "c": {"strategy": {"calmar": 1.5}, "gate": {"passed": True}}}
    assert rs.select(results) == "c"
    assert rs.select({"a": results["a"]}) is None


def test_the_study_uses_development_data_only_and_records_four_trials(snapshot, tmp_path):
    root, market, *_ = snapshot
    registry = tmp_path / "trials.jsonl"
    eval_start, dev_end = T0 + 200 * DAY, T0 + 320 * DAY
    summary = rs.run_study(market, tmp_path / "out", eval_start=eval_start, dev_end=dev_end, registry=registry)
    rows = [json.loads(line) for line in registry.read_text().splitlines()]
    assert [r["trial"] for r in rows] == list(rs.VARIANTS) and {r["family"] for r in rows} == {rs.FAMILY}
    assert set(summary["variants"]) == set(rs.VARIANTS)
    # a different future after the window changes nothing
    wild, _ = write_snapshot(tmp_path / "wild", 330, wild_after=dev_end)
    again = rs.run_study(wild, tmp_path / "out2", eval_start=eval_start, dev_end=dev_end,
                         registry=tmp_path / "trials2.jsonl")
    assert again["reference_overlay"]["strategy"] == summary["reference_overlay"]["strategy"]
    assert {k: v["strategy"] for k, v in again["variants"].items()} == \
        {k: v["strategy"] for k, v in summary["variants"].items()}
    with pytest.raises(ValueError, match="empty output directory"):
        rs.run_study(market, tmp_path / "out", eval_start=eval_start, dev_end=dev_end, registry=registry)
