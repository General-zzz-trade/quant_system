"""ETH/BTC relative-value study on synthetic data: accounting, causality, one-run discipline."""
from __future__ import annotations

import csv
import json
import math
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from lab import pair_study as ps
from lab.pair import build, load_inputs

HOUR = 3_600_000
DAY = 24 * HOUR
T0 = 1_609_459_200_000          # 2021-01-01


def frame(times, btc, eth):
    return pd.DataFrame({"open_time": times, "open_b": btc[0], "high_b": btc[1], "low_b": btc[2], "close_b": btc[3],
                         "open_e": eth[0], "high_e": eth[1], "low_e": eth[2], "close_e": eth[3]})


def test_pair_accounting_by_hand():
    start = T0 + DAY
    times = [start - HOUR, start, start + HOUR]
    btc = ([100.0] * 3, [100.0, 100.0, 100.0], [100.0, 90.0, 100.0], [100.0, 100.0, 90.0])
    eth = ([10.0] * 3, [10.0, 10.5, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.5])
    funding = {"BTCUSDT": {start + HOUR: (D("0.001"), D("90"))}, "ETHUSDT": {}}
    run = ps.simulate_pair(frame(times, btc, eth), funding, {start: 1}, start=start, end=start + 2 * HOUR,
                           capital=200.0)
    # BTC leg: floor(200/100/0.001)*0.001 = 2 BTC short at 100*(1-0.0003); ETH leg 20 long at 10*(1+0.0003)
    short_px, long_px = 100 * (1 - 3e-4), 10 * (1 + 3e-4)
    fees = 2 * short_px * 5e-4 + 20 * long_px * 5e-4
    funding_paid = 2 * 90 * 0.001                       # the short BTC leg receives positive funding
    final = 200 - fees + funding_paid + (-2) * (90 - short_px) + 20 * (10.5 - long_px)
    assert run.curve["equity"].iloc[-1] == pytest.approx(final)
    assert run.positions == 1 and run.round_trips == 0 and run.fees == pytest.approx(fees)
    # short BTC: worst high 100 vs entry; long ETH: worst low 10 vs entry
    assert run.worst_leg_move == pytest.approx(max((100 - short_px) / short_px, (long_px - 10) / long_px))
    assert run.isolated_breaches == 0


def test_isolated_breach_is_reported():
    start = T0 + DAY
    times = [start - HOUR, start, start + HOUR]
    btc = ([100.0] * 3, [100.0, 100.0, 111.0], [100.0] * 3, [100.0, 100.0, 110.0])
    eth = ([10.0] * 3, [10.0] * 3, [10.0] * 3, [10.0] * 3)
    run = ps.simulate_pair(frame(times, btc, eth), {"BTCUSDT": {}, "ETHUSDT": {}}, {start: 1}, start=start,
                           end=start + 2 * HOUR, capital=200.0)
    assert run.isolated_breaches == 1 and run.worst_leg_move > 0.095


def test_decision_calendar():
    s = pd.Series([1.0, -1.0, np.nan, 1.0, -1.0, 1.0, 1.0, -1.0], index=[T0 + i * DAY for i in range(8)])
    assert ps.decisions(s, 3, start=T0 + DAY, end=T0 + 8 * DAY) == {T0 + DAY: -1, T0 + 4 * DAY: -1, T0 + 7 * DAY: -1}
    assert ps.decisions(s, 3, start=T0 + 2 * DAY, end=T0 + 8 * DAY) == {T0 + 5 * DAY: 1}   # NaN at day 2 skipped


def test_ml_block_ignores_labels_that_reach_into_it():
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.Index([T0 + i * DAY for i in range(n)])
    cols = list(ps.P_FEATURES) + list(ps.D_FEATURES)
    feats = pd.DataFrame(rng.normal(size=(n, len(cols))), index=idx, columns=cols)
    feats["sigma_pair"] = 0.02
    closes = pd.DataFrame({"BTCUSDT": 100 * np.exp(np.cumsum(rng.normal(0, .02, n))),
                           "ETHUSDT": 10 * np.exp(np.cumsum(rng.normal(0, .03, n)))}, index=idx)
    cand = {"kind": "ml", "model": "ridge", "features": "PD", "horizon": 3}
    oos = T0 + 300 * DAY
    a, blocks = ps.ml_predictions(feats, closes, cand, oos_start=oos, limit=T0 + 395 * DAY)
    changed = closes.copy()
    changed.loc[changed.index >= oos] *= 5.0                   # touches only labels ending at/after the block start
    b, _ = ps.ml_predictions(feats, changed, cand, oos_start=oos, limit=T0 + 395 * DAY)
    assert blocks and a.notna().sum() > 50
    first_block = a.index < oos + ps.BLOCK               # later blocks may legitimately train on those labels
    pd.testing.assert_series_equal(a[first_block], b[first_block])
    assert not a[~first_block].equals(b[~first_block])


# -- synthetic snapshot ------------------------------------------------------------------------------
def write_snapshot(root, days, seed=3, wild_after=None):
    market, premium = root / "market", root / "premium"
    market.mkdir(parents=True)
    premium.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    n = days * 24
    common = np.cumsum(rng.normal(0, 0.006, n))
    for symbol, px, extra in (("BTCUSDT", 30000.0, 0.0), ("ETHUSDT", 2000.0, 0.004)):
        closes = px * np.exp(common + np.cumsum(rng.normal(0, extra or 1e-9, n)))
        if wild_after is not None:
            closes = np.where(T0 + np.arange(n) * HOUR >= wild_after, closes * 4, closes)
        with (market / f"{symbol}_1h.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["open_time", "open", "high", "low", "close", "volume"])
            prev = closes[0]
            for i, c in enumerate(closes):
                w.writerow([T0 + i * HOUR, f"{prev:.4f}", f"{max(prev, c) * 1.002:.4f}", f"{min(prev, c) * .998:.4f}",
                            f"{c:.4f}", "1000"])
                prev = c
        with (market / f"{symbol}_funding.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["funding_time", "rate", "mark_price"])
            for i in range(0, n, 8):
                w.writerow([T0 + i * HOUR, f"{rng.normal(1e-4, 5e-5):.8f}", f"{closes[i]:.4f}"])
        with (premium / f"{symbol}_premium_1h.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["open_time", "open", "high", "low", "close"])
            for i in range(n):
                v = f"{rng.normal(0, 3e-4):.8f}"
                w.writerow([T0 + i * HOUR, v, v, v, v])
        with (market / f"{symbol}_metrics.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
                        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"])
            oi = 1e5
            for i in range(days * 288):
                oi *= math.exp(rng.normal(0, 0.002))
                w.writerow([T0 + i * 300_000, f"{oi:.2f}", "0", "1.1", f"{math.exp(rng.normal(0, .1)):.4f}",
                            f"{math.exp(rng.normal(0, .1)):.4f}", f"{math.exp(rng.normal(0, .2)):.4f}"])
    return market, premium


def test_features_do_not_change_when_later_data_changes(tmp_path):
    limit = T0 + 140 * DAY
    a = build(load_inputs(*write_snapshot(tmp_path / "a", 150), limit))[1]
    b = build(load_inputs(*write_snapshot(tmp_path / "b", 150, wild_after=T0 + 120 * DAY), limit))[1]
    cut = a.index <= T0 + 120 * DAY
    pd.testing.assert_frame_equal(a[cut], b[cut])
    assert a.loc[a.index > T0 + 125 * DAY, "mom_1"].ne(b.loc[b.index > T0 + 125 * DAY, "mom_1"]).any()
    assert a[list(ps.D_FEATURES)].iloc[-1].notna().all()


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("pair")
    market, premium = write_snapshot(root / "data", 420)
    registry = root / "trials.jsonl"
    window = {"oos_start": T0 + 300 * DAY, "limit": T0 + 410 * DAY}
    summary = ps.run_study(market, premium, root / "out", registry=registry, **window)
    return root, market, premium, registry, window, summary


def test_study_runs_once_and_records_every_trial(study):
    root, market, premium, registry, window, summary = study
    rows = [json.loads(line) for line in registry.read_text().splitlines()]
    assert len(rows) == len(ps.CANDIDATES) == 10 and {r["family"] for r in rows} == {ps.FAMILY}
    assert summary["deflated_sharpe"]["n_trials"] == 10
    frozen = json.loads((root / "out" / "final_spec.json").read_text())
    assert frozen["passed"] is False and frozen["trials"] == 10
    assert set(summary["gate"]["checks"]) == {"deflated_sharpe_probability", "sharpe_vs_best_static_pair",
                                              "cost_x2_total_return", "max_drawdown_hourly",
                                              "round_trips_per_year", "no_wipeout"}
    with pytest.raises(ValueError, match="runs once"):
        ps.run_study(market, premium, root / "out2", registry=registry, **window)


def test_ml_candidates_trade_out_of_sample(study):
    summary = study[-1]
    ml = [c for cid, c in summary["candidates"].items() if not cid.startswith("rule")]
    assert all(c["decisions"] > 5 for c in ml)
    assert all(b["train_rows"] >= ps.MIN_TRAIN_ROWS for c in ml for b in c["blocks"])


def test_wiped_curve_reports_without_dividing_by_zero():
    days = [(T0 // DAY) + i for i in (-1, 0, 400, 800)]
    curve = pd.Series([167.0, 100.0, 0.0, 0.0], index=days)
    out = ps.yearly(curve)
    assert out[2021] == pytest.approx(100 / 167 - 1) and out[2022] == -1.0 and math.isnan(out[2023])
