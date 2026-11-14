"""Mid-frequency feasibility machinery on synthetic data: causality, accounting, one-run discipline."""
from __future__ import annotations

import gzip
import json
import math

import numpy as np
import pandas as pd
import pytest

from lab import midfreq as mf
from lab import midfreq_study as study
from lab import trials

BAR = mf.BAR
DAY = mf.DAY
T0 = 1_664_582_400_000          # 2022-10-01 00:00 UTC
SYMBOLS = [f"S{k}USDT" for k in range(8)]


def write_snapshot(root, days, *, seed=3, reversal=0.0, wild_after=None):
    """Eight symbols of 5-minute bars. ``reversal`` plants an hourly idiosyncratic move that reverts next hour."""
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    bars = days * 288
    hours = bars // 12
    times = T0 + np.arange(bars) * BAR
    for k, symbol in enumerate(SYMBOLS):
        shock = rng.normal(0, 0.004, hours)
        hourly = shock - reversal * np.concatenate([[0.0], shock[:-1]])
        r = np.repeat(hourly / 12, 12) + rng.normal(0, 0.0004, bars)
        close = (20 + 5 * k) * np.exp(np.cumsum(r))
        if wild_after is not None:
            close = np.where(times + BAR > wild_after, close * 3, close)
        opened = np.concatenate([[close[0]], close[:-1]])
        quote = rng.uniform(900, 1100, bars)
        frame = pd.DataFrame({"open_time": times, "open": opened, "high": np.maximum(opened, close) * 1.0005,
                              "low": np.minimum(opened, close) * 0.9995, "close": close, "quote_volume": quote,
                              "taker_buy_quote_volume": quote * rng.uniform(0.4, 0.6, bars)})
        with gzip.open(root / f"{symbol}_5m.csv.gz", "wt") as stream:
            frame.to_csv(stream, index=False)
        funding = pd.DataFrame({"funding_time": T0 + np.arange(days * 3) * 8 * 3_600_000, "rate": 0.0001})
        with gzip.open(root / f"{symbol}_funding.csv.gz", "wt") as stream:
            funding.to_csv(stream, index=False)
    listed = {s: {"step": "0.001", "min_notional": "5"} for s in SYMBOLS[:6]}
    (root / "exchange_info.json").write_text(json.dumps(listed))
    (root / "manifest.json").write_text(json.dumps({"universe": [{"symbol": s} for s in SYMBOLS]}))
    return root


def test_the_panel_is_one_contiguous_grid_cut_at_the_end(tmp_path):
    root = write_snapshot(tmp_path / "s", 3)
    panel = mf.load_panel(root, end_ms=T0 + 2 * DAY)
    assert panel.symbols == SYMBOLS and len(panel.times) == 2 * 288 and panel.close.shape == (576, 8)
    assert (np.diff(panel.times) == BAR).all() and panel.times[-1] + BAR == T0 + 2 * DAY
    assert all(times.max() < T0 + 2 * DAY for times, _ in panel.funding)
    assert panel.rules[0] == {"step": 0.001, "min_notional": 5.0} and panel.rules[7] == mf.FALLBACK_RULES


def test_features_and_ridge_predictions_ignore_every_bar_after_the_decision(tmp_path):
    limit, cut = T0 + 40 * DAY, T0 + 30 * DAY
    out = {}
    for name, wild in (("calm", None), ("wild", cut)):
        panel = mf.load_panel(write_snapshot(tmp_path / name, 41, wild_after=wild), end_ms=limit)
        feats = mf.features(panel)
        rows = mf.decision_rows(panel, 12, start=T0 + 10 * DAY, end=limit)
        scores, blocks = mf.ridge_scores(panel, feats, rows, 12, first_test=T0 + 10 * DAY, block=7 * DAY,
                                         limit=limit, min_train=500)
        out[name] = (panel, feats, rows, scores, blocks)
    panel, feats, rows, scores, blocks = out["calm"]
    known = panel.times + BAR <= cut
    for name, values in feats.items():
        np.testing.assert_array_equal(values[known], out["wild"][1][name][known], err_msg=name)
    decided_before = panel.times[rows] + BAR <= cut
    assert np.isfinite(scores[decided_before]).sum() > 1000
    np.testing.assert_array_equal(scores[decided_before], out["wild"][3][decided_before])
    assert not np.allclose(scores[~decided_before], out["wild"][3][~decided_before], equal_nan=True)
    assert [b["test_start"] for b in blocks][:2] == [T0 + 10 * DAY, T0 + 17 * DAY]   # warm-up data trains block 1
    assert all(b["train_rows"] >= 500 for b in blocks) and np.isfinite(scores[0]).all()


def test_decisions_sit_on_the_horizon_clock_and_returns_run_fill_to_fill(tmp_path):
    panel = mf.load_panel(write_snapshot(tmp_path / "s", 3))
    rows = mf.decision_rows(panel, 12, start=T0 + DAY, end=T0 + 2 * DAY)
    assert ((panel.times[rows] + BAR) % (12 * BAR) == 0).all() and len(rows) == 24
    assert panel.times[rows[0]] + BAR == T0 + DAY and panel.times[rows[-1]] + BAR == T0 + 2 * DAY - 12 * BAR
    expected = panel.open[rows[0] + 13] / panel.open[rows[0] + 1] - 1
    np.testing.assert_allclose(mf.forward_return(panel, rows, 12)[0], expected)
    last = mf.decision_rows(panel, 12, start=T0, end=T0 + 4 * DAY)
    assert last[-1] + 1 + 12 < len(panel.times)                 # a decision without a full exit is not listed


def _panel(prices, *, funding=None, rules=None):
    """A six-symbol panel from a [bars, 6] array of open prices (closes equal opens)."""
    prices = np.asarray(prices, dtype=float)
    n = prices.shape[1]
    times = T0 + np.arange(len(prices)) * BAR
    empty = (np.array([], dtype=np.int64), np.array([]))
    return mf.Panel([f"S{k}" for k in range(n)], times, prices, prices, prices, prices, np.ones_like(prices),
                    np.ones_like(prices) / 2, funding or [empty] * n,
                    rules or [{"step": 0.0, "min_notional": 5.0}] * n)


def test_one_rebalance_by_hand_with_fees_slippage_and_funding():
    prices = np.array([[10.0] * 6, [10.0] * 6, [11, 10, 10, 10, 10, 9], [11, 10, 10, 10, 10, 9]])
    funding = [(np.array([T0 + 2 * BAR]), np.array([0.001]))] + [(np.array([], dtype=np.int64), np.array([]))] * 5
    scores = np.array([[6, 5, 4, 3, 2, 1.0], [np.nan] * 6])     # long S0-S2, short S3-S5; then no scores: flat
    run = mf.simulate(_panel(prices, funding=funding), np.array([0, 1]), scores, capital=600.0, fee=0.001,
                      slippage=0.01)
    buy, sell = 10 * 1.01, 10 * 0.99                             # six legs of 100 USDT at the first open (10)
    cash = 600 - 3 * (10 * buy) * 1.001 + 3 * (10 * sell) * (1 - 0.001)
    cash -= 10 * 11 * 0.001                                      # long S0 pays funding at the mark of bar 2 (11)
    # with no scores every leg is closed at the second fill: longs sell 1% low, shorts buy 1% high, plus the fee
    cash += (10 * 11 + 2 * 10 * 10) * 0.99 * (1 - 0.001) - (2 * 10 * 10 + 10 * 9) * 1.01 * 1.001
    assert run.equity.iloc[-1] == pytest.approx(cash) and run.equity.iloc[0] == 600.0
    assert (run.opened_legs, run.skipped_legs, run.wiped, run.average_legs) == (6, 0, False, 3.0)


def test_a_leg_below_the_minimum_order_is_skipped_and_lots_round_down():
    rules = [{"step": 1.0, "min_notional": 5.0}, {"step": 0.0, "min_notional": 500.0}] + \
        [{"step": 0.0, "min_notional": 5.0}] * 4
    prices = np.array([[30.0, 10, 10, 10, 10, 10]] * 4)
    run = mf.simulate(_panel(prices, rules=rules), np.array([0]), np.array([[6, 5, 4, 3, 2, 1.0]]), capital=600.0,
                      fee=0.0, slippage=0.0)
    assert (run.opened_legs, run.skipped_legs) == (5, 1)         # S1 needs 500 USDT; S0 buys 3 whole lots of 30
    assert mf._lots(100.0, 30.0, rules[0]) == 3.0 and mf._lots(100.0, 30.0, rules[1]) == 0.0
    assert mf._lots(4.0, 30.0, {"step": 0.0, "min_notional": 5.0}) == 0.0


def test_fewer_than_six_scored_symbols_means_no_position_and_a_symbol_that_stops_is_closed():
    prices = np.array([[10.0] * 6, [10.0] * 6, [12.0, 10, 10, 10, 10, 10], [np.nan, 10, 10, 10, 10, 10],
                       [np.nan, 10, 10, 10, 10, 10]])
    five = np.array([[np.nan, 5, 4, 3, 2, 1.0]])
    assert mf.simulate(_panel(prices), np.array([0]), five, capital=600.0, fee=0.0, slippage=0.0).opened_legs == 0
    scores = np.array([[6, 5, 4, 3, 2, 1.0], [6, 5, 4, 3, 2, 1.0], [np.nan] * 6])
    run = mf.simulate(_panel(prices), np.array([0, 1, 2]), scores, capital=600.0, fee=0.0, slippage=0.0)
    assert run.equity.iloc[-1] == pytest.approx(600 + 10 * 2)    # S0 bought at 10, closed at its last price 12


def test_a_wipeout_stops_trading_and_is_reported():
    prices = np.array([[10.0] * 6, [10.0] * 6, [0.01, 0.01, 0.01, 100, 100, 100], [0.01, 0.01, 0.01, 100, 100, 100]])
    scores = np.array([[6, 5, 4, 3, 2, 1.0], [6, 5, 4, 3, 2, 1.0]])
    run = mf.simulate(_panel(prices), np.array([0, 1]), scores, capital=600.0, fee=0.0, slippage=0.0)
    assert run.wiped and run.equity.iloc[-1] == 0.0


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("midfreq")
    data = write_snapshot(root / "data", 60, reversal=0.8)
    registry = root / "trials.jsonl"
    window = {"start": T0 + 20 * DAY, "limit": T0 + 60 * DAY, "block": 10 * DAY}
    return root, data, registry, window, study.run_study(data, root / "out", registry=registry, **window)


def test_the_study_runs_once_records_seven_trials_and_never_opens_mainnet(result):
    root, data, registry, window, summary = result
    rows = trials.load(registry)
    assert [r["trial"] for r in rows] == [c["id"] for c in study.CANDIDATES] and len(rows) == 7
    assert {r["family"] for r in rows} == {"midfreq_usdm"} and all(r["count"] == 1 for r in rows)
    spec = json.loads((root / "out/final_spec.json").read_text())
    assert spec["passed"] is False and spec["mainnet"].startswith("refused")
    assert spec["conclusion"] == summary["conclusion"] in {"feasible", "maker_only", "not_feasible"}
    assert summary["gate"]["passed"] is (summary["conclusion"] == "feasible")
    with pytest.raises(ValueError, match="empty output directory"):
        study.run_study(data, root / "out", registry=registry, **window)
    with pytest.raises(ValueError, match="runs once"):
        study.run_study(data, root / "again", registry=registry, **window)


def test_a_planted_hourly_reversal_is_found_and_costs_are_ordered(result):
    _, _, _, _, summary = result
    runs = summary["candidates"]["xs_rev_1h"]["runs"]
    assert runs["zero"]["total_return"] > 0.5                    # the planted effect is large before costs
    assert runs["zero"]["total_return"] > runs["maker"]["total_return"] > runs["taker"]["total_return"] > \
        runs["taker_x2"]["total_return"]
    assert summary["candidates"]["xs_mom_4h"]["runs"]["zero"]["total_return"] < runs["zero"]["total_return"]
    assert set(runs) == {"taker", "taker_x2", "maker", "zero", "taker_10k"}
    assert summary["candidates"]["ridge_1h"]["blocks"] and summary["candidates"]["ridge_1h"]["mean_ic"] > 0.05


def _checks(**kw):
    side = {"total_return": 0.2, "max_drawdown": 0.1, "wiped": False}
    taker, x2 = {**side, **kw.get("taker", {})}, {**side, **kw.get("x2", {})}
    return study.gate_checks(taker, x2, {"probability": kw.get("probability", 0.99)})


@pytest.mark.parametrize("kw,failed", [
    ({}, set()),
    ({"taker": {"total_return": 0.0}}, {"taker_total_return"}),
    ({"probability": 0.949}, {"deflated_sharpe_probability"}),
    ({"probability": float("nan")}, {"deflated_sharpe_probability"}),
    ({"x2": {"total_return": -0.01}}, {"taker_x2_total_return"}),
    ({"taker": {"max_drawdown": 0.31}}, {"max_drawdown_daily"}),
    ({"x2": {"wiped": True}}, {"no_wipeout"}),
])
def test_the_gate(kw, failed):
    assert {name for name, check in _checks(**kw).items() if not check["ok"]} == failed


def test_the_three_conclusions():
    maker = {"total_return": 0.1, "max_drawdown": 0.2, "wiped": False}
    assert study.conclusion(_checks(), maker) == "feasible"
    failing = _checks(taker={"total_return": -0.3})
    assert study.conclusion(failing, maker) == "maker_only"
    for bad in ({"total_return": 0.0}, {"max_drawdown": 0.31}, {"wiped": True}, {"total_return": math.nan}):
        assert study.conclusion(failing, {**maker, **bad}) == "not_feasible"
