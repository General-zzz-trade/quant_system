"""The volatility-expansion breakout study's machinery (lab/vol_breakout.py; registered in
research/results/vol_breakout_2026-11-17/README.md before any code, owner 2026-11-17 "跑回测验证"): the channel
breakout and its exits, entries only on the universe's days, the slots, the panel filter and the sizes, the weights
on the hours, and the measures. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import panel_study as ps
from lab import vol_breakout as vb
from trading.domain.strategy.ideas import ON

H, DAY = vb.H, vb.DAY
T0 = 1_640_995_200_000                                   # 2022-01-01


def _daily(close, high=None, low=None):
    close = np.asarray(close, float)
    index = pd.Index(T0 + np.arange(1, len(close) + 1, dtype=np.int64) * DAY)
    high = close + 0.5 if high is None else np.asarray(high, float)
    low = close - 0.5 if low is None else np.asarray(low, float)
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=index)


def test_a_close_above_the_20_day_high_goes_long_and_leaves_below_the_10_day_low():
    close = [100.0] * 25 + [103.0, 104.0, 104.5, 104.0, 99.0, 99.0]
    d = _daily(close)
    trades = vb.breakout_trades("X", d)
    assert len(trades) == 1
    t = trades[0]
    assert (t.side, t.entry, t.exit, t.reason) == (1, int(d.index[25]), int(d.index[29]), "channel")


def test_a_close_below_the_20_day_low_goes_short_and_leaves_above_the_10_day_high():
    close = [100.0] * 25 + [97.0, 96.0, 96.0, 101.0, 101.0]
    d = _daily(close)
    (t,) = vb.breakout_trades("X", d)
    assert (t.side, t.entry, t.exit, t.reason) == (-1, int(d.index[25]), int(d.index[28]), "channel")


def test_a_trade_ends_after_14_days_and_the_next_one_starts_no_earlier_than_the_next_day():
    close = [100.0] * 25 + [101.0 + k for k in range(30)]               # a new high every day
    d = _daily(close)
    trades = vb.breakout_trades("X", d)
    first, second = trades[0], trades[1]
    assert first.exit - first.entry == 14 * DAY and first.reason == "time"
    assert second.entry == first.exit + DAY and second.side == 1


def test_entries_wait_for_a_day_the_contract_is_in_the_universe():
    close = [100.0] * 25 + [101.0 + k for k in range(5)]
    d = _daily(close)
    allowed = pd.Series(True, index=d.index)
    allowed.iloc[25:27] = False
    (t,) = vb.breakout_trades("X", d, allowed=allowed)
    assert t.entry == int(d.index[27])


def test_a_trade_still_open_at_the_data_end_ends_there_and_says_so():
    close = [100.0] * 25 + [101.0, 102.0, 103.0]
    d = _daily(close)
    (t,) = vb.breakout_trades("X", d)
    assert t.exit == int(d.index[-1]) and t.reason == "data_end"


def test_a_missing_day_leaves_no_signal_until_the_channel_is_whole_again():
    close = [100.0] * 25 + [103.0]
    d = _daily(close).drop(index=T0 + 10 * DAY)                           # day 10 never completed
    assert vb.breakout_trades("X", d) == []


def test_positions_follow_each_trade_from_its_entry_to_its_exit():
    close = [100.0] * 25 + [103.0, 104.0, 104.5, 104.0, 99.0, 99.0]
    d = _daily(close)
    pos = vb.positions(vb.breakout_trades("X", d), d.index.to_numpy())
    assert pos.iloc[:25].eq(0).all() and pos.iloc[25:29].eq(1).all() and pos.iloc[29:].eq(0).all()


def _trade(symbol, side, entry_day, exit_day):
    return vb.Trade(symbol, side, T0 + entry_day * DAY, T0 + exit_day * DAY, "channel")


def test_the_slots_go_to_the_most_traded_first_and_free_at_the_exit():
    trades = [_trade("A", 1, 1, 5), _trade("B", 1, 1, 9), _trade("C", -1, 1, 9), _trade("D", 1, 5, 9),
              _trade("E", 1, 6, 9)]
    volume = {("A", T0 + DAY): 3.0, ("B", T0 + DAY): 9.0, ("C", T0 + DAY): 1.0, ("D", T0 + 5 * DAY): 1.0,
              ("E", T0 + 6 * DAY): 1.0}
    taken = vb.allocate(trades, lambda t: volume[(t.symbol, t.entry)], slots=2)
    assert [t.symbol for t in taken] == ["B", "A", "D"]                    # C: no slot; D: A's slot freed at day 5


def test_the_filter_keeps_the_top_fifth_of_the_day_s_predictions():
    t = np.repeat(np.array([T0 + DAY, T0 + 2 * DAY]), 10)
    sym = np.tile(np.array([f"S{k}" for k in range(10)]), 2)
    pred = np.r_[np.arange(10.0), np.arange(10.0)[::-1]]
    passed = vb.top_share(pd.DataFrame({"time": t, "symbol": sym, "pred": pred}), share=0.2)
    assert sorted(passed[T0 + DAY]) == ["S8", "S9"] and sorted(passed[T0 + 2 * DAY]) == ["S0", "S1"]


def test_sizes_are_equal_or_shrink_the_more_volatile_never_grow():
    vol = pd.DataFrame({"time": [T0] * 3, "symbol": ["A", "B", "C"], "pred": [np.log(0.02), np.log(0.04),
                                                                                np.log(0.08)]})
    sizes = vb.vol_sizes(vol)
    assert sizes[("A", T0)] == pytest.approx(ON)                          # calmer than the median: not enlarged
    assert sizes[("B", T0)] == pytest.approx(ON)                          # the median
    assert sizes[("C", T0)] == pytest.approx(ON * 0.5)


def test_weights_hold_from_the_bar_decided_at_the_entry_to_the_bar_before_the_exit_decision():
    index = pd.Index(T0 + np.arange(0, 6 * 24, dtype=np.int64) * H)
    trade = vb.Trade("A", -1, T0 + 2 * DAY, T0 + 4 * DAY, "channel")
    w = vb.weights([trade], {trade: 0.5}, index)
    on = w[w != 0]
    assert on.index[0] == T0 + 2 * DAY - H                                # closes at the entry decision
    assert on.index[-1] == T0 + 4 * DAY - 2 * H                           # the bar closing at the exit decision is 0
    assert set(on) == {-0.5}


def test_a_trade_s_return_is_its_curve_s_change_over_one_slot():
    at = pd.Index(T0 + np.arange(0, 10 * 24, dtype=np.int64) * H)
    curve = pd.Series(150.0, index=at)
    curve[curve.index >= T0 + 3 * DAY] = 165.0
    curve[curve.index >= T0 + 5 * DAY + H] = 140.0                       # the next trade, a day after the exit
    a, b = vb.Trade("A", 1, T0 + 2 * DAY, T0 + 4 * DAY, "channel"), vb.Trade("A", 1, T0 + 5 * DAY, T0 + 8 * DAY, "time")
    assert vb.trade_returns([a, b], curve, slot=150.0) == pytest.approx([0.1, -25 / 150])


def test_the_account_is_its_capital_plus_every_contract_s_money_made():
    days = T0 + np.arange(0, 4, dtype=np.int64) * DAY
    a = pd.Series([150.0, 165.0, 165.0, 165.0], index=days)
    b = pd.Series([150.0, 135.0], index=days[2:])                          # trades later
    daily = vb.account_daily({"A": a, "B": b}, capital=1500.0, slot=150.0, days=days)
    assert daily.tolist() == pytest.approx([15 / 1500, 0.0, -15 / 1515])


def test_the_filter_s_value_needs_a_spread_years_and_enough_trades():
    rng = np.random.default_rng(0)
    years = np.repeat([2022, 2023, 2024, 2025], 60)
    accepted = np.tile([True, False], 120)
    good = np.where(accepted, 0.02, 0.0) + rng.normal(0, 0.03, 240)
    verdict = vb.filter_value(good, accepted, years)
    assert verdict["value"] and verdict["checks"]["welch_t_at_least_2"]
    noise = rng.normal(0, 0.03, 240)
    assert not vb.filter_value(noise, accepted, years)["value"]
    few = vb.filter_value(good[:80], accepted[:80], years[:80])
    assert not few["checks"]["each_side_at_least_50"]
    two_years = np.where(accepted, np.where(years < 2024, 0.06, -0.005), 0.0) + rng.normal(0, 0.01, 240)
    verdict = vb.filter_value(two_years, accepted, years)
    assert verdict["checks"]["welch_t_at_least_2"] and not verdict["checks"]["years_at_least_3"]
    assert not verdict["value"]


def test_walk_forward_predictions_are_evaluate_s_on_labelled_rows_and_exist_for_the_rest():
    rng = np.random.default_rng(1)
    days, names = 3 * 365, 8
    t = np.repeat(1_609_459_200_000 + np.arange(days, dtype=np.int64) * DAY, names)
    x = pd.DataFrame({"f": rng.normal(size=len(t)), "g": rng.normal(size=len(t))})
    y = 0.5 * x["f"].to_numpy() + rng.normal(size=len(t))
    y[-14 * names:] = np.nan                                               # the last fortnight has no label yet
    end = t + 14 * DAY
    years = (2021, 2022, 2023)
    ev = ps.evaluate(x, y, t, end, years, kind="regression", model="ridge")
    pred = vb.walk_forward_predictions(x, y, t, end, years, kind="regression", model="ridge")
    keep = np.isfinite(y)
    assert np.allclose(pred[keep], ev["predictions"], equal_nan=True)
    assert np.isfinite(pred[~keep]).all()
    assert np.isnan(pred[t < 1_640_995_200_000]).all()                     # 2021 is training only


def test_the_priority_is_the_volume_the_universe_ranks_by():
    rng = np.random.default_rng(2)
    index = pd.Index(T0 + np.arange(1, 121, dtype=np.int64) * DAY)
    vols = {s: pd.Series(rng.uniform(1, 100, 120), index=index) for s in ("BTCUSDT", "ETHUSDT", "A", "B", "C", "D")}
    first = {s: T0 for s in vols}
    times = index.to_numpy()[70:]
    medians = vb.median_volumes(vols, first, times)
    member = ps.liquid_universe(vols, first, times, top=3)
    rank = medians.rank(axis=1, ascending=False, method="first")
    assert ((rank <= 3) & medians.notna()).drop(columns=list(ps.ALWAYS)).equals(member.drop(columns=list(ps.ALWAYS)))
