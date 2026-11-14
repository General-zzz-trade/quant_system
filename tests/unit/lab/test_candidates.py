"""Signals of the forward-test candidates on synthetic data: frozen model, causality, schedules, events."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lab import candidates as c
from lab.pair import load_inputs
from lab.pair_study import fit_predict
from tests.unit.lab.test_pair_study import write_snapshot

HOUR = 3_600_000
DAY = 24 * HOUR
T0 = 1_609_459_200_000          # 2021-01-01


def test_the_frozen_ridge_predicts_exactly_like_the_study_fit():
    rng = np.random.default_rng(1)
    x, y, new = rng.normal(size=(300, 6)), rng.normal(size=300) * 3, rng.normal(size=(40, 6)) * 4
    x[:, 5] = 2.0                                           # a constant feature must not divide by zero
    frozen = c.fit_ridge(x.copy(), y)
    np.testing.assert_allclose(c.predict_frozen(frozen, new), fit_predict("ridge", x.copy(), y, new), rtol=1e-10)
    assert frozen["rows"] == 300 and frozen["alpha"] == 300.0


def test_a_missing_feature_gives_no_prediction():
    frozen = c.fit_ridge(np.random.default_rng(2).normal(size=(50, 3)), np.arange(50.0))
    out = c.predict_frozen(frozen, np.array([[0.0, 1.0, 2.0], [0.0, np.nan, 2.0]]))
    assert math.isfinite(out[0]) and math.isnan(out[1])


def test_sizing_targets_30_percent_volatility_with_a_cap():
    assert c.sized_weight(0.30 / math.sqrt(365)) == pytest.approx(0.9)          # 30% annual vol: capped at 0.9
    assert c.sized_weight(0.60 / math.sqrt(365)) == pytest.approx(0.5)
    assert all(math.isnan(c.sized_weight(v)) for v in (float("nan"), 0.0, -1.0))


def test_symbol_features_do_not_change_when_later_data_changes(tmp_path):
    limit = T0 + 140 * DAY
    a = c.symbol_features(load_inputs(*write_snapshot(tmp_path / "a", 150), limit), "ETHUSDT")
    b = c.symbol_features(load_inputs(*write_snapshot(tmp_path / "b", 150, wild_after=T0 + 120 * DAY), limit),
                          "ETHUSDT")
    cut = a.index <= T0 + 120 * DAY
    pd.testing.assert_frame_equal(a[cut], b[cut])
    assert a.loc[a.index > T0 + 125 * DAY, "mom_1"].ne(b.loc[b.index > T0 + 125 * DAY, "mom_1"]).any()
    assert list(a.columns) == [*c.B_FEATURES, "sigma"] and a[list(c.B_FEATURES)].iloc[-1].notna().all()


def test_directional_labels_look_exactly_seven_days_ahead():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 110.0, 99.0],
                      index=[T0 + i * DAY for i in range(9)])
    labels = c.directional_labels(close)
    assert labels["y"].iloc[0] == pytest.approx(0.10) and labels["y"].iloc[1] == pytest.approx(99 / 101 - 1)
    assert labels["y"].iloc[2:].isna().all()
    assert (labels["label_end"] - labels.index == 7 * DAY).all()


def _hourly(days, closes=None):
    n = days * 24
    closes = np.full(n, 100.0) if closes is None else np.asarray(closes, dtype=float)
    opens = T0 + np.arange(n) * HOUR
    return pd.DataFrame({"open_time": opens, "open": closes, "high": closes, "low": closes, "close": closes,
                         "volume": 1.0, "available_at": opens + HOUR})


def test_weekly_targets_decide_every_seven_days_and_keep_the_last_target_without_a_prediction():
    hourly, start = _hourly(30), T0 + 2 * DAY
    sigma = pd.Series(0.60 / math.sqrt(365), index=[T0 + i * DAY for i in range(31)])
    prediction = pd.Series({start: -0.4, start + DAY: 9.0, start + 7 * DAY: float("nan"), start + 14 * DAY: 0.2,
                            start + 21 * DAY: 0.1})
    sigma.loc[start + 21 * DAY] = float("nan")              # unknown volatility: keep the previous target
    targets = c.weekly_targets(hourly, prediction, sigma, start=start).set_index("available_at")
    assert not targets.loc[:start - HOUR, "ready"].any() and (targets.loc[:start - HOUR, "target"] == 0).all()
    assert targets.loc[start:, "ready"].all()
    np.testing.assert_allclose(targets.loc[start:start + 14 * DAY - HOUR, "target"], -0.5)   # day 1 is off the grid
    np.testing.assert_allclose(targets.loc[start + 14 * DAY:, "target"], 0.5)


def _metrics(days, oi):
    stamps = T0 + np.arange(days * 288) * 300_000
    return pd.DataFrame({"timestamp": stamps, "sum_open_interest": np.asarray(oi, dtype=float)})


def _flush_case(*, oi_falls: bool, second_flush_after_hours: int | None = None, days=200, crash_day=190,
                oi_rows_before_close: int = 18):
    rng = np.random.default_rng(5)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, days * 24)))
    oi = 1e5 * np.exp(np.cumsum(rng.normal(0, 0.0005, days * 288)))
    hits = [crash_day * 24] + ([crash_day * 24 + second_flush_after_hours] if second_flush_after_hours else [])
    for hour in hits:
        closes[hour:] *= 0.80                                # a 20% fall inside one hour
        if oi_falls:
            oi[(hour + 1) * 12 - oi_rows_before_close:] *= 0.70   # down 30%, from this many 5-min rows before the close
    return _hourly(days, closes), _metrics(days, oi), T0 + (crash_day * 24 + 1) * HOUR


def test_a_flush_needs_both_the_price_fall_and_the_open_interest_fall():
    hourly, metrics, crash_close = _flush_case(oi_falls=True)
    out = c.flush_targets(hourly, metrics).set_index("available_at")
    assert out.loc[crash_close, "event"] and out.loc[crash_close, "entry"]
    hourly, metrics, crash_close = _flush_case(oi_falls=False)
    out = c.flush_targets(hourly, metrics).set_index("available_at")
    assert not out.loc[crash_close:, "event"].any() and (out.loc[crash_close:, "target"] == 0).all()


def test_a_flush_position_lasts_72_hours_after_the_last_event_at_a_fixed_size():
    hourly, metrics, crash_close = _flush_case(oi_falls=True)
    out = c.flush_targets(hourly, metrics).set_index("available_at")
    last_event = out.index[out["event"]].max()
    held = out.loc[crash_close:last_event + 72 * HOUR - HOUR, "target"]
    assert held.nunique() == 1 and 0 < held.iloc[0] <= 0.9
    assert (out.loc[last_event + 72 * HOUR:, "target"] == 0).all()
    assert out["entry"].sum() == 1                          # later events in the same hold do not re-enter


def test_a_second_flush_during_the_hold_restarts_the_clock_but_not_the_size():
    hourly, metrics, crash_close = _flush_case(oi_falls=True, second_flush_after_hours=60)
    out = c.flush_targets(hourly, metrics).set_index("available_at")
    second = crash_close + 60 * HOUR
    assert out.loc[second, "event"] and not out.loc[second, "entry"]
    last_event = out.index[out["event"]].max()
    assert last_event >= second and out.loc[second + 71 * HOUR, "target"] == out.loc[crash_close, "target"]
    assert (out.loc[last_event + 72 * HOUR:, "target"] == 0).all()


def test_flush_thresholds_use_only_earlier_values_and_need_180_days():
    hourly, metrics, crash_close = _flush_case(oi_falls=True)
    signal = c.flush_signal(hourly, metrics).set_index("available_at")
    assert not signal["armed"].iloc[:180 * 24].any() and signal["armed"].iloc[181 * 24:].all()
    i = signal.index.get_loc(crash_close)
    for value, limit in (("r24", "q_r"), ("oi24", "q_oi")):   # the 5% quantile of the 8760 values before this one
        previous = signal[value].iloc[max(0, i - 365 * 24):i].dropna()
        assert signal[limit].iloc[i] == pytest.approx(np.quantile(previous, 0.05), rel=1e-12)
        assert signal[value].iloc[i] < signal[limit].iloc[i] < 0
    later = hourly.copy()
    later.loc[later["available_at"] > crash_close, "close"] *= 3.0
    changed = c.flush_signal(later, metrics).set_index("available_at")
    pd.testing.assert_frame_equal(signal.loc[:crash_close], changed.loc[:crash_close])


def test_an_open_interest_row_stamped_at_the_close_is_not_known_at_that_close():
    hourly, metrics, crash_close = _flush_case(oi_falls=True, oi_rows_before_close=0)   # the fall is stamped at t
    out = c.flush_signal(hourly, metrics).set_index("available_at")
    assert not out.loc[crash_close, "event"]                # usable only from t + 5 minutes
    assert out.loc[crash_close + HOUR, "event"]
    hourly, metrics, crash_close = _flush_case(oi_falls=True, oi_rows_before_close=1)   # stamped at t - 5 minutes
    assert c.flush_signal(hourly, metrics).set_index("available_at").loc[crash_close, "event"]
