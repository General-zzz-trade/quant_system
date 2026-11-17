"""The feature-pool study's machinery (lab/feature_study.py; owner 2026-11-17, "执行"): the targets as defined before
any result, the year folds with the label window purged, the fixed models, the shuffled-target null and the skill
measures. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import feature_study as st

DAY = st.DAY
T0 = 1_609_459_200_000                     # 2021-01-01


def _closes(values, start=T0):
    return pd.Series(np.asarray(values, dtype=float), index=pd.Index(start + np.arange(1, len(values) + 1) * DAY))


def test_the_trend_target_needs_a_large_and_one_way_move():
    flat = np.full(60, 100.0)
    rising = np.concatenate([np.full(40, 100.0) * (1 + 0.01 * np.sin(np.arange(40))), 100 * 1.02 ** np.arange(1, 21)])
    choppy = np.concatenate([np.full(40, 100.0) * (1 + 0.01 * np.sin(np.arange(40))),
                             100 * (1 + 0.15 * np.where(np.arange(20) % 2, 1, -1))])
    for values, want in ((rising, 1.0), (choppy, 0.0)):
        y = st.targets(_closes(values), horizon=14)
        t = _closes(values).index[39]                                   # 14 days ahead are bars 40..53
        assert y.loc[t, "trend"] == want
    y = st.targets(_closes(flat), horizon=14)
    assert y["trend"].dropna().eq(0).all()
    assert y["trend"].iloc[-14:].isna().all() and y["fwd_vol"].iloc[-14:].isna().all()   # no future, no label


def test_the_volatility_and_direction_targets():
    values = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.02, 80)))
    c = _closes(values)
    y = st.targets(c, horizon=7)
    t = c.index[30]
    r = np.diff(np.log(values))
    assert y.loc[t, "fwd_vol"] == pytest.approx(np.log(np.std(r[30:37], ddof=1)))
    assert y.loc[t, "up"] == float(values[37] > values[30])
    assert y.loc[t, "label_end"] == t + 7 * DAY


def test_each_test_year_trains_only_on_rows_whose_labels_ended_before_it():
    index = pd.Index(T0 + np.arange(0, 3 * 365, dtype=np.int64) * DAY)
    label_end = pd.Series(index.to_numpy() + 14 * DAY, index=index)
    folds = st.year_folds(index, label_end, years=(2021, 2022, 2023))
    assert [f["year"] for f in folds] == [2022, 2023]
    for f in folds:
        start = st.year_start(f["year"])
        assert label_end.iloc[f["train"]].max() < start                      # purged (positions)
        assert index[f["test"]].min() >= start and index[f["test"]].max() < st.year_start(f["year"] + 1)
    assert len(folds[0]["train"]) == 365 - 14                                # 2021's last 14 rows are purged


def test_a_real_signal_beats_the_shuffled_null_and_noise_does_not():
    rng = np.random.default_rng(1)
    n = 3 * 365
    index = pd.Index(T0 + np.arange(n, dtype=np.int64) * DAY)
    x = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n)}, index=index)
    y_real = pd.Series((x["signal"] + rng.normal(0, 0.5, n) > 0).astype(float), index=index)
    y_noise = pd.Series((rng.normal(size=n) > 0).astype(float), index=index)
    label_end = pd.Series(index.to_numpy() + 14 * DAY, index=index)
    years = (2021, 2022, 2023)
    real = st.evaluate(x, y_real, label_end, years, kind="binary", model="logistic")
    noise = st.evaluate(x, y_noise, label_end, years, kind="binary", model="logistic")
    assert real["auc"] > 0.7 and abs(noise["auc"] - 0.5) < 0.1
    null = st.null_scores(x, y_real, label_end, years, kind="binary", model="logistic", runs=5, block=30, seed=0)
    assert len(null) == 5 and max(null) < real["auc"]


def test_the_volatility_skill_is_measured_against_the_naive_forecast():
    rng = np.random.default_rng(2)
    n = 3 * 365
    index = pd.Index(T0 + np.arange(n, dtype=np.int64) * DAY)
    naive = pd.Series(rng.normal(size=n), index=index)
    extra = pd.Series(rng.normal(size=n), index=index)
    y = naive + 0.8 * extra + rng.normal(0, 0.3, n)
    x = pd.DataFrame({"naive": naive, "extra": extra}, index=index)
    label_end = pd.Series(index.to_numpy() + 7 * DAY, index=index)
    out = st.evaluate(x, y, label_end, (2021, 2022, 2023), kind="regression", model="ridge", naive="naive")
    assert out["r2_gain"] > 0.2                                              # the extra feature adds skill
    alone = st.evaluate(x[["naive"]], y, label_end, (2021, 2022, 2023), kind="regression", model="ridge", naive="naive")
    assert abs(alone["r2_gain"]) < 0.02                                      # the naive forecast adds nothing to itself


def test_missing_features_are_filled_from_the_training_rows_only():
    index = pd.Index(T0 + np.arange(0, 2 * 365, dtype=np.int64) * DAY)
    x = pd.DataFrame({"a": np.r_[np.full(365, 1.0), np.full(365, np.nan)]}, index=index)
    filled = st.fill(x.iloc[:365], x.iloc[365:])
    assert (filled[1]["a"] == 1.0).all()                                     # the training median, not the test's
