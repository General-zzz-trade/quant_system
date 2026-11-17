"""The whole-market panel study's machinery (lab/panel_study.py; owner 2026-11-17, "执行"): the liquid universe from
past volume only, one contract's features, the day's ranks and market state, the targets, the pooled walk-forward
and its daily rank correlation, and the two nulls. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import panel_study as ps
from lab.causality import violations
from tests.unit.lab.test_feature_pool import synthetic_inputs

H, DAY = ps.H, ps.DAY
T0 = 1_609_459_200_000


def _volumes(levels, days=120):
    index = pd.Index(T0 + np.arange(1, days + 1, dtype=np.int64) * DAY)
    return {s: pd.Series(np.full(days, float(v)), index=index) for s, v in levels.items()}


def test_the_universe_is_the_most_traded_of_the_month_before_with_btc_and_eth_always():
    vols = _volumes({"BTCUSDT": 1, "ETHUSDT": 2, "AUSDT": 50, "BUSDT": 40, "CUSDT": 30, "NEWUSDT": 99})
    first = {s: T0 for s in vols} | {"NEWUSDT": T0 + 80 * DAY}                    # listed late
    times = T0 + np.array([90, 119], dtype=np.int64) * DAY
    member = ps.liquid_universe(vols, first, times, top=2)
    assert set(member.columns[member.iloc[0]]) == {"AUSDT", "BUSDT", "BTCUSDT", "ETHUSDT"}
    assert not member["NEWUSDT"].iloc[0] and not member["NEWUSDT"].iloc[1]     # fewer than 60 days of bars
    rng = np.random.default_rng(7)                                                 # close, moving volumes
    close = {s: v * rng.uniform(0.9, 1.1, len(v)) + 30 for s, v in _volumes({k: 1 for k in vols}).items()}
    days = T0 + np.arange(70, 120, dtype=np.int64) * DAY
    cuts = [int(days[5]), int(days[30])]
    assert violations(lambda m: ps.liquid_universe(m, first, days, top=2), close, cuts) == []


@pytest.fixture(scope="module")
def one():
    m = synthetic_inputs()
    return m["hourly"]["BTCUSDT"], m["funding"]["BTCUSDT"]


TIMES = T0 + np.arange(310, 400, dtype=np.int64) * DAY


def test_one_contract_s_features_have_values_and_read_nothing_early(one):
    hourly, funding = one
    out = ps.symbol_features(hourly, funding, TIMES)
    assert list(out.columns) == list(ps.SYMBOL_FEATURES)
    assert [c for c in out.columns if not np.isfinite(out[c]).mean() > 0.9] == []
    ret7 = out["ret_7_z"].iloc[5]
    d = ps.fp.daily(hourly)["close"]
    t = int(TIMES[5])
    raw = np.log(d.loc[t] / d.loc[t - 7 * DAY])
    sigma = np.log(d[d.index <= t]).diff().iloc[-30:].std()
    assert ret7 == pytest.approx(raw / (sigma * np.sqrt(7)))
    cuts = [int(TIMES[10]) + 7 * H, int(TIMES[60])]
    assert violations(lambda m: ps.symbol_features(m["h"], m["f"], TIMES), {"h": hourly, "f": funding}, cuts) == []


def test_the_day_s_ranks_and_the_market_state_are_joined_by_time():
    t = np.repeat(T0 + np.array([1, 2], dtype=np.int64) * DAY, 3)
    panel = pd.DataFrame({"symbol": list("abcabc"), **{n: [1.0, 2.0, 3.0, 3.0, 2.0, 1.0] for n in ps.RANKED}},
                         index=pd.Index(t))
    market = pd.DataFrame({n: [10.0, 20.0] for n in ps.MARKET}, index=pd.Index(np.unique(t)))
    out = ps.cross_section(panel, market)
    assert out["rank_ret_7_z"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0, 1.0, 2 / 3, 1 / 3])
    assert out["breadth_ma30"].tolist() == [10.0] * 3 + [20.0] * 3


def test_the_direction_target_is_the_next_fortnight_over_its_usual_spread():
    rng = np.random.default_rng(0)
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))
    closes = pd.Series(values, index=pd.Index(T0 + np.arange(1, 121, dtype=np.int64) * DAY))
    times = closes.index.to_numpy()
    y = ps.symbol_targets(closes, times)
    k = 50
    sigma = np.std(np.diff(np.log(values[k - 30:k + 1])), ddof=1)
    assert y["ret_14_scaled"].iloc[k] == pytest.approx(np.log(values[k + 14] / values[k]) / (sigma * np.sqrt(14)))
    assert y["end_14"].iloc[k] == times[k] + 14 * DAY and np.isnan(y["ret_14_scaled"].iloc[-1])


def _panel(signal_strength, seed=0, days=3 * 365, names=30):
    rng = np.random.default_rng(seed)
    t = np.repeat(T0 + np.arange(days, dtype=np.int64) * DAY, names)
    s = np.tile(np.array([f"S{k}" for k in range(names)]), days)
    f = rng.normal(size=len(t))
    y = signal_strength * f + rng.normal(size=len(t))
    return pd.DataFrame({"f": f, "noise": rng.normal(size=len(t))}), y, t, s


def test_a_cross_sectional_signal_shows_in_the_daily_rank_correlation_and_noise_does_not():
    years = (2021, 2022, 2023)
    x, y, t, _ = _panel(0.5)
    real = ps.evaluate(x, y, t, t + 14 * DAY, years, kind="regression", model="ridge")
    assert real["ic"]["mean_ic"] > 0.3 and real["ic"]["t_spaced"] > 3
    x, y, t, _ = _panel(0.0, seed=1)
    noise = ps.evaluate(x, y, t, t + 14 * DAY, years, kind="regression", model="ridge")
    assert abs(noise["ic"]["mean_ic"]) < 0.05


def test_the_two_nulls_keep_what_they_should():
    x, y, t, s = _panel(0.5, days=200, names=5)
    rng = np.random.default_rng(3)
    within = ps.shuffled(y, t, s, how="within_day", rng=rng)
    for day in np.unique(t)[:20]:
        assert sorted(within[t == day]) == sorted(y[t == day])                     # each day's values, moved
    assert not np.allclose(within, y)
    blocks = ps.shuffled(y, t, s, how="blocks", rng=rng, block=30)
    for name in np.unique(s):
        assert sorted(blocks[s == name]) == sorted(y[s == name])                   # each contract's own values
    real = ps.evaluate(x, y, t, t + 14 * DAY, (2021,), kind="regression", model="ridge")
    assert real["n_test"] == 0                                                     # one year: nothing to test


def test_a_day_s_rows_weigh_one_in_all():
    t = np.array([1, 1, 1, 2, 2, 3], dtype=np.int64)
    w = ps._weights(t)
    assert w.tolist() == pytest.approx([1 / 3] * 3 + [0.5] * 2 + [1.0])


def test_a_frozen_model_predicts_after_export_exactly_as_before(tmp_path):
    x, y, t, _ = _panel(0.5, days=200, names=10)
    x.loc[x.index[:50], "noise"] = np.nan                                         # filled from the training medians
    binary = (y > 0).astype(float)
    for kind, model, target in (("binary", "logistic", binary), ("regression", "ridge", y),
                                ("binary", "lgbm", binary), ("regression", "lgbm", y)):
        frozen = ps.fit_final(x, target, t, kind=kind, model=model)
        path = ps.save_frozen(frozen, tmp_path / f"{kind}_{model}")
        again = ps.load_frozen(path)
        assert np.allclose(ps.predict_frozen(again, x), ps.predict_frozen(frozen, x), rtol=0, atol=1e-6)
        assert path.suffix in (".json", ".txt")                                    # never a pickle
