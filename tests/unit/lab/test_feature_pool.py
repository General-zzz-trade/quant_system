"""The BTC/ETH feature pool (lab/feature_pool.py; owner 2026-11-17, "执行"): every family has values, nothing is
read before it is known (lab/causality.py over every source), a few values by hand, a missing source gives missing
values, and the market breadth. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import feature_pool as fp
from lab.causality import violations

H, DAY = fp.H, fp.DAY
T0 = 1_609_459_200_000                    # 2021-01-01
N_DAYS = 420


def _hourly(rng, drift, start_price):
    n = N_DAYS * 24
    close = start_price * np.exp(np.cumsum(rng.normal(drift, 0.006, n)))
    open_ = np.concatenate([[start_price], close[:-1]])
    high, low = np.maximum(open_, close) * 1.002, np.minimum(open_, close) * 0.998
    volume = rng.uniform(100, 300, n)
    quote = volume * close
    t = T0 + np.arange(n, dtype=np.int64) * H
    return pd.DataFrame({"open_time": t, "open": open_, "high": high, "low": low, "close": close, "volume": volume,
                         "quote_volume": quote, "taker_buy_quote_volume": quote * rng.uniform(0.4, 0.6, n),
                         "available_at": t + H})


def _series(index, values):
    return pd.Series(np.asarray(values, dtype=float), index=pd.Index(np.asarray(index, dtype=np.int64)))


def synthetic_inputs(seed=0, *, full=True):
    rng = np.random.default_rng(seed)
    out = {k: {} for k in ("hourly", "funding", "metrics", "spot", "premium", "depth", "bvol", "dvol", "cm")}
    days = T0 + np.arange(N_DAYS, dtype=np.int64) * DAY
    for k, s in enumerate(fp.SYMBOLS):
        h = _hourly(rng, 0.0001 * (k + 1), 30000 / (k * 10 + 1))
        out["hourly"][s] = h
        settle = T0 + np.arange(N_DAYS * 3, dtype=np.int64) * 8 * H
        out["funding"][s] = _series(settle, rng.normal(0.0001, 0.0002, len(settle)))
        stamp = T0 + np.arange(N_DAYS * 288, dtype=np.int64) * 300_000
        out["metrics"][s] = pd.DataFrame({
            "timestamp": stamp, "sum_open_interest": 1e5 * np.exp(np.cumsum(rng.normal(0, 0.001, len(stamp)))),
            "sum_open_interest_value": 1e9, "count_toptrader_long_short_ratio": rng.uniform(0.8, 1.2, len(stamp)),
            "sum_toptrader_long_short_ratio": rng.uniform(0.8, 1.2, len(stamp)),
            "count_long_short_ratio": rng.uniform(0.8, 1.2, len(stamp)),
            "sum_taker_long_short_vol_ratio": rng.uniform(0.7, 1.3, len(stamp))})
        if not full:
            continue
        known = h["available_at"].to_numpy()
        out["spot"][s] = pd.DataFrame({"close": h["close"].to_numpy() * (1 - rng.normal(0.0003, 0.0002, len(h))),
                                       "quote_volume": h["quote_volume"].to_numpy() * 0.4,
                                       "taker_buy_quote_volume": h["quote_volume"].to_numpy() * 0.2},
                                      index=pd.Index(known))
        out["premium"][s] = _series(known, rng.normal(0.0001, 0.0003, len(h)))
        late = known >= T0 + 200 * DAY                                    # depth starts later, as Binance's does
        out["depth"][s] = pd.DataFrame({c: rng.uniform(1e6, 3e6, late.sum()) for c in ("bid_1", "ask_1",
                                                                                        "bid_5", "ask_5")},
                                       index=pd.Index(known[late]))
        later = known >= T0 + 300 * DAY
        out["bvol"][s] = _series(known[later], rng.uniform(40, 80, later.sum()))
        out["dvol"][s] = _series(known, 60 + np.cumsum(rng.normal(0, 0.1, len(h))))
        out["cm"][s] = pd.DataFrame({"FlowInExNtv": rng.uniform(1e4, 2e4, N_DAYS),
                                     "FlowOutExNtv": rng.uniform(1e4, 2e4, N_DAYS),
                                     "SplyExNtv": 2e6 * np.exp(np.cumsum(rng.normal(0, 0.001, N_DAYS))),
                                     "AdrActCnt": rng.uniform(5e5, 9e5, N_DAYS), "TxCnt": rng.uniform(2e5, 4e5, N_DAYS),
                                     "CapMrktCurUSD": 1e12}, index=pd.Index(days + fp.CM_LAG_DAYS * DAY))
    out["fred"] = {}
    if full:
        weekday = days[pd.to_datetime(days, unit="ms").dayofweek < 5]
        for series, lag in fp.FRED_LAG_DAYS.items():
            out["fred"][series] = _series(weekday + lag * DAY, 100 * np.exp(np.cumsum(rng.normal(0, 0.01,
                                                                                                    len(weekday)))))
        out["stable"] = _series(days + DAY, 1e11 * np.exp(np.cumsum(rng.normal(0.0005, 0.001, N_DAYS))))
        out["breadth"] = pd.DataFrame({k: rng.uniform(0, 1, N_DAYS) for k in fp.BREADTH}, index=pd.Index(days + DAY))
    else:
        out["stable"], out["breadth"] = None, None
    return out


TIMES = T0 + np.arange(310, 400, dtype=np.int64) * DAY


@pytest.fixture(scope="module")
def inputs():
    return synthetic_inputs()


@pytest.mark.parametrize("asset", fp.SYMBOLS)
def test_every_family_has_values_after_its_warm_up(inputs, asset):
    out = fp.build(inputs, asset, TIMES)
    assert list(out.columns) == list(fp.FEATURES) and list(out.index) == list(TIMES)
    empty = [c for c in fp.FEATURES if not np.isfinite(out[c]).mean() > 0.9]
    assert empty == []


@pytest.mark.parametrize("asset", fp.SYMBOLS)
def test_nothing_is_read_before_it_is_known(inputs, asset):
    cuts = [int(TIMES[10]) + 7 * H, int(TIMES[40]), int(TIMES[70]) + 13 * H]
    assert violations(lambda m: fp.build(m, asset, TIMES), inputs, cuts) == []


def test_values_by_hand(inputs):
    s, t = "BTCUSDT", int(TIMES[20])
    out = fp.build(inputs, s, np.array([t])).iloc[0]
    h = inputs["hourly"][s]
    close_at = lambda when: h.loc[h["available_at"] == when, "close"].iloc[0]       # noqa: E731
    assert out["ret_7"] == pytest.approx(np.log(close_at(t) / close_at(t - 7 * DAY)))
    rates = inputs["funding"][s]
    assert out["funding_3d"] == pytest.approx(rates[rates.index <= t].iloc[-9:].mean())
    sp = inputs["fred"]["SP500"]
    then = sp[sp.index <= t - 28 * DAY].iloc[-1]
    assert out["spx_ret_28"] == pytest.approx(np.log(sp[sp.index <= t].iloc[-1] / then))


def test_a_value_is_used_only_from_the_day_it_is_known():
    base = synthetic_inputs(1)
    d = T0 + 350 * DAY                                                   # a value dated d, known at d + 2 days
    for when, seen in ((d + DAY, False), (d + 2 * DAY, True)):
        m = dict(base)
        m["fred"] = {"VIXCLS": _series([d + fp.FRED_LAG_DAYS["VIXCLS"] * DAY], [99.0])}
        assert bool(fp.build(m, "BTCUSDT", np.array([when]))["vix"].iloc[0] == 99.0) is seen


def test_a_missing_source_gives_missing_values_not_an_error():
    out = fp.build(synthetic_inputs(2, full=False), "ETHUSDT", TIMES)
    missing = ("basis_1d", "depth_imb_1", "dvol", "bvol", "breadth_ma30", "spx_ret_28", "stable_chg_7",
               "exch_netflow_7", "premium_1d")
    assert all(out[c].isna().all() for c in missing)
    assert np.isfinite(out["ret_30"]).all() and np.isfinite(out["oi_chg_7d"]).all()


def test_the_market_breadth_counts_what_each_day_knows():
    idx = pd.Index(T0 + np.arange(1, 61, dtype=np.int64) * DAY)
    up = np.exp(np.linspace(0, 0.5, 60))
    closes = pd.DataFrame({"BTCUSDT": up, "AUSDT": up[::-1], "BUSDT": up ** 2}, index=idx)
    funding = pd.DataFrame({"BTCUSDT": 0.0001, "AUSDT": -0.0001, "BUSDT": np.nan}, index=idx)
    out = fp.market_breadth(closes, funding)
    assert out["breadth_ma30"].iloc[-1] == pytest.approx(2 / 3)       # BTC and B above their means, A below
    assert out["funding_pos_share"].iloc[-1] == pytest.approx(0.5)   # B unknown, not negative
    assert out["alt_minus_btc_30"].iloc[-1] == pytest.approx(np.median([np.log(up[-1] / up[-31]) * -1,
                                                                        2 * np.log(up[-1] / up[-31])])
                                                             - np.log(up[-1] / up[-31]))
    cuts = [int(idx[35]), int(idx[50])]
    assert violations(lambda m: fp.market_breadth(m["c"], m["f"]), {"c": closes, "f": funding}, cuts) == []


def test_the_market_panel_keeps_complete_days_before_the_limit_and_leaves_out_what_it_is_told(tmp_path):
    import gzip
    for symbol, price in (("BTCUSDT", 100.0), ("XAUUSDT", 2000.0), ("AUSDT", 1.0)):
        t = T0 + np.arange(24 * 3 + 5, dtype=np.int64) * H                 # 3 whole days and 5 hours
        with gzip.open(tmp_path / f"{symbol}_1h.csv.gz", "wt") as h:
            pd.DataFrame({"open_time": t, "close": price + np.arange(len(t))}).to_csv(h, index=False)
        ft = T0 + np.arange(12, dtype=np.int64) * 8 * H
        with gzip.open(tmp_path / f"{symbol}_funding.csv.gz", "wt") as h:
            pd.DataFrame({"funding_time": ft, "rate": 0.0001}).to_csv(h, index=False)
    closes, funding = fp.load_market(tmp_path, T0 + 3 * DAY, exclude=frozenset({"XAUUSDT"}))
    assert list(closes.columns) == ["AUSDT", "BTCUSDT"]
    assert list(closes.index) == [T0 + DAY, T0 + 2 * DAY, T0 + 3 * DAY]   # the partial fourth day is not known
    assert closes.loc[T0 + DAY, "BTCUSDT"] == 123.0
    assert funding.loc[T0 + 3 * DAY, "BTCUSDT"] == pytest.approx(0.0001)
    assert np.isnan(funding.loc[T0 + DAY, "BTCUSDT"])                      # fewer than 9 settlements by then
