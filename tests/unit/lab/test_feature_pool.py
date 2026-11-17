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


# -- technical indicators (owner 2026-11-17: "下一步如何去收集特征池包括技术分析指标", "执行") -------------------
def _bars_of(close, high=None, low=None, volume=None, step=DAY):
    close = np.asarray(close, dtype=float)
    high = close * 1.01 if high is None else np.asarray(high, dtype=float)
    low = close * 0.99 if low is None else np.asarray(low, dtype=float)
    volume = np.full(len(close), 1000.0) if volume is None else np.asarray(volume, dtype=float)
    index = pd.Index(T0 + np.arange(1, len(close) + 1, dtype=np.int64) * step)
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": high, "low": low, "close": close,
                         "volume": volume}, index=index)


def test_rsi_stochastic_donchian_and_bollinger_by_hand():
    up = _bars_of(100 + np.arange(80.0))
    ta = fp.technical(up)
    assert ta["rsi"].iloc[-1] == pytest.approx(100.0)                     # only rises
    assert ta["stoch_k"].iloc[-1] == pytest.approx((up["close"].iloc[-1] - up["low"].iloc[-14:].min())
                                                   / (up["high"].iloc[-14:].max() - up["low"].iloc[-14:].min()))
    assert ta["donchian_20"].iloc[-1] == pytest.approx((up["close"].iloc[-1] - up["low"].iloc[-20:].min())
                                                       / (up["high"].iloc[-20:].max() - up["low"].iloc[-20:].min()))
    swing = _bars_of(100 + np.where(np.arange(80) % 2, 1.0, -1.0))
    assert fp.technical(swing)["rsi"].iloc[-1] == pytest.approx(50.0, abs=3.0)
    c = up["close"].iloc[-20:]
    mid, sd = c.mean(), c.std(ddof=0)
    assert ta["bb_pctb"].iloc[-1] == pytest.approx((c.iloc[-1] - (mid - 2 * sd)) / (4 * sd))
    assert ta["bb_width"].iloc[-1] == pytest.approx(4 * sd / mid)


def test_trend_indicators_point_the_way_of_a_steady_trend():
    rising = fp.technical(_bars_of(100 * 1.01 ** np.arange(260)))
    falling = fp.technical(_bars_of(100 * 0.99 ** np.arange(260)))
    for name in ("di_diff", "aroon_osc", "ma_stack", "psar_dir"):
        assert rising[name].iloc[-1] > 0 > falling[name].iloc[-1], name
    k = np.arange(260.0)                                                   # the MACD histogram is acceleration
    assert fp.technical(_bars_of(100 + 0.01 * k ** 2))["macd_hist"].iloc[-1] > 0
    assert fp.technical(_bars_of(100 + 20 * np.sqrt(k)))["macd_hist"].iloc[-1] < 0
    assert rising["adx"].iloc[-1] > 40 and rising["ichimoku_dist"].iloc[-1] > 0 > falling["ichimoku_dist"].iloc[-1]
    assert rising["chop"].iloc[-1] < 40                                    # trending, not choppy


def test_volatility_squeeze_and_choppiness():
    rng = np.random.default_rng(3)
    n = 200
    close = 100 + np.cumsum(rng.normal(0, 2, n))
    close[120:] = close[119] + rng.normal(0, 0.05, n - 120)               # closes go quiet ...
    high, low = close + 3.0, close - 3.0                                    # ... while the bars stay wide
    ta = fp.technical(_bars_of(close, high, low))
    assert ta["squeeze_on"].iloc[-1] == 1.0 and ta["squeeze_bars"].iloc[-1] >= 50
    assert ta["bb_kc_ratio"].iloc[-1] < 1.0 and ta["chop"].iloc[-1] > 60
    assert ta["atr_pct"].iloc[-1] == pytest.approx(6.0 / close[-1], rel=0.2)


def test_volume_indicators_and_hurst():
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    ta = fp.technical(_bars_of(close, volume=rng.uniform(500, 1500, 400)))
    assert -1 <= ta["obv_slope"].min() and ta["obv_slope"].max() <= 1
    assert 0 <= ta["mfi"].min() and ta["mfi"].max() <= 100
    assert abs(ta["hurst"].iloc[-150:].mean() - 0.5) < 0.15                # a random walk
    trend = fp.technical(_bars_of(100 + np.arange(400.0) + rng.normal(0, 0.1, 400)))
    assert trend["hurst"].iloc[-1] > 0.8                                    # persistent


def test_bars_of_four_hours_are_complete_and_known_at_their_end():
    h = synthetic_inputs(5)["hourly"]["BTCUSDT"]
    bars = fp.bars(h, 4)
    assert (bars.index.to_numpy() % (4 * H) == 0).all()
    first = h.iloc[:4]
    assert bars["high"].iloc[0] == first["high"].max() and bars["close"].iloc[0] == first["close"].iloc[-1]
    assert bars.index[0] == first["available_at"].iloc[-1]
    partial = fp.bars(h.iloc[:6], 4)
    assert len(partial) == 1                                                # the 2 hours of the next bar are not one


def test_the_pool_carries_every_indicator_on_both_timeframes(inputs):
    names = [f"ta_{n}_{tf}" for tf in ("4h", "1d") for n in fp.TECHNICAL]
    assert set(names) <= set(fp.FEATURES)
    out = fp.build(inputs, "BTCUSDT", TIMES)
    assert all(np.isfinite(out[c]).mean() > 0.9 for c in names)


# -- the second version (owner 2026-11-17: "还有哪些特征是需要用到", then "执行") ---------------------------------
def v2_inputs(seed=0):
    out = synthetic_inputs(seed)
    rng = np.random.default_rng(seed + 100)
    days = T0 + np.arange(N_DAYS, dtype=np.int64) * DAY
    out["quarterly"], out["coinbase"], out["upbit"] = {}, {}, {}
    for s in fp.SYMBOLS:
        h = out["hourly"][s]
        known = h["available_at"].to_numpy(dtype=np.int64)
        perp = h["close"].to_numpy(float)
        expiry = (known // (90 * DAY) + 1) * 90 * DAY + 8 * H                 # a contract every 90 days
        expiry = np.where(expiry - known < 7 * DAY, expiry + 90 * DAY, expiry)
        out["quarterly"][s] = pd.DataFrame({"close": perp * (1 + 0.1 * (expiry - known) / DAY / 365),
                                            "expiry": expiry}, index=pd.Index(known))
        spot = out["spot"][s]["close"].to_numpy(float)
        out["coinbase"][s] = _series(known, spot * 1.002)
        out["upbit"][s] = _series(known, spot * 1300 * (1 + 0.02 * np.sin(np.arange(len(h)) / 500)))
    out["usdt_usd"] = _series(out["hourly"]["BTCUSDT"]["available_at"], np.ones(len(out["hourly"]["BTCUSDT"])))
    out["fng"] = _series(days + DAY, rng.uniform(10, 90, N_DAYS))
    out["fomc"] = T0 + np.arange(20, N_DAYS + 90, 45, dtype=np.int64) * DAY      # the schedule runs ahead
    weekday = days[pd.to_datetime(days, unit="ms").dayofweek < 5]
    for series, lag in fp.FRED_V2_LAG_DAYS.items():
        out["fred"][series] = _series(weekday + lag * DAY, 50 + np.cumsum(rng.normal(0, 0.5, len(weekday))))
    for s in fp.SYMBOLS:
        out["cm"][s] = out["cm"][s].assign(CapMVRVCur=rng.uniform(1, 3, N_DAYS), HashRate=rng.uniform(1e8, 2e8, N_DAYS))
    return out


@pytest.fixture(scope="module")
def inputs2():
    return v2_inputs()


@pytest.mark.parametrize("asset", fp.SYMBOLS)
def test_every_second_version_feature_has_values(inputs2, asset):
    out = fp.build(inputs2, asset, TIMES, version=2)
    assert list(out.columns) == list(fp.FEATURES_V2)
    assert [c for c in fp.V2 if not np.isfinite(out[c]).mean() > 0.9] == []
    assert list(fp.build(inputs2, asset, TIMES).columns) == list(fp.FEATURES)       # the first version unchanged


@pytest.mark.parametrize("asset", fp.SYMBOLS)
def test_the_second_version_reads_nothing_before_it_is_known(inputs2, asset):
    cuts = [int(TIMES[10]) + 7 * H, int(TIMES[40]), int(TIMES[70]) + 13 * H]
    assert violations(lambda m: fp.build(m, asset, TIMES, version=2), inputs2, cuts) == []


def test_second_version_values_by_hand(inputs2):
    t = int(TIMES[30])
    row = fp.build(inputs2, "BTCUSDT", np.array([t]), version=2).iloc[0]
    assert row["qbasis_annual"] == pytest.approx(0.1)                       # futures priced 10 % a year above
    assert row["cb_premium_1d"] == pytest.approx(0.002)
    assert row["dow"] == pd.Timestamp(t, unit="ms").dayofweek
    nxt = inputs2["fomc"][inputs2["fomc"] + 18 * H > t].min()
    assert row["days_to_fomc"] == pytest.approx((nxt + 18 * H - t) / DAY)
    assert 0 < row["days_to_monthly_expiry"] <= 31 and row["days_to_monthly_expiry"] <= row["days_to_quarterly_expiry"]


def test_the_front_quarter_is_the_nearest_with_a_week_left():
    q = pd.DataFrame({"open_time": [T0, T0, T0 + H, T0 + H],
                      "expiry": [T0 + 5 * DAY, T0 + 90 * DAY, T0 + 8 * DAY, T0 + 90 * DAY],
                      "close": [1.0, 2.0, 3.0, 4.0], "contract": ["a", "b", "c", "d"]})
    front = fp.front_quarter(q, T0 + 10 * H)
    assert list(front.index) == [T0 + H, T0 + 2 * H] and list(front["close"]) == [2.0, 3.0]
