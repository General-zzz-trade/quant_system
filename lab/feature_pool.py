"""The BTC/ETH feature pool (owner 2026-11-17: "来使用加入特征池的方式去做，参考更多的数据去做", "执行").

Every input is a frame or series indexed by the time its row is known (epoch ms), except the hourly bars (an
``available_at`` column) and Binance's metrics (a ``timestamp`` column, used ``METRIC_LAG`` later), so that
lab/causality.py reads them all. ``build`` returns one row per decision time (a UTC midnight) from values known at
or before it; a value older than its family's age limit is missing, never carried on. A source a snapshot lacks
(depth before 2023, BVOL, a FRED series) gives missing values, never an error.

Families (``FEATURES``): price and volatility, flow, funding and premium, positioning (open interest and long/short
ratios), spot against perpetual, order book depth, implied volatility, the other coin, the whole market (``breadth``,
from ``market_breadth``), macro (FRED), stablecoins and on-chain (Coin Metrics).
"""
from __future__ import annotations

import gzip
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lab.pair import METRIC_LAG

H, DAY = 3_600_000, 86_400_000
SYMBOLS = ("BTCUSDT", "ETHUSDT")
FRED_LAG_DAYS = {"DTWEXBGS": 9, "DGS10": 2, "SP500": 2, "NASDAQCOM": 2, "VIXCLS": 2}
STABLE_LAG_DAYS, CM_LAG_DAYS = 1, 2
CM_ASSET = {"BTCUSDT": "btc", "ETHUSDT": "eth"}
BVOL_SYMBOL = {"BTCUSDT": "BTCBVOLUSDT", "ETHUSDT": "ETHBVOLUSDT"}
DVOL_CURRENCY = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}

FEATURES = (
    # price and volatility
    "ret_1", "ret_3", "ret_7", "ret_14", "ret_30", "ret_60", "ret_90", "rv_7", "rv_30", "rv_ratio_7_30",
    "rv_pct_365", "range_ratio_14_90", "er_14", "er_30", "dist_ma_20", "dist_ma_50", "dist_ma_200", "skew_30",
    "vov_60", "hourly_rv_ratio",
    # flow
    "taker_1d", "taker_7d", "volume_z_30",
    # funding and premium
    "funding_3d", "funding_z_90", "funding_chg_7d", "premium_1d", "premium_7d",
    # positioning
    "oi_chg_1d", "oi_chg_7d", "oi_chg_30d", "oi_price_div_7d", "top_ls_z_30", "acct_ls_z_30", "taker_ls_1d",
    # spot against perpetual
    "basis_1d", "basis_7d", "spot_share_7d", "spot_taker_1d",
    # order book depth
    "depth_imb_1", "depth_imb_5", "depth_1_z_30",
    # implied volatility
    "dvol", "dvol_chg_7d", "iv_minus_rv", "bvol", "bvol_chg_7d",
    # the other coin
    "other_ret_7", "other_ret_30", "rel_ret_7", "rel_ret_30", "corr_30",
    # the whole market
    "breadth_ma30", "dispersion_7", "funding_pos_share", "alt_minus_btc_30",
    # macro
    "spx_ret_28", "ndx_ret_28", "vix", "vix_chg_7", "dxy_ret_30", "us10y_chg_30",
    # stablecoins and on-chain
    "stable_chg_7", "stable_chg_30", "exch_netflow_7", "exch_supply_chg_30", "active_addr_z_90", "tx_z_90",
)
BREADTH = ("breadth_ma30", "dispersion_7", "funding_pos_share", "alt_minus_btc_30")
# Technical indicators on 4-hour and daily bars (owner 2026-11-17: "下一步如何去收集特征池包括技术分析指标", "执行")
TECHNICAL = ("rsi", "macd_hist", "adx", "di_diff", "aroon_osc", "ichimoku_dist", "donchian_20", "donchian_55",
             "ma_stack", "psar_dir", "stoch_k", "stoch_d", "cci", "bb_pctb", "bb_width", "atr_pct", "squeeze_on",
             "squeeze_bars", "bb_kc_ratio", "chop", "obv_slope", "mfi", "vwap_dev", "vol_rel", "hurst")
TIMEFRAMES = {"4h": 4, "1d": 24}
FEATURES = FEATURES + tuple(f"ta_{n}_{tf}" for tf in TIMEFRAMES for n in TECHNICAL)
# The second version (owner 2026-11-17: "还有哪些特征是需要用到", then "执行"); FEATURES stays the first version's.
FRED_V2_LAG_DAYS = {"WALCL": 3, "WTREGEN": 3, "RRPONTSYD": 1, "BAMLH0A0HYM2": 2, "DGS2": 2, "T10Y2Y": 2,
                    "DCOILWTICO": 9, "DEXJPUS": 9, "DEXKOUS": 9}
FG_LAG_DAYS = 1
V2 = ("rv_intraday_1d", "har_rv_7", "har_rv_30", "parkinson_7", "garman_klass_7", "ewma_vol", "jump_share_7",
      "downside_share_7", "vol_of_rv_30",
      "walcl_chg_z", "tga_chg_z", "rrp_chg_z", "hy_spread", "hy_spread_chg_28", "us2y_chg_30", "curve_10y2y",
      "oil_ret_28", "usdjpy_ret_28",
      "qbasis_annual", "qbasis_chg_7", "cb_premium_1d", "cb_premium_7d", "kimchi_dev_30", "kimchi_chg_7",
      "fng", "fng_chg_7",
      "dow", "days_to_monthly_expiry", "days_to_quarterly_expiry", "days_to_fomc", "days_since_fomc",
      "mvrv", "mvrv_chg_30", "hashrate_chg_30")
FEATURES_V2 = FEATURES + V2
COINBASE_PRODUCT = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}
UPBIT_MARKET = {"BTCUSDT": "KRW-BTC", "ETHUSDT": "KRW-ETH"}


# -- loading ---------------------------------------------------------------------------------------------------
def _ms(day: str) -> int:
    d = date.fromisoformat(day)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as handle:
        return pd.read_csv(handle)


def _known(frame: pd.DataFrame, index: np.ndarray, limit: int) -> pd.DataFrame:
    out = frame.set_index(pd.Index(index.astype(np.int64))).sort_index()
    return out[out.index <= limit]


def load_inputs(public_dir: Path, feature_dir: Path, limit: int, *, breadth: pd.DataFrame | None = None) -> dict:
    """Every source of both coins, each row kept only when it is known at or before ``limit``."""
    public_dir, feature_dir = Path(public_dir), Path(feature_dir)
    out: dict = {k: {} for k in ("hourly", "funding", "metrics", "spot", "premium", "depth", "bvol", "dvol", "cm")}
    for s in SYMBOLS:
        hourly = pd.read_csv(public_dir / f"{s}_1h.csv")
        hourly["available_at"] = hourly["open_time"] + H
        out["hourly"][s] = hourly[hourly["available_at"] <= limit].reset_index(drop=True)
        funding = pd.read_csv(public_dir / f"{s}_funding.csv")
        out["funding"][s] = _known(funding[["rate"]], funding["funding_time"].to_numpy(), limit)["rate"]
        metrics = pd.read_csv(public_dir / f"{s}_metrics.csv")
        out["metrics"][s] = metrics[metrics["timestamp"] + METRIC_LAG <= limit].reset_index(drop=True)
        spot = _read(feature_dir / f"{s}_spot_1h.csv.gz")
        if spot is not None:
            out["spot"][s] = _known(spot[["close", "quote_volume", "taker_buy_quote_volume"]],
                                    spot["open_time"].to_numpy() + H, limit)
        premium = _read(feature_dir / f"{s}_premium_1h.csv.gz")
        if premium is not None:
            out["premium"][s] = _known(premium[["close"]], premium["open_time"].to_numpy() + H, limit)["close"]
        depth = _read(feature_dir / f"{s}_depth_1h.csv.gz")
        if depth is not None and len(depth):
            wide = depth.pivot_table(index="hour", columns="percentage", values="mean_notional")
            cols = {"bid_1": -1, "ask_1": 1, "bid_5": -5, "ask_5": 5}
            wide = pd.DataFrame({k: wide[v] if v in wide.columns else np.nan for k, v in cols.items()})
            out["depth"][s] = _known(wide, wide.index.to_numpy() + H, limit)
        bvol = _read(feature_dir / f"{BVOL_SYMBOL[s]}_1h.csv.gz")
        if bvol is not None and len(bvol):
            out["bvol"][s] = _known(bvol[["last_value"]], bvol["hour"].to_numpy() + H, limit)["last_value"]
        dvol = _read(feature_dir / f"{DVOL_CURRENCY[s]}_dvol_1h.csv.gz")
        if dvol is not None and len(dvol):
            out["dvol"][s] = _known(dvol[["close"]], dvol["open_time"].to_numpy() + H, limit)["close"]
        cm = _read(feature_dir / f"coinmetrics_{CM_ASSET[s]}.csv.gz")
        if cm is not None and len(cm):
            known = np.array([_ms(d) for d in cm["date"]]) + CM_LAG_DAYS * DAY
            out["cm"][s] = _known(cm.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce"), known, limit)
    for key in ("quarterly", "coinbase", "upbit"):
        out[key] = {}
    for s in SYMBOLS:
        q = _read(feature_dir / f"{s}_quarterly_1h.csv.gz")
        if q is not None and len(q):
            out["quarterly"][s] = front_quarter(q, limit)
        cb = _read(feature_dir / f"coinbase_{COINBASE_PRODUCT[s]}_1h.csv.gz")
        if cb is not None and len(cb):
            out["coinbase"][s] = _known(cb[["close"]], cb["open_time"].to_numpy() + H, limit)["close"]
        up = _read(feature_dir / f"upbit_{UPBIT_MARKET[s]}_1h.csv.gz")
        if up is not None and len(up):
            out["upbit"][s] = _known(up[["close"]], up["open_time"].to_numpy() + H, limit)["close"]
    usdt = _read(feature_dir / "coinbase_USDT-USD_1h.csv.gz")
    out["usdt_usd"] = None if usdt is None or not len(usdt) else _known(
        usdt[["close"]], usdt["open_time"].to_numpy() + H, limit)["close"]
    fg = _read(feature_dir / "fear_greed.csv.gz")
    out["fng"] = None if fg is None or not len(fg) else _known(
        fg[["value"]], np.array([_ms(d) for d in fg["date"]]) + FG_LAG_DAYS * DAY, limit)["value"]
    fomc = _read(feature_dir / "fomc_days.csv.gz")
    out["fomc"] = np.array([], dtype=np.int64) if fomc is None else np.array([_ms(d) for d in fomc["date"]],
                                                                             dtype=np.int64)
    out["fred"] = {}
    for series, lag in {**FRED_LAG_DAYS, **FRED_V2_LAG_DAYS}.items():
        f = _read(feature_dir / f"fred_{series}.csv.gz")
        if f is not None and len(f):
            out["fred"][series] = _known(f[["value"]], np.array([_ms(d) for d in f["date"]]) + lag * DAY,
                                         limit)["value"]
    stable = _read(feature_dir / "stablecoins_usd.csv.gz")
    out["stable"] = None if stable is None else _known(
        stable[["total_usd"]], np.array([_ms(d) for d in stable["date"]]) + STABLE_LAG_DAYS * DAY, limit)["total_usd"]
    out["breadth"] = None if breadth is None else breadth[breadth.index <= limit]
    return out


# -- helpers: values known at each decision time -----------------------------------------------------------------
def _last(series: pd.Series | None, times: np.ndarray, max_age: int) -> np.ndarray:
    """The last value known at or before each time; missing when none is, or when it is older than ``max_age``."""
    out = np.full(len(times), np.nan)
    if series is None or not len(series):
        return out
    st, v = series.index.to_numpy(dtype=np.int64), series.to_numpy(dtype=float)
    idx = np.searchsorted(st, times, side="right") - 1
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    out[ok & (times - st[np.maximum(idx, 0)] > max_age)] = np.nan
    return out


def _window_mean(series: pd.Series | None, times: np.ndarray, span: int, min_count: int) -> np.ndarray:
    """Mean of the values known in (t - span, t]; missing with fewer than ``min_count`` of them."""
    if series is None or not len(series):
        return np.full(len(times), np.nan)
    st, v = series.index.to_numpy(dtype=np.int64), series.to_numpy(dtype=float)
    finite = np.isfinite(v)
    csum = np.concatenate([[0.0], np.cumsum(np.where(finite, v, 0.0))])
    ccount = np.concatenate([[0], np.cumsum(finite)])
    hi, lo = np.searchsorted(st, times, side="right"), np.searchsorted(st, times - span, side="right")
    count = ccount[hi] - ccount[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (csum[hi] - csum[lo]) / count
    return np.where(count >= min_count, mean, np.nan)


def _z(x: pd.Series, n: int) -> pd.Series:
    return (x - x.rolling(n, min_periods=n).mean()) / x.rolling(n, min_periods=n).std()


def _log_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.log(a / b)
    return np.where(np.isfinite(out), out, np.nan)


def daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily bars of complete UTC days, indexed by the day's end (when the day is known)."""
    day = hourly["open_time"] // DAY
    g = hourly.groupby(day)
    out = pd.DataFrame({"close": g["close"].last(), "high": g["high"].max(), "low": g["low"].min(),
                        "quote": g["quote_volume"].sum(), "taker": g["taker_buy_quote_volume"].sum(),
                        "hours": g["close"].count()})
    out = out[out["hours"] == 24]
    out.index = (out.index.to_numpy(dtype=np.int64) + 1) * DAY
    return out


def bars(hourly: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Bars of ``hours`` hours (aligned to UTC midnight), complete ones only, indexed by their end (when they are
    known); volume is the quote volume."""
    group = hourly["open_time"] // (hours * H)
    g = hourly.groupby(group)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["quote_volume"].sum(), "n": g["close"].count()})
    out = out[out["n"] == hours].drop(columns="n")
    out.index = (out.index.to_numpy(dtype=np.int64) + 1) * hours * H
    return out


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _psar(high: np.ndarray, low: np.ndarray, step: float = 0.02, top: float = 0.2) -> np.ndarray:
    """Parabolic SAR direction (+1 rising, -1 falling), the standard rule."""
    n = len(high)
    out = np.zeros(n)
    if n < 2:
        return out
    up, af, ep, sar = True, step, high[0], low[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < sar:
                up, sar, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, top)
        else:
            sar = max(sar, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > sar:
                up, sar, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, top)
        out[i] = 1.0 if up else -1.0
    return out


def _hurst(close: pd.Series, window: int = 100, lags: tuple[int, ...] = tuple(range(2, 17))) -> pd.Series:
    """Rolling Hurst exponent: the slope of log(spread of k-bar log changes) on log(k), over ``window`` bars."""
    logp = np.log(close)
    x = np.log(np.array(lags, dtype=float))
    x = x - x.mean()
    ys = pd.concat([np.log((logp - logp.shift(k)).rolling(window, min_periods=window).std()) for k in lags], axis=1)
    return pd.Series((ys.sub(ys.mean(axis=1), axis=0) * x).sum(axis=1, min_count=len(lags)) / (x ** 2).sum(),
                     index=close.index)


def technical(b: pd.DataFrame) -> dict[str, pd.Series]:
    """``TECHNICAL`` on one series of bars (open, high, low, close, volume; index: when each bar is known), each
    value from that bar and the ones before it."""
    o, h, low, c, v = b["open"], b["high"], b["low"], b["close"], b["volume"]
    out: dict[str, pd.Series] = {}
    gain, loss = c.diff().clip(lower=0), (-c.diff()).clip(lower=0)
    avg_gain, avg_loss = _wilder(gain, 14), _wilder(loss, 14)
    out["rsi"] = (100 * avg_gain / (avg_gain + avg_loss)).where(avg_gain + avg_loss > 0, 50.0)
    macd = c.ewm(span=12, adjust=False, min_periods=26).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / c
    tr = pd.concat([h - low, (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(axis=1)
    up_move, down_move = h.diff(), -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr14 = _wilder(tr, 14)
    plus_di, minus_di = 100 * _wilder(plus_dm, 14) / atr14, 100 * _wilder(minus_dm, 14) / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    out["adx"], out["di_diff"] = _wilder(dx, 14), plus_di - minus_di
    since_high = h.rolling(26, min_periods=26).apply(lambda w: 25 - int(np.argmax(w)), raw=True)
    since_low = low.rolling(26, min_periods=26).apply(lambda w: 25 - int(np.argmin(w)), raw=True)
    out["aroon_osc"] = 100 * (since_low - since_high) / 25

    def mid(n: int) -> pd.Series:
        return (h.rolling(n, min_periods=n).max() + low.rolling(n, min_periods=n).min()) / 2
    span_a, span_b = ((mid(9) + mid(26)) / 2).shift(26), mid(52).shift(26)
    cloud_top, cloud_bottom = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    dist = pd.Series(np.where(c > cloud_top, (c - cloud_top) / c, np.where(c < cloud_bottom, (c - cloud_bottom) / c,
                                                                          0.0)), index=c.index)
    out["ichimoku_dist"] = dist.where(cloud_top.notna())
    for n in (20, 55):
        hi, lo = h.rolling(n, min_periods=n).max(), low.rolling(n, min_periods=n).min()
        out[f"donchian_{n}"] = (c - lo) / (hi - lo)
    ma20, ma50, ma200 = (c.rolling(n, min_periods=n).mean() for n in (20, 50, 200))
    stack = np.where((ma20 > ma50) & (ma50 > ma200), 1.0, np.where((ma20 < ma50) & (ma50 < ma200), -1.0, 0.0))
    out["ma_stack"] = pd.Series(stack, index=c.index).where(ma200.notna())
    out["psar_dir"] = pd.Series(_psar(h.to_numpy(float), low.to_numpy(float)), index=c.index).where(
        pd.Series(np.arange(len(c)) >= 20, index=c.index))
    hi14, lo14 = h.rolling(14, min_periods=14).max(), low.rolling(14, min_periods=14).min()
    out["stoch_k"] = (c - lo14) / (hi14 - lo14)
    out["stoch_d"] = out["stoch_k"].rolling(3, min_periods=3).mean()
    tp = (h + low + c) / 3
    mean_dev = tp.rolling(20, min_periods=20).apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True)
    out["cci"] = (tp - tp.rolling(20, min_periods=20).mean()) / (0.015 * mean_dev)
    mid20, sd20 = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std(ddof=0)
    out["bb_pctb"] = (c - (mid20 - 2 * sd20)) / (4 * sd20)
    out["bb_width"] = 4 * sd20 / mid20
    out["atr_pct"] = atr14 / c
    atr20 = _wilder(tr, 20)
    kc_width = 2 * 1.5 * atr20
    out["bb_kc_ratio"] = (4 * sd20) / kc_width
    squeeze = (out["bb_kc_ratio"] < 1).astype(float).where(out["bb_kc_ratio"].notna())
    run = squeeze.fillna(0.0).groupby((squeeze.fillna(0.0) != squeeze.fillna(0.0).shift()).cumsum()).cumsum()
    out["squeeze_on"], out["squeeze_bars"] = squeeze, run.where(squeeze.notna())
    span14 = h.rolling(14, min_periods=14).max() - low.rolling(14, min_periods=14).min()
    out["chop"] = 100 * np.log10(tr.rolling(14, min_periods=14).sum() / span14) / np.log10(14)
    obv = (np.sign(c.diff()).fillna(0.0) * v).cumsum()
    out["obv_slope"] = (obv - obv.shift(20)) / v.rolling(20, min_periods=20).sum()
    flow = tp * v
    pos = flow.where(tp.diff() > 0, 0.0).rolling(14, min_periods=14).sum()
    neg = flow.where(tp.diff() < 0, 0.0).rolling(14, min_periods=14).sum()
    out["mfi"] = (100 * pos / (pos + neg)).where(pos + neg > 0, 50.0)
    vwap = (tp * v).rolling(20, min_periods=20).sum() / v.rolling(20, min_periods=20).sum()
    out["vwap_dev"] = c / vwap - 1
    out["vol_rel"] = v / v.rolling(20, min_periods=20).mean()
    out["hurst"] = _hurst(c)
    del o
    return out


def _technical(hourly: pd.DataFrame, times: np.ndarray) -> dict:
    out = {}
    for tf, hours in TIMEFRAMES.items():
        ta = technical(bars(hourly, hours))
        for name in TECHNICAL:
            out[f"ta_{name}_{tf}"] = _last(ta[name], times, 2 * hours * H)
    return out


def _efficiency(close: pd.Series, n: int) -> pd.Series:
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return ((close - close.shift(n)).abs() / path).where(path > 0)


# -- the pool --------------------------------------------------------------------------------------------------
def _price(d: pd.DataFrame, hourly: pd.DataFrame, times: np.ndarray) -> dict:
    close = d["close"]
    r = np.log(close).diff()
    rv7, rv30 = r.rolling(7, min_periods=7).std(), r.rolling(30, min_periods=30).std()
    rng14 = (d["high"].rolling(14, min_periods=14).max() - d["low"].rolling(14, min_periods=14).min()) / close
    daily_series = {f"ret_{k}": np.log(close / close.shift(k)) for k in (1, 3, 7, 14, 30, 60, 90)}
    daily_series.update({
        "rv_7": rv7, "rv_30": rv30, "rv_ratio_7_30": rv7 / rv30,
        "rv_pct_365": rv30.rolling(365, min_periods=180).rank(pct=True),
        "range_ratio_14_90": rng14 / rng14.rolling(90, min_periods=90).mean(),
        "er_14": _efficiency(close, 14), "er_30": _efficiency(close, 30),
        **{f"dist_ma_{n}": close / close.rolling(n, min_periods=n).mean() - 1 for n in (20, 50, 200)},
        "skew_30": r.rolling(30, min_periods=30).skew(),
        "vov_60": rv7.rolling(60, min_periods=60).std() / rv30,
        "taker_1d": d["taker"] / d["quote"],
        "taker_7d": d["taker"].rolling(7, min_periods=7).sum() / d["quote"].rolling(7, min_periods=7).sum(),
        "volume_z_30": _z(np.log(d["quote"]), 30),
    })
    out = {k: _last(v, times, 2 * DAY) for k, v in daily_series.items()}
    hr = pd.Series(np.log(hourly["close"].to_numpy(float)), index=hourly["available_at"].to_numpy()).diff()
    ratio = hr.rolling(24, min_periods=24).std() / hr.rolling(720, min_periods=600).std()
    out["hourly_rv_ratio"] = _last(ratio, times, 2 * H)
    return out


def _derivatives(inputs: dict, s: str, d: pd.DataFrame, times: np.ndarray, ret_7: np.ndarray) -> dict:
    funding = inputs["funding"][s]
    f3 = funding.rolling(9, min_periods=9).mean() if len(funding) else funding
    f3_daily = pd.Series(_last(f3, d.index.to_numpy(np.int64), 9 * H), index=d.index)
    out = {"funding_3d": _last(f3, times, 9 * H), "funding_z_90": _last(_z(f3_daily, 90), times, 2 * DAY)}
    out["funding_chg_7d"] = out["funding_3d"] - _last(f3, times - 7 * DAY, 9 * H)
    premium = inputs["premium"].get(s)
    out["premium_1d"] = _window_mean(premium, times, DAY, 20)
    out["premium_7d"] = _window_mean(premium, times, 7 * DAY, 140)
    m = inputs["metrics"][s]
    stamp = m["timestamp"].to_numpy(dtype=np.int64) + METRIC_LAG
    oi = pd.Series(pd.to_numeric(m["sum_open_interest"], errors="coerce").to_numpy(float), index=stamp)
    now = _last(oi, times, H)
    for k in (1, 7, 30):
        out[f"oi_chg_{k}d"] = _log_ratio(now, _last(oi, times - k * DAY, H))
    out["oi_price_div_7d"] = out["oi_chg_7d"] - ret_7
    for col, name in (("sum_toptrader_long_short_ratio", "top_ls_z_30"), ("count_long_short_ratio", "acct_ls_z_30")):
        v = pd.to_numeric(m[col], errors="coerce").to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            logv = pd.Series(np.where(v > 0, np.log(v), np.nan), index=stamp)
        at_day = pd.Series(_last(logv, d.index.to_numpy(np.int64), H), index=d.index)
        out[name] = _last(_z(at_day, 30), times, 2 * DAY)
    v = pd.to_numeric(m["sum_taker_long_short_vol_ratio"], errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        taker = pd.Series(np.where(v > 0, np.log(v), np.nan), index=stamp)
    out["taker_ls_1d"] = _window_mean(taker, times, DAY, 200)
    return out


def _spot(inputs: dict, s: str, times: np.ndarray) -> dict:
    spot, hourly = inputs["spot"].get(s), inputs["hourly"][s]
    if spot is None or not len(spot):
        return {k: np.full(len(times), np.nan) for k in ("basis_1d", "basis_7d", "spot_share_7d", "spot_taker_1d")}
    perp = hourly.set_index("available_at")
    both = perp[["close", "quote_volume"]].join(spot, how="inner", rsuffix="_spot")
    basis = both["close"] / both["close_spot"] - 1
    share = both["quote_volume_spot"] / (both["quote_volume_spot"] + both["quote_volume"])
    taker = spot["taker_buy_quote_volume"] / spot["quote_volume"]
    return {"basis_1d": _window_mean(basis, times, DAY, 20), "basis_7d": _window_mean(basis, times, 7 * DAY, 140),
            "spot_share_7d": _window_mean(share, times, 7 * DAY, 140),
            "spot_taker_1d": _window_mean(taker, times, DAY, 20)}


def _depth(inputs: dict, s: str, d_index: np.ndarray, times: np.ndarray) -> dict:
    depth = inputs["depth"].get(s)
    if depth is None or not len(depth):
        return {k: np.full(len(times), np.nan) for k in ("depth_imb_1", "depth_imb_5", "depth_1_z_30")}
    imb1 = (depth["bid_1"] - depth["ask_1"]) / (depth["bid_1"] + depth["ask_1"])
    imb5 = (depth["bid_5"] - depth["ask_5"]) / (depth["bid_5"] + depth["ask_5"])
    total = np.log(depth["bid_1"] + depth["ask_1"])
    total_daily = pd.Series(_window_mean(total, d_index, DAY, 20), index=d_index)
    return {"depth_imb_1": _window_mean(imb1, times, DAY, 20), "depth_imb_5": _window_mean(imb5, times, DAY, 20),
            "depth_1_z_30": _last(_z(total_daily, 30), times, 2 * DAY)}


def _implied(inputs: dict, s: str, times: np.ndarray, rv_30: np.ndarray) -> dict:
    dvol, bvol = inputs["dvol"].get(s), inputs["bvol"].get(s)
    out = {"dvol": _last(dvol, times, 2 * H), "bvol": _last(bvol, times, 2 * H)}
    out["dvol_chg_7d"] = _log_ratio(out["dvol"], _last(dvol, times - 7 * DAY, 2 * H))
    out["bvol_chg_7d"] = _log_ratio(out["bvol"], _last(bvol, times - 7 * DAY, 2 * H))
    out["iv_minus_rv"] = out["dvol"] / 100 - rv_30 * np.sqrt(365)
    return out


def _other(d: pd.DataFrame, o: pd.DataFrame, times: np.ndarray, own: dict) -> dict:
    oc = o["close"]
    out = {f"other_ret_{k}": _last(np.log(oc / oc.shift(k)), times, 2 * DAY) for k in (7, 30)}
    out["rel_ret_7"] = own["ret_7"] - out["other_ret_7"]
    out["rel_ret_30"] = own["ret_30"] - out["other_ret_30"]
    both = pd.DataFrame({"a": np.log(d["close"]).diff(), "b": np.log(oc).diff()}).dropna()
    out["corr_30"] = _last(both["a"].rolling(30, min_periods=30).corr(both["b"]), times, 2 * DAY)
    return out


def _macro(inputs: dict, times: np.ndarray) -> dict:
    fred = inputs.get("fred", {})
    age = 6 * DAY                                                    # a long weekend plus the release lag

    def change(series_id: str, days: int, log: bool = True) -> np.ndarray:
        s = fred.get(series_id)
        now, then = _last(s, times, age), _last(s, times - days * DAY, age)
        return _log_ratio(now, then) if log else now - then
    vix = _last(fred.get("VIXCLS"), times, age)
    return {"spx_ret_28": change("SP500", 28), "ndx_ret_28": change("NASDAQCOM", 28), "vix": vix,
            "vix_chg_7": change("VIXCLS", 7), "dxy_ret_30": _log_ratio(_last(fred.get("DTWEXBGS"), times, 14 * DAY),
                                                                         _last(fred.get("DTWEXBGS"), times - 30 * DAY,
                                                                               14 * DAY)),
            "us10y_chg_30": change("DGS10", 30, log=False)}


def _chain(inputs: dict, s: str, times: np.ndarray) -> dict:
    stable = inputs.get("stable")
    out = {"stable_chg_7": _log_ratio(_last(stable, times, 3 * DAY), _last(stable, times - 7 * DAY, 3 * DAY)),
           "stable_chg_30": _log_ratio(_last(stable, times, 3 * DAY), _last(stable, times - 30 * DAY, 3 * DAY))}
    cm = inputs["cm"].get(s)
    if cm is None or not len(cm):
        return {**out, **{k: np.full(len(times), np.nan) for k in ("exch_netflow_7", "exch_supply_chg_30",
                                                                     "active_addr_z_90", "tx_z_90")}}
    net = (cm["FlowInExNtv"] - cm["FlowOutExNtv"]).rolling(7, min_periods=7).sum() / cm["SplyExNtv"]
    out["exch_netflow_7"] = _last(net, times, 3 * DAY)
    out["exch_supply_chg_30"] = _log_ratio(_last(cm["SplyExNtv"], times, 3 * DAY),
                                           _last(cm["SplyExNtv"], times - 30 * DAY, 3 * DAY))
    out["active_addr_z_90"] = _last(_z(np.log(cm["AdrActCnt"]), 90), times, 3 * DAY)
    out["tx_z_90"] = _last(_z(np.log(cm["TxCnt"]), 90), times, 3 * DAY)
    return out


# -- the second version's families ---------------------------------------------------------------------------
def front_quarter(q: pd.DataFrame, limit: int) -> pd.DataFrame:
    """For each hour the quarterly contract nearest to expiry with at least 7 days left: its close and expiry,
    indexed by when the hour's close is known."""
    q = q.assign(known=q["open_time"].to_numpy(dtype=np.int64) + H)
    q = q[(q["expiry"] - q["known"] >= 7 * DAY) & (q["known"] <= limit)].sort_values(["known", "expiry"])
    first = q.groupby("known").first()
    return pd.DataFrame({"close": first["close"].astype(float).to_numpy(),
                         "expiry": first["expiry"].to_numpy(dtype=np.int64)}, index=first.index.astype(np.int64))


def _on_days(series: pd.Series | None, times: np.ndarray, max_age: int) -> pd.Series:
    """The last known value at every UTC midnight from the series' first one to the last decision time."""
    if series is None or not len(series) or not len(times):
        return pd.Series(dtype=float)
    first = -(-int(series.index[0]) // DAY) * DAY
    days = np.arange(first, int(times.max()) + DAY, DAY, dtype=np.int64)
    return pd.Series(_last(series, days, max_age), index=days)


def _change_z(series: pd.Series | None, times: np.ndarray, max_age: int, days: int = 28) -> np.ndarray:
    """The ``days`` change, over the spread of such changes in the year before (unit-free)."""
    d = _on_days(series, times, max_age)
    if d.empty:
        return np.full(len(times), np.nan)
    chg = d - d.shift(days)
    z = chg / chg.rolling(365, min_periods=180).std()
    return _last(z, times, DAY)


def _realized(hourly: pd.DataFrame, times: np.ndarray) -> dict:
    r = pd.Series(np.log(hourly["close"].to_numpy(float)), index=hourly["available_at"].to_numpy()).diff()
    rv24 = np.sqrt((r ** 2).rolling(24, min_periods=24).sum())
    d = bars(hourly, 24)
    rv_day = pd.Series(_last(rv24, d.index.to_numpy(np.int64), 0), index=d.index)
    hl, co = np.log(d["high"] / d["low"]), np.log(d["close"] / d["open"])
    rd = np.log(d["close"]).diff()
    bv = (np.pi / 2) * (r.abs() * r.abs().shift()).rolling(168, min_periods=168).sum()
    rv168 = (r ** 2).rolling(168, min_periods=168).sum()
    down = (r.where(r < 0, 0.0) ** 2).rolling(168, min_periods=168).sum()
    daily = {"har_rv_7": np.log(rv_day.rolling(7, min_periods=7).mean()),
             "har_rv_30": np.log(rv_day.rolling(30, min_periods=30).mean()),
             "parkinson_7": 0.5 * np.log((hl ** 2).rolling(7, min_periods=7).mean() / (4 * np.log(2))),
             "garman_klass_7": 0.5 * np.log((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2)
                                            .rolling(7, min_periods=7).mean().clip(lower=1e-12)),
             "ewma_vol": 0.5 * np.log((rd ** 2).ewm(alpha=0.06, adjust=False, min_periods=30).mean()),
             "vol_of_rv_30": np.log(rv_day).rolling(30, min_periods=30).std()}
    out = {k: _last(v, times, 2 * DAY) for k, v in daily.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["rv_intraday_1d"] = _last(np.log(rv24), times, 2 * H)
        out["jump_share_7"] = _last((1 - bv / rv168).clip(lower=0.0), times, 2 * H)
        out["downside_share_7"] = _last(down / rv168, times, 2 * H)
    return out


def _liquidity(inputs: dict, times: np.ndarray) -> dict:
    fred = inputs.get("fred", {})
    weekly, daily_age = 10 * DAY, 6 * DAY
    hy = fred.get("BAMLH0A0HYM2")
    return {"walcl_chg_z": _change_z(fred.get("WALCL"), times, weekly),
            "tga_chg_z": _change_z(fred.get("WTREGEN"), times, weekly),
            "rrp_chg_z": _change_z(fred.get("RRPONTSYD"), times, daily_age),
            "hy_spread": _last(hy, times, daily_age),
            "hy_spread_chg_28": _last(hy, times, daily_age) - _last(hy, times - 28 * DAY, daily_age),
            "us2y_chg_30": _last(fred.get("DGS2"), times, daily_age) - _last(fred.get("DGS2"), times - 30 * DAY,
                                                                               daily_age),
            "curve_10y2y": _last(fred.get("T10Y2Y"), times, daily_age),
            "oil_ret_28": _log_ratio(_last(fred.get("DCOILWTICO"), times, 14 * DAY),
                                     _last(fred.get("DCOILWTICO"), times - 28 * DAY, 14 * DAY)),
            "usdjpy_ret_28": _log_ratio(_last(fred.get("DEXJPUS"), times, 14 * DAY),
                                        _last(fred.get("DEXJPUS"), times - 28 * DAY, 14 * DAY))}


def _basis_and_premiums(inputs: dict, s: str, times: np.ndarray) -> dict:
    nan = np.full(len(times), np.nan)
    out = {k: nan.copy() for k in ("qbasis_annual", "qbasis_chg_7", "cb_premium_1d", "cb_premium_7d",
                                   "kimchi_dev_30", "kimchi_chg_7")}
    perp = inputs["hourly"][s].set_index("available_at")["close"]
    q = inputs.get("quarterly", {}).get(s)
    if q is not None and len(q):
        both = q.join(perp.rename("perp"), how="inner")
        left = (both["expiry"] - both.index.to_numpy(dtype=np.int64)) / DAY
        basis = (both["close"] / both["perp"] - 1) * 365 / left
        out["qbasis_annual"] = _window_mean(basis, times, DAY, 20)
        out["qbasis_chg_7"] = out["qbasis_annual"] - _window_mean(basis, times - 7 * DAY, DAY, 20)
    spot = inputs.get("spot", {}).get(s)
    cb = inputs.get("coinbase", {}).get(s)
    if cb is not None and spot is not None and len(cb) and len(spot):
        frame = pd.DataFrame({"cb": cb}).join(spot["close"].rename("spot"), how="inner")
        usdt = inputs.get("usdt_usd")
        rate = pd.Series(_last(usdt, frame.index.to_numpy(np.int64), 2 * H), index=frame.index).fillna(1.0) \
            if usdt is not None else pd.Series(1.0, index=frame.index)
        premium = frame["cb"] / (frame["spot"] * rate) - 1
        out["cb_premium_1d"] = _window_mean(premium, times, DAY, 20)
        out["cb_premium_7d"] = _window_mean(premium, times, 7 * DAY, 140)
    up = inputs.get("upbit", {}).get(s)
    if up is not None and spot is not None and len(up) and len(spot):
        frame = pd.DataFrame({"krw": up}).join(spot["close"].rename("spot"), how="inner")
        ratio = np.log(frame["krw"] / frame["spot"])
        d = _on_days(ratio, times, 2 * H)
        d = pd.Series(_window_mean(ratio, d.index.to_numpy(np.int64), DAY, 20), index=d.index)
        out["kimchi_dev_30"] = _last(d - d.rolling(30, min_periods=30).mean(), times, DAY)
        out["kimchi_chg_7"] = _last(d - d.shift(7), times, DAY)
    return out


def _sentiment_calendar_chain(inputs: dict, s: str, times: np.ndarray) -> dict:
    fng = inputs.get("fng")
    out = {"fng": _last(fng, times, 3 * DAY)}
    out["fng_chg_7"] = out["fng"] - _last(fng, times - 7 * DAY, 3 * DAY)
    stamp = pd.to_datetime(times, unit="ms", utc=True)
    out["dow"] = stamp.dayofweek.to_numpy(dtype=float)

    def next_expiry(quarterly: bool) -> np.ndarray:
        result = np.empty(len(times))
        for k, t in enumerate(stamp):
            year, month = t.year, t.month
            while True:
                if not quarterly or month in (3, 6, 9, 12):
                    last = pd.Timestamp(year + (month == 12), month % 12 + 1, 1, tz="UTC") - pd.Timedelta(days=1)
                    expiry = last - pd.Timedelta(days=(last.dayofweek - 4) % 7) + pd.Timedelta(hours=8)
                    if expiry > t:
                        result[k] = (expiry - t) / pd.Timedelta(days=1)
                        break
                year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return result
    out["days_to_monthly_expiry"], out["days_to_quarterly_expiry"] = next_expiry(False), next_expiry(True)
    fomc = np.sort(np.asarray(inputs.get("fomc", []), dtype=np.int64))
    to, since = np.full(len(times), np.nan), np.full(len(times), np.nan)
    if len(fomc):
        statement_to, statement_since = fomc + 18 * H, fomc + 20 * H      # about 14:00 US Eastern
        i = np.searchsorted(statement_to, times, side="right")
        ok = i < len(fomc)
        to[ok] = (statement_to[i[ok]] - times[ok]) / DAY
        j = np.searchsorted(statement_since, times, side="right") - 1
        ok = j >= 0
        since[ok] = (times[ok] - statement_since[j[ok]]) / DAY
    out["days_to_fomc"], out["days_since_fomc"] = to, since
    cm = inputs["cm"].get(s)
    mvrv = cm["CapMVRVCur"] if cm is not None and "CapMVRVCur" in cm else None
    hashrate = cm["HashRate"].rolling(7, min_periods=7).mean() if cm is not None and "HashRate" in cm else None
    out["mvrv"] = _last(mvrv, times, 3 * DAY)
    out["mvrv_chg_30"] = _log_ratio(out["mvrv"], _last(mvrv, times - 30 * DAY, 3 * DAY))
    out["hashrate_chg_30"] = _log_ratio(_last(hashrate, times, 3 * DAY), _last(hashrate, times - 30 * DAY, 3 * DAY))
    return out


def build(inputs: dict, asset: str, times: np.ndarray, version: int = 1) -> pd.DataFrame:
    """Every feature of ``FEATURES`` (``FEATURES_V2`` with ``version=2``) for ``asset`` at each decision time
    (index), from values known by then."""
    times = np.asarray(times, dtype=np.int64)
    other = SYMBOLS[1] if asset == SYMBOLS[0] else SYMBOLS[0]
    d, o = daily(inputs["hourly"][asset]), daily(inputs["hourly"][other])
    out = _price(d, inputs["hourly"][asset], times)
    out.update(_derivatives(inputs, asset, d, times, out["ret_7"]))
    out.update(_spot(inputs, asset, times))
    out.update(_depth(inputs, asset, d.index.to_numpy(np.int64), times))
    out.update(_implied(inputs, asset, times, out["rv_30"]))
    out.update(_other(d, o, times, out))
    breadth = inputs.get("breadth")
    out.update({k: _last(breadth[k] if breadth is not None else None, times, 2 * DAY) for k in BREADTH})
    out.update(_macro(inputs, times))
    out.update(_chain(inputs, asset, times))
    out.update(_technical(inputs["hourly"][asset], times))
    names = FEATURES
    if version >= 2:
        out.update(_realized(inputs["hourly"][asset], times))
        out.update(_liquidity(inputs, times))
        out.update(_basis_and_premiums(inputs, asset, times))
        out.update(_sentiment_calendar_chain(inputs, asset, times))
        names = FEATURES_V2
    return pd.DataFrame({k: np.asarray(out[k], dtype=float) for k in names}, index=pd.Index(times))


def load_market(snapshot: Path, limit: int, *, exclude: frozenset[str] = frozenset()) -> tuple[pd.DataFrame,
                                                                                                    pd.DataFrame]:
    """Daily closes and 3-day mean funding of every contract in a whole-market snapshot (``exclude`` left out), each
    file read and reduced at once; rows known after ``limit`` dropped. Index: the day's end."""
    closes, funding = {}, {}
    for path in sorted(Path(snapshot).glob("*_1h.csv.gz")):
        symbol = path.name[:-len("_1h.csv.gz")]
        if symbol in exclude:
            continue
        h = pd.read_csv(path, usecols=["open_time", "close"])
        h = h[h["open_time"] + H <= limit]
        if h.empty:
            continue
        day = h["open_time"] // DAY
        g = h.groupby(day)["close"]
        full = g.count() == 24
        c = g.last()[full]
        closes[symbol] = pd.Series(c.to_numpy(float), index=(c.index.to_numpy(dtype=np.int64) + 1) * DAY)
        fpath = path.with_name(f"{symbol}_funding.csv.gz")
        if fpath.exists():
            f = pd.read_csv(fpath, usecols=["funding_time", "rate"])
            f = f[f["funding_time"] <= limit]
            rates = pd.Series(f["rate"].to_numpy(float), index=f["funding_time"].to_numpy(dtype=np.int64))
            f3 = rates.sort_index().rolling(9, min_periods=9).mean()
            funding[symbol] = pd.Series(_last(f3, closes[symbol].index.to_numpy(np.int64), 9 * H),
                                        index=closes[symbol].index)
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(funding).sort_index()


def market_breadth(closes: pd.DataFrame, funding_3d: pd.DataFrame, btc: str = "BTCUSDT") -> pd.DataFrame:
    """The whole market each day (index: the day's end, when it is known): the share of contracts above their
    30-day mean, the cross-sectional spread of 7-day returns, the share with positive 3-day funding and the median
    30-day return of the others less BTC's. ``closes`` and ``funding_3d`` are daily panels on the same index."""
    above = (closes > closes.rolling(30, min_periods=30).mean()).where(closes.notna())
    ret7, ret30 = np.log(closes / closes.shift(7)), np.log(closes / closes.shift(30))
    others = ret30.drop(columns=[btc], errors="ignore")
    btc30 = ret30[btc] if btc in ret30 else pd.Series(np.nan, index=closes.index)
    return pd.DataFrame({"breadth_ma30": above.mean(axis=1), "dispersion_7": ret7.std(axis=1),
                         "funding_pos_share": (funding_3d > 0).where(funding_3d.notna()).mean(axis=1),
                         "alt_minus_btc_30": others.median(axis=1) - btc30}, index=closes.index)
