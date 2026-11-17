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
    out["fred"] = {}
    for series, lag in FRED_LAG_DAYS.items():
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


def build(inputs: dict, asset: str, times: np.ndarray) -> pd.DataFrame:
    """Every feature of ``FEATURES`` for ``asset`` at each decision time (index), from values known by then."""
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
    return pd.DataFrame({k: np.asarray(out[k], dtype=float) for k in FEATURES}, index=pd.Index(times))


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
