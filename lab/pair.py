"""ETH vs BTC relative value: daily samples, features and labels (pre-registered 2026-11-14).

A sample is a UTC midnight ``t`` (the close of the daily bar). The traded
object is a dollar-neutral pair, +1 = long ETH / short BTC, whose return over
``h`` days is ``ETH(t+h)/ETH(t) - BTC(t+h)/BTC(t)``.

Feature families (every input is timestamped and used only when known at t):

- ``P`` price, from 1h klines (known at the bar close):
  pair log-ratio momentum over 1/3/7/14/30 days and its deviation from the
  30-day mean (in units of the pair's daily volatility), short/long pair
  volatility ratio, ETH/BTC volatility ratio, BTC momentum over 7/30 days,
  30-day ETH-BTC return correlation.
- ``D`` derivatives, as ETH minus BTC differences:
  funding (mean of the last 9 settlements, and its 90-day z-score), premium
  index (perp vs index; mean of the last 24 and 168 closed hours), open
  interest change over 1/7 days, taker buy/sell volume ratio (mean log over
  the last 24h), global account long/short ratio (log, 30-day z-score).
  Metrics rows are used only from ``timestamp <= t - 5 min``; a stale or
  sparse day gives NaN, never an older value. The top-trader position ratio
  is not used: Binance's archive leaves it empty for most of 2022, so it has
  almost no history before the first test block.

All raw inputs are cut at ``limit`` before anything is computed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from lab.data import DAY, HOUR, aggregate, frame_from_bars
from data.quality.causal_bars import load_bars
from runner.offline.report import load_funding

SYMBOLS = ("BTCUSDT", "ETHUSDT")
P_FEATURES = ("mom_1", "mom_3", "mom_7", "mom_14", "mom_30", "dev_30", "vol_ratio", "rel_vol",
              "btc_mom_7", "btc_mom_30", "corr_30")
D_FEATURES = ("funding_3d", "funding_z", "premium_1d", "premium_7d", "oi_chg_1d", "oi_chg_7d", "taker_1d",
              "global_ls_z")
METRIC_LAG = 5 * 60_000


def load_inputs(data_dir: Path, premium_dir: Path, limit: int) -> dict:
    """Bars, hourly frames, funding, premium and metrics for both symbols, cut at ``limit``."""
    out = {"bars": {}, "hourly": {}, "funding": {}, "premium": {}, "metrics": {}}
    for s in SYMBOLS:
        bars = [b for b in load_bars(Path(data_dir) / f"{s}_1h.csv") if b.open_time < limit]
        out["bars"][s] = bars
        out["hourly"][s] = frame_from_bars(bars)
        out["funding"][s] = {t: v for t, v in load_funding(Path(data_dir) / f"{s}_funding.csv").items() if t < limit}
        prem = pd.read_csv(Path(premium_dir) / f"{s}_premium_1h.csv")
        prem = prem[prem["open_time"] + HOUR <= limit]
        out["premium"][s] = pd.Series(prem["close"].to_numpy(float), index=prem["open_time"].to_numpy() + HOUR)
        metrics = pd.read_csv(Path(data_dir) / f"{s}_metrics.csv")
        out["metrics"][s] = metrics[metrics["timestamp"] + METRIC_LAG <= limit].reset_index(drop=True)
    return out


def daily_closes(hourly: dict) -> pd.DataFrame:
    """Daily closes of both symbols indexed by the decision time t (ms)."""
    cols = {}
    for s in SYMBOLS:
        d = aggregate(hourly[s], 24)
        cols[s] = pd.Series(d["close"].to_numpy(float), index=d["available_at"].to_numpy())
    return pd.DataFrame(cols).dropna()


def price_features(closes: pd.DataFrame) -> pd.DataFrame:
    eth, btc = np.log(closes["ETHUSDT"]), np.log(closes["BTCUSDT"])
    ratio = eth - btc
    d_ratio, d_eth, d_btc = ratio.diff(), eth.diff(), btc.diff()
    sigma = d_ratio.rolling(30, min_periods=30).std()
    sigma_btc = d_btc.rolling(30, min_periods=30).std()
    out = pd.DataFrame(index=closes.index)
    for lb in (1, 3, 7, 14, 30):
        out[f"mom_{lb}"] = (ratio - ratio.shift(lb)) / (sigma * math.sqrt(lb))
    out["dev_30"] = (ratio - ratio.rolling(30, min_periods=30).mean()) / ratio.rolling(30, min_periods=30).std()
    out["vol_ratio"] = d_ratio.rolling(7, min_periods=7).std() / sigma
    out["rel_vol"] = d_eth.rolling(30, min_periods=30).std() / sigma_btc
    for lb in (7, 30):
        out[f"btc_mom_{lb}"] = (btc - btc.shift(lb)) / (sigma_btc * math.sqrt(lb))
    out["corr_30"] = d_eth.rolling(30, min_periods=30).corr(d_btc)
    out["sigma_pair"] = sigma
    return out


def _funding_mean(funding: dict, times: np.ndarray, n: int = 9) -> np.ndarray:
    ft = np.array(sorted(funding), dtype=np.int64)
    rates = np.array([float(funding[t][0]) for t in ft])
    csum = np.concatenate([[0.0], np.cumsum(rates)])
    idx = np.searchsorted(ft, times, side="right")          # settlements at or before t
    out = np.full(len(times), np.nan)
    ok = idx >= n
    out[ok] = (csum[idx[ok]] - csum[idx[ok] - n]) / n
    fresh = np.zeros(len(times), dtype=bool)
    fresh[idx > 0] = times[idx > 0] - ft[idx[idx > 0] - 1] <= 8 * HOUR
    out[~fresh] = np.nan
    return out


def _window_mean(series: pd.Series, times: np.ndarray, span: int, min_count: int) -> np.ndarray:
    """Mean of values stamped in (t - span, t]; NaN with fewer than ``min_count`` values."""
    st = series.index.to_numpy(dtype=np.int64)
    v = series.to_numpy(float)
    csum, ccount = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))]), np.concatenate(
        [[0], np.cumsum(np.isfinite(v))])
    hi = np.searchsorted(st, times, side="right")
    lo = np.searchsorted(st, times - span, side="right")
    count = ccount[hi] - ccount[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (csum[hi] - csum[lo]) / count
    return np.where(count >= min_count, mean, np.nan)


def _last_value(series: pd.Series, times: np.ndarray, max_age: int) -> np.ndarray:
    st = series.index.to_numpy(dtype=np.int64)
    v = series.to_numpy(float)
    idx = np.searchsorted(st, times, side="right") - 1
    out = np.full(len(times), np.nan)
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    out[ok & (times - st[np.maximum(idx, 0)] > max_age)] = np.nan
    return out


def _log_positive(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series(np.where(v > 0, np.log(v), np.nan))


def _zscore(x: pd.Series, n: int) -> pd.Series:
    return (x - x.rolling(n, min_periods=n).mean()) / x.rolling(n, min_periods=n).std()


def derivative_features(inputs: dict, times: np.ndarray) -> pd.DataFrame:
    per = {}
    for s in SYMBOLS:
        m = inputs["metrics"][s]
        stamp = m["timestamp"].to_numpy(dtype=np.int64) + METRIC_LAG     # usable from here on
        oi = pd.Series(pd.to_numeric(m["sum_open_interest"], errors="coerce").to_numpy(float), index=stamp)
        taker = pd.Series(_log_positive(m["sum_taker_long_short_vol_ratio"]).to_numpy(), index=stamp)
        acct = pd.Series(_log_positive(m["count_long_short_ratio"]).to_numpy(), index=stamp)
        oi_now = _last_value(oi, times, HOUR)
        with np.errstate(invalid="ignore", divide="ignore"):
            oi_1d = np.log(oi_now / _last_value(oi, times - DAY, HOUR))
            oi_7d = np.log(oi_now / _last_value(oi, times - 7 * DAY, HOUR))
        per[s] = {
            "funding": _funding_mean(inputs["funding"][s], times),
            "premium_1d": _window_mean(inputs["premium"][s], times, DAY, 20),
            "premium_7d": _window_mean(inputs["premium"][s], times, 7 * DAY, 140),
            "oi_1d": np.where(np.isfinite(oi_1d), oi_1d, np.nan),
            "oi_7d": np.where(np.isfinite(oi_7d), oi_7d, np.nan),
            "taker": _window_mean(taker, times, DAY, 200),
            "acct": _last_value(acct, times, HOUR),
        }
    e, b = per["ETHUSDT"], per["BTCUSDT"]
    out = pd.DataFrame(index=pd.Index(times))
    funding = pd.Series((e["funding"] - b["funding"]) * 1e4, index=out.index)
    out["funding_3d"] = funding
    out["funding_z"] = _zscore(funding, 90)
    out["premium_1d"] = (e["premium_1d"] - b["premium_1d"]) * 1e4
    out["premium_7d"] = (e["premium_7d"] - b["premium_7d"]) * 1e4
    out["oi_chg_1d"] = e["oi_1d"] - b["oi_1d"]
    out["oi_chg_7d"] = e["oi_7d"] - b["oi_7d"]
    out["taker_1d"] = e["taker"] - b["taker"]
    out["global_ls_z"] = _zscore(pd.Series(e["acct"] - b["acct"], index=out.index), 30)
    return out


def labels(closes: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Pair return over the next ``horizon_days`` and the label end time."""
    eth, btc = closes["ETHUSDT"], closes["BTCUSDT"]
    times = closes.index.to_numpy(dtype=np.int64)
    future = pd.Series(times + horizon_days * DAY, index=closes.index)
    eth_f = eth.reindex(future.to_numpy()).to_numpy()
    btc_f = btc.reindex(future.to_numpy()).to_numpy()
    y = eth_f / eth.to_numpy() - btc_f / btc.to_numpy()
    return pd.DataFrame({"y": y, "label_end": future.to_numpy()}, index=closes.index)


def build(inputs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily closes, features: P + D + sigma_pair) indexed by decision time."""
    closes = daily_closes(inputs["hourly"])
    feats = price_features(closes).join(derivative_features(inputs, closes.index.to_numpy(dtype=np.int64)))
    return closes, feats


__all__ = ["SYMBOLS", "P_FEATURES", "D_FEATURES", "load_inputs", "build", "labels", "daily_closes"]
