"""Signals of the three forward-test candidates (pre-registered 2026-11-14).

Protocol: research/protocols/2026-11-14-forward-candidates.md. Everything here is a pure function of
inputs that are known at the decision time, so the development run and a forward look call the same code.

- A ``pair_ridge_pd_7d``: the frozen form of the pair study's selected candidate (features in lab/pair.py).
- B ``dir_ridge_7d``: one pooled Ridge on each symbol's own price and derivatives features, 7-day horizon.
- C ``flush_reversal_72h``: long for 72 hours after a 24h price fall and a 24h open-interest fall that are
  both in the lowest 5% of their own previous 365 days. No fitted parameter.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lab.data import DAY, HOUR, aggregate
from lab.pair import METRIC_LAG, _funding_mean, _last_value, _log_positive, _window_mean, _zscore

HORIZON_DAYS = 7
TARGET_VOL, MAX_WEIGHT, VOL_LOOKBACK = 0.30, 0.9, 30          # the lab BASE sizing
B_PRICE = ("mom_1", "mom_3", "mom_7", "mom_14", "mom_30", "dev_30", "vol_ratio")
B_DERIVATIVES = ("funding_3d", "funding_z", "premium_1d", "premium_7d", "oi_chg_1d", "oi_chg_7d", "taker_1d",
                 "global_ls_z")
B_FEATURES = B_PRICE + B_DERIVATIVES
FLUSH_QUANTILE, FLUSH_WINDOW, FLUSH_MIN_HISTORY, FLUSH_HOLD = 0.05, 365 * 24, 180 * 24, 72 * HOUR


# -- shared ------------------------------------------------------------------------------------------
def daily_close(hourly: pd.DataFrame) -> pd.Series:
    """Daily closes indexed by the time they are known (UTC midnight, ms)."""
    daily = aggregate(hourly, 24)
    return pd.Series(daily["close"].to_numpy(float), index=daily["available_at"].to_numpy(dtype=np.int64))


def daily_vol(close: pd.Series) -> pd.Series:
    """Standard deviation of the last 30 daily log returns."""
    return np.log(close).diff().rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std()


def sized_weight(vol_daily: float) -> float:
    """Volatility-targeted weight magnitude; NaN when the volatility is unknown."""
    if not math.isfinite(vol_daily) or vol_daily <= 0:
        return float("nan")
    return min(MAX_WEIGHT, TARGET_VOL / (vol_daily * math.sqrt(365.0)))


def fit_ridge(x: np.ndarray, y: np.ndarray) -> dict:
    """The pair study's fit (lab.pair_study.fit_predict), returned as frozen numbers."""
    from sklearn.linear_model import Ridge
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    model = Ridge(alpha=float(len(x))).fit(np.clip((x - mu) / sd, -5, 5), np.clip(y, -5, 5))
    return {"mean": mu.tolist(), "std": sd.tolist(), "coef": model.coef_.tolist(),
            "intercept": float(model.intercept_), "rows": int(len(x)), "alpha": float(len(x))}


def predict_frozen(model: dict, x: np.ndarray) -> np.ndarray:
    """Predictions of a frozen Ridge; rows with a missing feature give NaN."""
    z = np.clip((x - np.asarray(model["mean"])) / np.asarray(model["std"]), -5, 5)
    out = z @ np.asarray(model["coef"]) + model["intercept"]
    return np.where(np.isfinite(x).all(axis=1), out, np.nan)


# -- B: per-symbol features and weekly targets -----------------------------------------------------------
def symbol_features(inputs: dict, symbol: str) -> pd.DataFrame:
    """B_FEATURES plus ``sigma`` (30-day daily volatility) for one symbol, indexed by decision time."""
    close = daily_close(inputs["hourly"][symbol])
    times = close.index.to_numpy(dtype=np.int64)
    logp = np.log(close)
    sigma = daily_vol(close)
    out = pd.DataFrame(index=close.index)
    for lb in (1, 3, 7, 14, 30):
        out[f"mom_{lb}"] = (logp - logp.shift(lb)) / (sigma * math.sqrt(lb))
    out["dev_30"] = (logp - logp.rolling(30, min_periods=30).mean()) / logp.rolling(30, min_periods=30).std()
    out["vol_ratio"] = logp.diff().rolling(7, min_periods=7).std() / sigma
    m = inputs["metrics"][symbol]
    stamp = m["timestamp"].to_numpy(dtype=np.int64) + METRIC_LAG
    oi = pd.Series(pd.to_numeric(m["sum_open_interest"], errors="coerce").to_numpy(float), index=stamp)
    taker = pd.Series(_log_positive(m["sum_taker_long_short_vol_ratio"]).to_numpy(), index=stamp)
    acct = pd.Series(_log_positive(m["count_long_short_ratio"]).to_numpy(), index=stamp)
    oi_now = _last_value(oi, times, HOUR)
    with np.errstate(invalid="ignore", divide="ignore"):
        oi_1d = np.log(oi_now / _last_value(oi, times - DAY, HOUR))
        oi_7d = np.log(oi_now / _last_value(oi, times - 7 * DAY, HOUR))
    funding = pd.Series(_funding_mean(inputs["funding"][symbol], times) * 1e4, index=out.index)
    out["funding_3d"] = funding
    out["funding_z"] = _zscore(funding, 90)
    out["premium_1d"] = _window_mean(inputs["premium"][symbol], times, DAY, 20) * 1e4
    out["premium_7d"] = _window_mean(inputs["premium"][symbol], times, 7 * DAY, 140) * 1e4
    out["oi_chg_1d"] = np.where(np.isfinite(oi_1d), oi_1d, np.nan)
    out["oi_chg_7d"] = np.where(np.isfinite(oi_7d), oi_7d, np.nan)
    out["taker_1d"] = _window_mean(taker, times, DAY, 200)
    out["global_ls_z"] = _zscore(pd.Series(_last_value(acct, times, HOUR), index=out.index), 30)
    out["sigma"] = sigma
    return out


def directional_labels(close: pd.Series, horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    """Return over the next ``horizon_days`` and the time the label ends."""
    times = close.index.to_numpy(dtype=np.int64)
    future = times + horizon_days * DAY
    return pd.DataFrame({"y": close.reindex(future).to_numpy() / close.to_numpy() - 1.0, "label_end": future},
                        index=close.index)


def weekly_targets(hourly: pd.DataFrame, prediction: pd.Series, sigma: pd.Series, *, start: int) -> pd.DataFrame:
    """Target weight at every hourly close: sign(prediction) x sized weight, decided every 7 days from ``start``.

    A decision time without a prediction or a volatility keeps the previous target. ``ready`` is False
    until the first decision that sets one.
    """
    times = hourly["available_at"].to_numpy(dtype=np.int64)
    target, ready = np.zeros(len(times)), np.zeros(len(times), dtype=bool)
    current, live = 0.0, False
    for i, t in enumerate(times.tolist()):
        if t >= start and (t - start) % (HORIZON_DAYS * DAY) == 0 and t in prediction.index:
            p, w = float(prediction.loc[t]), sized_weight(float(sigma.get(t, float("nan"))))
            if math.isfinite(p) and math.isfinite(w):
                current, live = float(np.sign(p)) * w, True
        target[i], ready[i] = current, live
    return pd.DataFrame({"available_at": times, "target": target, "ready": ready})


# -- C: liquidation-flush reversal ------------------------------------------------------------------
def flush_signal(hourly: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Per hourly close: 24h return, 24h open-interest change, their trailing thresholds and the event flag.

    A threshold is the 5% quantile of the previous 8760 hourly values (the current one excluded), and needs
    at least 4320 of them.
    """
    times = hourly["available_at"].to_numpy(dtype=np.int64)
    close = pd.Series(hourly["close"].to_numpy(float), index=times)
    r24 = pd.Series(close.to_numpy() / close.reindex(times - DAY).to_numpy() - 1.0, index=times)
    stamp = metrics["timestamp"].to_numpy(dtype=np.int64) + METRIC_LAG
    oi = pd.Series(pd.to_numeric(metrics["sum_open_interest"], errors="coerce").to_numpy(float), index=stamp)
    with np.errstate(invalid="ignore", divide="ignore"):
        change = np.log(_last_value(oi, times, HOUR) / _last_value(oi, times - DAY, HOUR))
    oi24 = pd.Series(np.where(np.isfinite(change), change, np.nan), index=times)

    def threshold(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(FLUSH_WINDOW, min_periods=FLUSH_MIN_HISTORY).quantile(FLUSH_QUANTILE)

    q_r, q_oi = threshold(r24), threshold(oi24)
    event = (r24 <= q_r) & (oi24 <= q_oi)
    return pd.DataFrame({"available_at": times, "r24": r24.to_numpy(), "oi24": oi24.to_numpy(),
                         "q_r": q_r.to_numpy(), "q_oi": q_oi.to_numpy(), "event": event.to_numpy(dtype=bool),
                         "armed": (q_r.notna() & q_oi.notna()).to_numpy()})


def flush_targets(hourly: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Long for 72 hours after each event; a new event during a hold restarts the clock, not the size."""
    signal = flush_signal(hourly, metrics)
    close = daily_close(hourly)
    sigma = daily_vol(close)
    day_times, day_sigma = sigma.index.to_numpy(dtype=np.int64), sigma.to_numpy(float)
    times = signal["available_at"].to_numpy(dtype=np.int64)
    target, entries = np.zeros(len(times)), np.zeros(len(times), dtype=bool)
    weight, until = 0.0, -1
    for i, (t, event) in enumerate(zip(times.tolist(), signal["event"].tolist())):
        if t >= until:
            weight = 0.0
        if event:
            if weight == 0.0:
                k = int(np.searchsorted(day_times, t, side="right")) - 1
                w = sized_weight(day_sigma[k]) if k >= 0 else float("nan")
                if math.isfinite(w):
                    weight, until, entries[i] = w, t + FLUSH_HOLD, True
            else:
                until = t + FLUSH_HOLD
        target[i] = weight
    return signal.assign(target=target, ready=signal["armed"].to_numpy(), entry=entries)


__all__ = ["B_FEATURES", "HORIZON_DAYS", "daily_close", "daily_vol", "sized_weight", "fit_ridge", "predict_frozen",
           "symbol_features", "directional_labels", "weekly_targets", "flush_signal", "flush_targets"]
