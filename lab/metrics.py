"""Portfolio metrics from hourly sleeve equity curves (daily resolution)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

DAY = 86_400_000


def daily_equity(curves: list[pd.DataFrame]) -> pd.Series:
    """Sum sleeve equity per hour, then take the last value of each UTC day.

    A bar closing exactly at 00:00 closes the PREVIOUS day.
    """
    total = None
    for curve in curves:
        s = curve.set_index("at")["equity"]
        total = s if total is None else total.add(s, fill_value=np.nan)
    total = total.dropna()
    days = (total.index.to_numpy() - 1) // DAY
    return total.groupby(days).last()


def summarize(daily: pd.Series, *, start_day: int | None = None, end_day: int | None = None) -> dict:
    s = daily
    if start_day is not None:
        s = s[s.index >= start_day]
    if end_day is not None:
        s = s[s.index < end_day]
    if len(s) < 31:
        return {"days": int(len(s)), "sharpe": float("nan"), "cagr": float("nan"), "max_drawdown": float("nan"),
                "total_return": float("nan"), "vol": float("nan")}
    r = s.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else float("nan")
    total = float(s.iloc[-1] / s.iloc[0] - 1)
    years = len(r) / 365.0
    cagr = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    peak = s.cummax()
    mdd = float(((peak - s) / peak).max())
    return {"days": int(len(r)), "sharpe": sharpe, "cagr": cagr, "max_drawdown": mdd, "total_return": total,
            "vol": float(r.std() * math.sqrt(365))}


def yearly(daily: pd.Series) -> dict[int, float]:
    years = pd.to_datetime(daily.index.to_numpy() * DAY, unit="ms", utc=True).year
    out = {}
    for year in sorted(set(years)):
        s = daily[years == year]
        if len(s) > 1:
            out[int(year)] = float(s.iloc[-1] / s.iloc[0] - 1)
    return out


def turnover(fills: list[dict], days: int) -> float:
    """Fills per year (one round trip is roughly two fills)."""
    return len(fills) / max(days, 1) * 365.0


def exposure(curves: list[pd.DataFrame]) -> float:
    gross = sum(c["qty"].abs() * c["mark"] for c in curves)
    total = sum(c["equity"] for c in curves)
    return float((gross / total).mean())


def deflated_sharpe_probability(daily_returns: pd.Series, trial_sharpes: list[float]) -> float:
    """Probability that the selected strategy's true Sharpe > 0 after selecting
    the best of ``len(trial_sharpes)`` trials (Bailey & Lopez de Prado 2014).

    Works in per-period (daily) Sharpe units throughout.
    """
    r = daily_returns.dropna()
    n = len(r)
    if n < 30 or r.std() <= 0:
        return float("nan")
    sr = r.mean() / r.std()
    trials = np.asarray([t for t in trial_sharpes if math.isfinite(t)], dtype=float) / math.sqrt(365)
    if len(trials) < 2:
        sr0 = 0.0
    else:
        gamma = 0.5772156649
        k = len(trials)
        sr0 = trials.std(ddof=1) * ((1 - gamma) * norm.ppf(1 - 1 / k) + gamma * norm.ppf(1 - 1 / (k * math.e)))
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))


def deflated_sharpe_probability_k(daily_returns: pd.Series, sharpe_std_annual: float, n_trials: int) -> float:
    """Same test as ``deflated_sharpe_probability``, but from the number of trials and the
    standard deviation of their annualized Sharpes (for trial sets where only some Sharpes
    were recorded). ``n_trials < 2`` means no selection: the benchmark Sharpe is 0."""
    r = daily_returns.dropna()
    n = len(r)
    if n < 30 or r.std() <= 0 or not math.isfinite(sharpe_std_annual) or sharpe_std_annual < 0:
        return float("nan")
    sr = r.mean() / r.std()
    sr0 = 0.0
    if n_trials >= 2:
        gamma = 0.5772156649
        sr0 = sharpe_std_annual / math.sqrt(365) * ((1 - gamma) * norm.ppf(1 - 1 / n_trials)
                                                     + gamma * norm.ppf(1 - 1 / (n_trials * math.e)))
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))
