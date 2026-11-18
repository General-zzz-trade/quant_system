"""The research gates of 2026-11-18 (owner, after the retrospective docs/research/retrospective-2026-11-18.md), for
studies registered from then on:

- ``regression_alpha``: the strategy's daily returns regressed on the benchmarks' (buy-and-hold, vol-target hold) with
  an intercept; the intercept annualized (x 365, the lab's calendar days) must be above 0 with a Newey-West t >= 2.
- ``portfolio_gain``: the benchmark and the strategy at risk parity (inverse volatility, fixed over the sample) must
  have a higher Sharpe than the benchmark alone.
- ``family_deflated``: the deflated Sharpe probability with the trials of the hypothesis' family, at least 0.9.
- ``power``, ``samples_needed``, ``min_detectable``: a design's power (two-sided test of a mean, normal
  approximation); a plan below 80 % power is not run as a decision study.
- ``sequential``: Wald's sequential probability ratio test of a forward look's per-event (or per-period) results:
  no effect (mean 0) against the effect the plan expects, known spread; it stops at the first crossing.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from lab.metrics import deflated_sharpe_probability_k

YEAR = 365
DEFLATED_THRESHOLD = 0.9


def _newey_west(x: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    """The Newey-West (Bartlett) covariance of OLS coefficients."""
    n = len(e)
    xe = x * e[:, None]
    s = xe.T @ xe
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = xe[lag:].T @ xe[:-lag]
        s += w * (g + g.T)
    bread = np.linalg.inv(x.T @ x)
    return bread @ s @ bread * (n / max(n - x.shape[1], 1))


def regression_alpha(strategy: pd.Series, benchmarks: pd.DataFrame, *, lags: int | None = None) -> dict:
    data = pd.concat([strategy.rename("_s"), benchmarks], axis=1, join="inner").dropna()
    y = data["_s"].to_numpy(float)
    x = np.column_stack([np.ones(len(data)), data.drop(columns="_s").to_numpy(float)])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    e = y - x @ coef
    lags = int(4 * (len(y) / 100.0) ** (2.0 / 9.0)) if lags is None else lags
    se = math.sqrt(max(_newey_west(x, e, lags)[0, 0], 0.0))
    t = float(coef[0] / se) if se > 0 else 0.0
    alpha = float(coef[0] * YEAR)
    return {"alpha_annual": alpha, "t_newey_west": t, "lags": lags, "days": int(len(y)),
            "betas": {c: float(b) for c, b in zip(benchmarks.columns, coef[1:])}, "passes": alpha > 0 and t >= 2.0}


def _sharpe(r: pd.Series) -> float:
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * math.sqrt(YEAR)) if sd > 0 else 0.0


def portfolio_gain(strategy: pd.Series, benchmark: pd.Series) -> dict:
    data = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1, join="inner").dropna()
    inv = 1.0 / data.std(ddof=1)
    w = inv / inv.sum()
    mixed = data["s"] * w["s"] + data["b"] * w["b"]
    with_, alone = _sharpe(mixed), _sharpe(data["b"])
    return {"weights": {"strategy": float(w["s"]), "benchmark": float(w["b"])}, "sharpe_with": with_,
            "sharpe_benchmark": alone, "passes": with_ > alone}


def family_deflated(daily: pd.Series, family_sharpes: Sequence[float], *,
                    threshold: float = DEFLATED_THRESHOLD) -> dict:
    """``family_sharpes``: the annualized Sharpes of every configuration tried for this hypothesis on this data."""
    n = len(family_sharpes)
    std = float(np.std(family_sharpes, ddof=1)) if n >= 2 else 0.0
    p = deflated_sharpe_probability_k(daily, std, n)
    return {"n_trials": n, "sharpe_std": std, "probability": p, "threshold": threshold,
            "passes": bool(p == p and p >= threshold)}


def _z(alpha: float, power_: float) -> float:
    return float(norm.ppf(1 - alpha / 2) + norm.ppf(power_))


def power(effect: float, sd: float, n: int, *, alpha: float = 0.05) -> float:
    z = abs(effect) / (sd / math.sqrt(n))
    c = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(z - c) + norm.cdf(-z - c))


def samples_needed(effect: float, sd: float, *, power_: float = 0.8, alpha: float = 0.05) -> int:
    return int(math.ceil((_z(alpha, power_) * sd / abs(effect)) ** 2))


def min_detectable(sd: float, n: int, *, power_: float = 0.8, alpha: float = 0.05) -> float:
    return _z(alpha, power_) * sd / math.sqrt(n)


def sequential(results: Sequence[float], *, effect: float, sd: float, alpha: float = 0.05, beta: float = 0.2) -> dict:
    """Wald's SPRT: H0 mean 0, H1 mean ``effect``. Returns the decision ("effect", "no effect", "continue") and the
    observation (1-based) at which it was reached."""
    upper, lower = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    llr = 0.0
    for i, x in enumerate(results, start=1):
        llr += (effect * float(x) - effect * effect / 2.0) / (sd * sd)
        if llr >= upper:
            return {"decision": "effect", "at": i, "llr": llr}
        if llr <= lower:
            return {"decision": "no effect", "at": i, "llr": llr}
    return {"decision": "continue", "at": None, "llr": llr}


__all__ = ["DEFLATED_THRESHOLD", "family_deflated", "min_detectable", "portfolio_gain", "power", "regression_alpha",
           "samples_needed", "sequential"]
