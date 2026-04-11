"""Canonical IC (Information Coefficient) computation.

Historical reason for this module: ``alpha/training/train_v12.py`` and
``monitoring/ic_decay_monitor.py`` each had their own tiny spearmanr
wrapper, with subtle differences in NaN handling and minimum-sample
thresholds.  This meant the ``avg_ic`` stored in ``config.json`` and
the ``live rolling IC`` reported by the monitor were NOT directly
comparable — a drift of a few percent was baked in by the code paths.

Both sides now import ``compute_ic`` from here so any future tweaks
propagate atomically.

Design:
  - ``compute_ic(preds, returns)`` — Spearman IC over full arrays,
    canonical NaN filtering (drop pairs where either is NaN/inf)
  - ``rolling_ic(preds, returns, window, min_samples=30)`` — last
    ``window`` samples, returns NaN when short
  - ``compute_ic_from_df(df, pred_col, ret_col)`` — DataFrame convenience
  - ``deflated_sharpe_ratio(sharpe, n_trials, skew, kurt, n_obs)`` —
    López de Prado DSR accounting for multiple-testing bias

Keep this module pure-Python/numpy/scipy so it can be loaded anywhere
(training subprocess, live monitor, pytest) without pulling in the
full Rust / ML model graph.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import spearmanr


def compute_ic(
    preds: np.ndarray,
    returns: np.ndarray,
    min_samples: int = 30,
) -> float:
    """Canonical Spearman IC between predictions and realized returns.

    Parameters
    ----------
    preds : np.ndarray
        Model predictions (any shape, will be ravelled).
    returns : np.ndarray
        Realized forward returns aligned with ``preds``.
    min_samples : int, default 30
        Return ``nan`` if fewer than this many valid pairs remain after
        NaN/inf filtering.  30 is a common statistical sanity floor for
        Spearman.

    Returns
    -------
    float
        IC in ``[-1, 1]`` or ``nan`` if insufficient data.  Always a
        Python float (never numpy scalar) so the value is directly JSON
        serializable.

    Notes
    -----
    Both arrays are ravelled so callers can pass 2-D (single column) or
    1-D without preprocessing.  NaN/inf pairs are dropped, not imputed,
    because imputation artificially inflates IC.
    """
    p = np.asarray(preds, dtype=np.float64).ravel()
    r = np.asarray(returns, dtype=np.float64).ravel()
    n = min(len(p), len(r))
    if n < min_samples:
        return float("nan")

    p = p[:n]
    r = r[:n]

    mask = np.isfinite(p) & np.isfinite(r)
    p = p[mask]
    r = r[mask]
    if len(p) < min_samples:
        return float("nan")

    # Spearman is rank-based so scale differences are fine.  Constant
    # columns (var ≈ 0) produce NaN; guard explicitly to avoid scipy
    # RuntimeWarnings in tests.
    if np.std(p) < 1e-12 or np.std(r) < 1e-12:
        return float("nan")

    ic, _ = spearmanr(p, r)
    if not np.isfinite(ic):
        return float("nan")
    return float(ic)


def rolling_ic(
    preds: np.ndarray,
    returns: np.ndarray,
    window: int,
    min_samples: int = 30,
) -> float:
    """IC over the last ``window`` samples.

    Used by live IC monitoring: 30d / 60d / 90d windows are the
    canonical buckets.  If the underlying arrays are shorter than
    ``window`` the full history is used (after the min_samples check
    inside compute_ic).
    """
    p = np.asarray(preds, dtype=np.float64).ravel()
    r = np.asarray(returns, dtype=np.float64).ravel()
    n = min(len(p), len(r))
    if n == 0:
        return float("nan")
    take = min(window, n)
    return compute_ic(p[-take:], r[-take:], min_samples=min_samples)


def compute_ic_from_df(
    df,
    pred_col: str,
    ret_col: str,
    min_samples: int = 30,
) -> float:
    """Convenience wrapper for pandas DataFrames.

    Extracts the two columns and delegates to ``compute_ic``.  Kept
    separate so the core function has zero pandas dependency.
    """
    return compute_ic(df[pred_col].values, df[ret_col].values, min_samples=min_samples)


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> Optional[float]:
    """Deflated Sharpe Ratio (López de Prado, 2014).

    Accounts for multiple-testing bias when the Sharpe reported is the
    maximum across many trial strategies.  A DSR > 0 means the observed
    Sharpe exceeds what you would expect from the best of ``n_trials``
    random strategies with the same variance.

    Parameters
    ----------
    sharpe : float
        Observed annualised Sharpe ratio.
    n_trials : int
        Number of independent trials from which ``sharpe`` was picked
        (the max).  For a grid of 20 deadzone × 5 min_hold combos this
        is 100.
    n_obs : int
        Number of return observations the Sharpe was computed on.
    skew, kurt : float
        Sample skewness and (normal) kurtosis of the return series
        that produced ``sharpe``.  Defaults assume Gaussian.

    Returns
    -------
    float or None
        DSR in standard-deviation units.  ``None`` if inputs are
        degenerate.

    Notes
    -----
    Reference: "The Deflated Sharpe Ratio: Correcting for Selection
    Bias, Backtest Overfitting, and Non-Normality" — Bailey, Lopez de
    Prado (2014).
    """
    if n_trials < 1 or n_obs < 2:
        return None
    if not math.isfinite(sharpe):
        return None

    # Expected max Sharpe of N iid random strategies under the null:
    #   E[max_N] ≈ (1 - γ) * Φ^-1(1 - 1/N) + γ * Φ^-1(1 - 1/(N*e))
    # where γ is Euler-Mascheroni ≈ 0.577.
    from scipy.stats import norm
    gamma = 0.5772156649
    if n_trials == 1:
        expected_max_sr = 0.0
    else:
        expected_max_sr = (
            (1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
            + gamma * norm.ppf(1 - 1.0 / (n_trials * math.e))
        )

    # SR variance under non-normality (Mertens, 2002):
    # Var(SR) ≈ (1 - skew*SR + ((kurt-1)/4)*SR^2) / (n_obs - 1)
    var_sr = (1 - skew * sharpe + ((kurt - 1) / 4.0) * sharpe ** 2) / max(n_obs - 1, 1)
    if var_sr <= 0:
        return None

    dsr = (sharpe - expected_max_sr) / math.sqrt(var_sr)
    return float(dsr)
