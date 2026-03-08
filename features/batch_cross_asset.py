# features/batch_cross_asset.py
"""Vectorized batch cross-asset feature computation.

Replaces the bar-by-bar loop in train_v7_alpha._build_cross_features() with
numpy/pandas vectorized operations. Produces identical 17 features.

Usage:
    from features.batch_cross_asset import build_cross_features_batch
    cross_map = build_cross_features_batch(["ETHUSDT", "SOLUSDT"])
    # Returns {symbol: DataFrame(index=timestamp, columns=17 features)}
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from features.cross_asset_computer import CROSS_ASSET_FEATURE_NAMES

logger = logging.getLogger(__name__)


# ── Schedule loaders (reused from batch_feature_engine) ───────

def _load_schedule(path: Path, ts_col: str, val_col: str) -> Dict[int, float]:
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row[ts_col])] = float(row[val_col])
    return schedule


def _forward_fill_schedule(
    bar_timestamps: np.ndarray,
    schedule: Dict[int, float],
) -> np.ndarray:
    """Vectorized forward-fill of schedule values to bar timestamps.

    Returns array of same length as bar_timestamps with forward-filled values.
    Bars before the first schedule entry get NaN.
    """
    n = len(bar_timestamps)
    if not schedule:
        return np.full(n, np.nan)

    sched_ts = np.array(sorted(schedule.keys()), dtype=np.int64)
    sched_vals = np.array([schedule[t] for t in sched_ts], dtype=np.float64)

    # searchsorted: find index of last schedule entry <= each bar timestamp
    idx = np.searchsorted(sched_ts, bar_timestamps, side="right") - 1
    result = np.full(n, np.nan)
    valid = idx >= 0
    result[valid] = sched_vals[idx[valid]]
    return result


# ── Vectorized indicator functions ────────────────────────────

def _rolling_ret(close: np.ndarray, lag: int) -> np.ndarray:
    """(close[t] - close[t-lag]) / close[t-lag], NaN for insufficient data."""
    n = len(close)
    ret = np.full(n, np.nan)
    if n <= lag:
        return ret
    ret[lag:] = (close[lag:] - close[:-lag]) / close[:-lag]
    return ret


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Wilder-style EMA: ema[0] = x[0], ema[t] = alpha*x[t] + (1-alpha)*ema[t-1]."""
    n = len(x)
    result = np.empty(n)
    result[0] = x[0]
    c = 1.0 - alpha
    for i in range(1, n):
        result[i] = alpha * x[i] + c * result[i - 1]
    return result


def _rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI with Wilder's smoothing. Returns NaN for first `period` bars."""
    n = len(close)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result

    delta = np.diff(close)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)

    alpha = 1.0 / period
    gain_ema = _ema(gains, alpha)
    loss_ema = _ema(losses, alpha)

    # RSI valid from index `period` onward (period deltas → period+1 closes)
    for i in range(period - 1, len(delta)):
        ge = gain_ema[i]
        le = loss_ema[i]
        if le < 1e-20:
            result[i + 1] = 100.0
        else:
            rs = ge / le
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return result


def _macd_line(close: np.ndarray) -> np.ndarray:
    """MACD line = EMA(12) - EMA(26). NaN for first 26 bars."""
    n = len(close)
    result = np.full(n, np.nan)
    if n < 27:
        return result

    alpha_12 = 2.0 / 13.0
    alpha_26 = 2.0 / 27.0
    ema12 = _ema(close, alpha_12)
    ema26 = _ema(close, alpha_26)

    result[26:] = ema12[26:] - ema26[26:]
    return result


def _sma(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average. NaN for first window-1 elements."""
    n = len(x)
    result = np.full(n, np.nan)
    if n < window:
        return result
    cumsum = np.cumsum(x)
    result[window - 1] = cumsum[window - 1] / window
    result[window:] = (cumsum[window:] - cumsum[:-window]) / window
    return result


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation (population). NaN for first window-1 elements."""
    n = len(x)
    result = np.full(n, np.nan)
    if n < window:
        return result
    # Use cumsum for mean, cumsum of squares for variance
    cumsum = np.cumsum(x)
    cumsum2 = np.cumsum(x ** 2)

    s = cumsum[window - 1:] - np.concatenate([[0], cumsum[:n - window]])
    s2 = cumsum2[window - 1:] - np.concatenate([[0], cumsum2[:n - window]])
    mean = s / window
    var = s2 / window - mean ** 2
    var = np.maximum(var, 0.0)  # numerical safety
    result[window - 1:] = np.sqrt(var)
    return result


def _mean_reversion(close: np.ndarray, window: int = 20) -> np.ndarray:
    """(close - SMA) / SMA. NaN during warmup."""
    sma = _sma(close, window)
    result = np.full(len(close), np.nan)
    valid = ~np.isnan(sma) & (np.abs(sma) > 1e-20)
    result[valid] = (close[valid] - sma[valid]) / sma[valid]
    return result


def _atr_norm(close: np.ndarray, high: np.ndarray, low: np.ndarray,
              period: int = 14) -> np.ndarray:
    """ATR(period) / close. NaN for first `period` bars."""
    n = len(close)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result

    # True range
    tr = np.empty(n - 1)
    for i in range(n - 1):
        hl = high[i + 1] - low[i + 1]
        hc = abs(high[i + 1] - close[i])
        lc = abs(low[i + 1] - close[i])
        tr[i] = max(hl, hc, lc)

    # SMA of true range
    atr = _sma(tr, period)
    for i in range(len(atr)):
        if not np.isnan(atr[i]) and close[i + 1] > 1e-20:
            result[i + 1] = atr[i] / close[i + 1]
    return result


def _bb_width(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Bollinger band width = (4 * std) / SMA. NaN during warmup."""
    sma = _sma(close, window)
    std = _rolling_std(close, window)
    result = np.full(len(close), np.nan)
    valid = ~np.isnan(sma) & ~np.isnan(std) & (np.abs(sma) > 1e-20)
    result[valid] = (4.0 * std[valid]) / sma[valid]
    return result


# ── Pair-wise features (vectorized rolling windows) ───────────

def _rolling_cov_var(
    x: np.ndarray, y: np.ndarray, window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling covariance(x, y) and variance(y) with population formula."""
    n = len(x)
    cov = np.full(n, np.nan)
    var_y = np.full(n, np.nan)
    if n < window:
        return cov, var_y

    # Use cumulative sums for O(n) computation
    cum_x = np.cumsum(x)
    cum_y = np.cumsum(y)
    cum_xy = np.cumsum(x * y)
    cum_y2 = np.cumsum(y * y)

    def _slice(cum, i):
        return cum[i] - (cum[i - window] if i >= window else 0.0)

    for i in range(window - 1, n):
        sx = _slice(cum_x, i)
        sy = _slice(cum_y, i)
        sxy = _slice(cum_xy, i)
        sy2 = _slice(cum_y2, i)
        mean_x = sx / window
        mean_y = sy / window
        cov[i] = sxy / window - mean_x * mean_y
        var_y[i] = sy2 / window - mean_y * mean_y

    return cov, var_y


def _rolling_beta(
    sym_ret: np.ndarray, bench_ret: np.ndarray, window: int,
) -> np.ndarray:
    """Rolling beta = cov(sym, bench) / var(bench). NaN during warmup."""
    cov, var_b = _rolling_cov_var(sym_ret, bench_ret, window)
    result = np.full(len(sym_ret), np.nan)
    valid = ~np.isnan(cov) & (np.abs(var_b) > 1e-20)
    result[valid] = cov[valid] / var_b[valid]
    return result


def _rolling_corr(
    x: np.ndarray, y: np.ndarray, window: int,
) -> np.ndarray:
    """Rolling Pearson correlation. NaN during warmup."""
    cov_xy, var_y = _rolling_cov_var(x, y, window)
    _, var_x = _rolling_cov_var(x, x, window)
    result = np.full(len(x), np.nan)
    denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
    valid = ~np.isnan(cov_xy) & (denom > 1e-20)
    result[valid] = cov_xy[valid] / denom[valid]
    return result


def _relative_strength(
    sym_ret: np.ndarray, bench_ret: np.ndarray, window: int = 20,
) -> np.ndarray:
    """Rolling cumulative return ratio: prod(1+sym) / prod(1+bench) over window."""
    n = len(sym_ret)
    result = np.full(n, np.nan)
    if n < window:
        return result

    log_sym = np.log1p(sym_ret)
    log_bench = np.log1p(bench_ret)
    cum_sym = np.cumsum(log_sym)
    cum_bench = np.cumsum(log_bench)

    for i in range(window - 1, n):
        start = i - window + 1
        s_sym = cum_sym[i] - (cum_sym[start - 1] if start > 0 else 0.0)
        s_bench = cum_bench[i] - (cum_bench[start - 1] if start > 0 else 0.0)
        result[i] = np.exp(s_sym - s_bench)

    return result


def _spread_zscore(
    sym_ret: np.ndarray, bench_ret: np.ndarray, beta: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Z-score of spread = sym_ret - beta * bench_ret over rolling window."""
    n = len(sym_ret)
    result = np.full(n, np.nan)

    spread = sym_ret - beta * bench_ret
    spread_mean = _sma(spread, window)
    spread_std = _rolling_std(spread, window)

    valid = (~np.isnan(spread_mean) & ~np.isnan(spread_std)
             & ~np.isnan(spread) & (spread_std > 1e-20))
    result[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]
    return result


# ── Main batch API ────────────────────────────────────────────

def _compute_btc_features(
    btc_close: np.ndarray,
    btc_high: np.ndarray,
    btc_low: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute 10 BTC-lead features for all bars (vectorized)."""
    return {
        "btc_ret_1": _rolling_ret(btc_close, 1),
        "btc_ret_3": _rolling_ret(btc_close, 3),
        "btc_ret_6": _rolling_ret(btc_close, 6),
        "btc_ret_12": _rolling_ret(btc_close, 12),
        "btc_ret_24": _rolling_ret(btc_close, 24),
        "btc_rsi_14": _rsi_wilder(btc_close, 14),
        "btc_macd_line": _macd_line(btc_close),
        "btc_mean_reversion_20": _mean_reversion(btc_close, 20),
        "btc_atr_norm_14": _atr_norm(btc_close, btc_high, btc_low, 14),
        "btc_bb_width_20": _bb_width(btc_close, 20),
    }


def _compute_pair_features(
    sym_close: np.ndarray,
    btc_close_aligned: np.ndarray,
    sym_funding: np.ndarray,
    btc_funding: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute 7 pair-wise features (vectorized)."""
    n = len(sym_close)

    # Returns (1-bar)
    sym_ret = np.full(n, np.nan)
    btc_ret = np.full(n, np.nan)
    sym_ret[1:] = (sym_close[1:] - sym_close[:-1]) / sym_close[:-1]
    btc_ret[1:] = (btc_close_aligned[1:] - btc_close_aligned[:-1]) / btc_close_aligned[:-1]

    # Replace NaN returns with 0 for rolling computations
    sym_ret_clean = np.nan_to_num(sym_ret, nan=0.0)
    btc_ret_clean = np.nan_to_num(btc_ret, nan=0.0)

    beta_30 = _rolling_beta(sym_ret_clean, btc_ret_clean, 30)
    beta_60 = _rolling_beta(sym_ret_clean, btc_ret_clean, 60)
    corr_30 = _rolling_corr(sym_ret_clean, btc_ret_clean, 30)
    rel_str = _relative_strength(sym_ret_clean, btc_ret_clean, 20)
    spread_z = _spread_zscore(sym_ret_clean, btc_ret_clean, beta_30, 20)

    # Funding difference
    funding_diff = sym_funding - btc_funding
    funding_diff_ma8 = np.full(n, np.nan)
    alpha = 2.0 / 9.0
    valid_start = None
    for i in range(n):
        if np.isnan(funding_diff[i]):
            continue
        if valid_start is None:
            valid_start = i
            funding_diff_ma8[i] = funding_diff[i]
        else:
            funding_diff_ma8[i] = alpha * funding_diff[i] + (1 - alpha) * funding_diff_ma8[i - 1]

    return {
        "rolling_beta_30": beta_30,
        "rolling_beta_60": beta_60,
        "relative_strength_20": rel_str,
        "rolling_corr_30": corr_30,
        "funding_diff": funding_diff,
        "funding_diff_ma8": funding_diff_ma8,
        "spread_zscore_20": spread_z,
    }


def compute_cross_features_for_symbol(
    symbol: str,
    sym_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    btc_funding_schedule: Dict[int, float],
    sym_funding_schedule: Dict[int, float],
) -> pd.DataFrame:
    """Compute all 17 cross-asset features for a single altcoin (vectorized).

    Args:
        symbol: Altcoin symbol (e.g. "ETHUSDT")
        sym_df: Altcoin OHLCV DataFrame with timestamp column
        btc_df: BTC OHLCV DataFrame with timestamp column
        btc_funding_schedule: {timestamp_ms: rate} for BTC
        sym_funding_schedule: {timestamp_ms: rate} for symbol

    Returns:
        DataFrame with 17 cross-asset feature columns, indexed by timestamp
    """
    sym_ts_col = "timestamp" if "timestamp" in sym_df.columns else "open_time"
    btc_ts_col = "timestamp" if "timestamp" in btc_df.columns else "open_time"

    sym_timestamps = sym_df[sym_ts_col].values.astype(np.int64)
    btc_timestamps = btc_df[btc_ts_col].values.astype(np.int64)
    n = len(sym_timestamps)

    # ── Align BTC bars to symbol timestamps ──
    # For each symbol bar, find the BTC bar with matching timestamp
    btc_ts_set = set(btc_timestamps.tolist())
    btc_ts_to_idx = {int(t): i for i, t in enumerate(btc_timestamps)}

    btc_close_all = btc_df["close"].values.astype(np.float64)
    btc_high_all = btc_df["high"].values.astype(np.float64) if "high" in btc_df.columns else btc_close_all.copy()
    btc_low_all = btc_df["low"].values.astype(np.float64) if "low" in btc_df.columns else btc_close_all.copy()

    # Build aligned BTC arrays (forward-fill where BTC bar exists)
    btc_close_aligned = np.full(n, np.nan)
    btc_high_aligned = np.full(n, np.nan)
    btc_low_aligned = np.full(n, np.nan)

    last_close = np.nan
    last_high = np.nan
    last_low = np.nan
    for i in range(n):
        ts = int(sym_timestamps[i])
        if ts in btc_ts_to_idx:
            bi = btc_ts_to_idx[ts]
            last_close = btc_close_all[bi]
            last_high = btc_high_all[bi]
            last_low = btc_low_all[bi]
        btc_close_aligned[i] = last_close
        btc_high_aligned[i] = last_high
        btc_low_aligned[i] = last_low

    # ── Forward-fill funding rates (vectorized) ──
    btc_fr = _forward_fill_schedule(sym_timestamps, btc_funding_schedule)
    sym_fr = _forward_fill_schedule(sym_timestamps, sym_funding_schedule)

    # ── Compute BTC-lead features on aligned BTC data ──
    # Only compute on bars where BTC data exists
    btc_feats = _compute_btc_features(btc_close_aligned, btc_high_aligned, btc_low_aligned)

    # ── Compute pair-wise features ──
    sym_close = sym_df["close"].values.astype(np.float64)
    pair_feats = _compute_pair_features(sym_close, btc_close_aligned, sym_fr, btc_fr)

    # ── Combine into DataFrame ──
    data = {}
    data.update(btc_feats)
    data.update(pair_feats)

    # Replace None-equivalent NaN during warmup (match Rust behavior)
    result = pd.DataFrame(data, index=sym_timestamps)
    return result


def build_cross_features_batch(
    symbols: List[str],
) -> Optional[Dict[str, pd.DataFrame]]:
    """Vectorized replacement for train_v7_alpha._build_cross_features().

    Computes 17 cross-asset features for each non-BTC symbol using
    numpy vectorized operations instead of bar-by-bar Rust FFI calls.

    Args:
        symbols: List of symbols (BTC is used as benchmark, skipped in output)

    Returns:
        {symbol: DataFrame(index=timestamp, columns=17 features)} or None
    """
    btc_path = Path("data_files/BTCUSDT_1h.csv")
    if not btc_path.exists():
        return None

    btc_df = pd.read_csv(btc_path)

    # Load BTC funding once
    btc_funding = _load_schedule(
        Path("data_files/BTCUSDT_funding.csv"), "timestamp", "funding_rate")

    result: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        if sym == "BTCUSDT":
            continue
        sym_path = Path(f"data_files/{sym}_1h.csv")
        if not sym_path.exists():
            continue

        sym_df = pd.read_csv(sym_path)
        sym_funding = _load_schedule(
            Path(f"data_files/{sym}_funding.csv"), "timestamp", "funding_rate")

        cross_df = compute_cross_features_for_symbol(
            sym, sym_df, btc_df, btc_funding, sym_funding)
        result[sym] = cross_df

    return result if result else None
