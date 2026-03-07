"""Multi-resolution feature engine for 1-minute alpha models.

Combines fast features (computed directly on 1m bars) with slow features
(computed on resampled 1h bars) and 4h features. The slow/4h features are
forward-filled back to the 1m index so every 1m row has the full feature set.

Usage:
    df_1m = pd.read_csv("data_files/BTCUSDT_1m.csv")
    feat_df = compute_multi_resolution_features(df_1m, "BTCUSDT")
"""
from __future__ import annotations

import logging
from pathlib import Path as _Path
from typing import List, Optional

import numpy as np
import pandas as pd

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES

logger = logging.getLogger(__name__)

try:
    from _quant_hotpath import cpp_compute_fast_1m_features, cpp_fast_1m_feature_names
    _USING_CPP_1M = True
except ImportError:
    _USING_CPP_1M = False
    logger.warning("C++ fast 1m features not available, using Python fallback")

# Fast features computed directly on 1m bars (short windows)
FAST_FEATURE_NAMES = (
    "ret_1", "ret_3", "ret_5", "ret_10",
    "rsi_6",
    "vol_5", "vol_20",
    "taker_imbalance",
    "trade_intensity",
    "cvd_10",
    "body_ratio", "upper_shadow", "lower_shadow",
    "vol_ratio_20",
    "aggressive_flow_zscore",
)


def resample_to_hourly(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m OHLCV bars to 1h bars using standard aggregation.

    Groups by 60-bar blocks aligned to the timestamp floor (epoch ms // 3600000).
    """
    ts_col = "open_time" if "open_time" in df_1m.columns else "timestamp"
    ts = df_1m[ts_col].values.astype(np.int64)
    hour_ms = 3_600_000
    group_keys = ts // hour_ms

    work = pd.DataFrame({
        "group": group_keys,
        "open_time": ts,
        "open": df_1m["open"].values.astype(np.float64),
        "high": df_1m["high"].values.astype(np.float64),
        "low": df_1m["low"].values.astype(np.float64),
        "close": df_1m["close"].values.astype(np.float64),
        "volume": df_1m["volume"].values.astype(np.float64),
    })

    for col in ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"):
        if col in df_1m.columns:
            work[col] = df_1m[col].values.astype(np.float64)
        else:
            work[col] = 0.0

    agg_spec = {
        "open_time": ("open_time", "first"),
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
        "volume": ("volume", "sum"),
        "quote_volume": ("quote_volume", "sum"),
        "trades": ("trades", "sum"),
        "taker_buy_volume": ("taker_buy_volume", "sum"),
        "taker_buy_quote_volume": ("taker_buy_quote_volume", "sum"),
    }

    agg = work.groupby("group", sort=True).agg(**agg_spec)
    agg = agg.reset_index(drop=True)
    return agg


def _compute_fast_features_cpp(df_1m: pd.DataFrame) -> pd.DataFrame:
    """C++ accelerated fast feature computation — single pass over all bars."""
    n = len(df_1m)
    opens = df_1m["open"].values.astype(np.float64) if "open" in df_1m.columns else df_1m["close"].values.astype(np.float64)
    highs = df_1m["high"].values.astype(np.float64) if "high" in df_1m.columns else df_1m["close"].values.astype(np.float64)
    lows = df_1m["low"].values.astype(np.float64) if "low" in df_1m.columns else df_1m["close"].values.astype(np.float64)
    closes = df_1m["close"].values.astype(np.float64)
    volumes = df_1m["volume"].values.astype(np.float64) if "volume" in df_1m.columns else np.zeros(n)
    trades_arr = df_1m["trades"].values.astype(np.float64) if "trades" in df_1m.columns else np.zeros(n)
    tbv = df_1m["taker_buy_volume"].values.astype(np.float64) if "taker_buy_volume" in df_1m.columns else np.zeros(n)

    result_np = cpp_compute_fast_1m_features(opens, highs, lows, closes, volumes, trades_arr, tbv)
    names = cpp_fast_1m_feature_names()
    return pd.DataFrame(result_np, columns=names, index=df_1m.index)


def _compute_fast_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute short-window features directly on 1m bars.

    Uses C++ when available (~10x faster), falls back to pandas vectorized.
    """
    if _USING_CPP_1M:
        return _compute_fast_features_cpp(df_1m)
    return _compute_fast_features_py(df_1m)


def _compute_fast_features_py(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute short-window features directly on 1m bars.

    All operations are vectorized via pandas/numpy for performance on >1M bars.
    """
    n = len(df_1m)
    close = pd.Series(df_1m["close"].values.astype(np.float64))
    open_ = pd.Series(df_1m["open"].values.astype(np.float64)) if "open" in df_1m.columns else close.copy()
    high = pd.Series(df_1m["high"].values.astype(np.float64)) if "high" in df_1m.columns else close.copy()
    low = pd.Series(df_1m["low"].values.astype(np.float64)) if "low" in df_1m.columns else close.copy()
    volume = pd.Series(df_1m["volume"].values.astype(np.float64)) if "volume" in df_1m.columns else pd.Series(np.zeros(n))
    trades = pd.Series(df_1m["trades"].values.astype(np.float64)) if "trades" in df_1m.columns else pd.Series(np.zeros(n))
    tbv = pd.Series(df_1m["taker_buy_volume"].values.astype(np.float64)) if "taker_buy_volume" in df_1m.columns else pd.Series(np.zeros(n))

    result = {}

    # Returns at multiple short horizons (vectorized)
    for h in (1, 3, 5, 10):
        result[f"ret_{h}"] = close.pct_change(h).values

    # RSI-6 via pandas EWM (vectorized)
    pct = close.pct_change()
    gains = pct.clip(lower=0.0)
    losses = (-pct).clip(lower=0.0)
    avg_gain = gains.ewm(span=6, adjust=False).mean()
    avg_loss = losses.ewm(span=6, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi.iloc[0] = np.nan
    result["rsi_6"] = rsi.values

    # Volatility via pandas rolling (vectorized)
    result["vol_5"] = pct.rolling(5, min_periods=3).std().values
    result["vol_20"] = pct.rolling(20, min_periods=10).std().values

    # Taker imbalance (vectorized)
    with np.errstate(invalid="ignore", divide="ignore"):
        taker_ratio = np.where(volume.values > 0, tbv.values / volume.values, 0.5)
    result["taker_imbalance"] = 2.0 * taker_ratio - 1.0

    # Trade intensity: trades / EMA(trades, 20) (vectorized)
    ema_trades = trades.ewm(span=20, adjust=False).mean()
    trade_int = trades / ema_trades.replace(0, np.nan)
    result["trade_intensity"] = trade_int.values

    # CVD-10: rolling sum of volume delta, normalized (vectorized)
    delta = tbv - (volume - tbv)  # buy - sell volume
    cvd_10_raw = delta.rolling(10, min_periods=10).sum()
    ema_vol = volume.ewm(span=20, adjust=False).mean()
    cvd_10 = cvd_10_raw / (ema_vol * 10).replace(0, np.nan)
    result["cvd_10"] = cvd_10.values

    # Candle structure (vectorized)
    body = (close - open_).abs()
    full_range = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        fr_vals = full_range.values
        body_ratio = np.where(fr_vals > 0, body.values / fr_vals, 0.0)
        upper_shadow = np.where(fr_vals > 0,
                                (high.values - np.maximum(close.values, open_.values)) / fr_vals, 0.0)
        lower_shadow = np.where(fr_vals > 0,
                                (np.minimum(close.values, open_.values) - low.values) / fr_vals, 0.0)
    result["body_ratio"] = body_ratio
    result["upper_shadow"] = upper_shadow
    result["lower_shadow"] = lower_shadow

    # Volume ratio: vol / EMA(vol, 20) (vectorized)
    vol_ratio = volume / ema_vol.replace(0, np.nan)
    result["vol_ratio_20"] = vol_ratio.values

    # Aggressive flow z-score: rolling z-score of taker_buy_ratio over 24 bars (vectorized)
    with np.errstate(invalid="ignore", divide="ignore"):
        tbr = pd.Series(np.where(volume.values > 0, tbv.values / volume.values, 0.5))
    tbr_mean = tbr.rolling(24, min_periods=24).mean()
    tbr_std = tbr.rolling(24, min_periods=24).std()
    agg_flow_z = (tbr - tbr_mean) / tbr_std.replace(0, np.nan)
    result["aggressive_flow_zscore"] = agg_flow_z.values

    return pd.DataFrame(result, index=df_1m.index)


# Slow features: computed on 1h resampled data, forward-filled to 1m
SLOW_FEATURE_NAMES = (
    "slow_rsi_14",
    "slow_atr_norm_14",
    "slow_close_vs_ma20",
    "slow_bb_width_20",
    "slow_basis",
    "slow_funding_zscore_24",
    "slow_fgi_normalized",
    "slow_vol_20",
    "slow_mean_reversion_20",
    "slow_ret_24",
)

# Mapping from slow feature name to source column in 1h feature DataFrame
_SLOW_SOURCE_MAP = {
    "slow_rsi_14": "rsi_14",
    "slow_atr_norm_14": "atr_norm_14",
    "slow_close_vs_ma20": "close_vs_ma20",
    "slow_bb_width_20": "bb_width_20",
    "slow_basis": "basis",
    "slow_funding_zscore_24": "funding_zscore_24",
    "slow_fgi_normalized": "fgi_normalized",
    "slow_vol_20": "vol_20",
    "slow_mean_reversion_20": "mean_reversion_20",
    "slow_ret_24": "ret_24",
}

# 4h features forwarded
SLOW_4H_FEATURE_NAMES = (
    "tf4h_close_vs_ma20",
    "tf4h_rsi_14",
    "tf4h_atr_norm_14",
    "tf4h_vol_20",
    "tf4h_mean_reversion_20",
)


def compute_multi_resolution_features(
    df_1m: pd.DataFrame,
    symbol: str,
    *,
    include_slow: bool = True,
    include_4h: bool = True,
) -> pd.DataFrame:
    """Compute multi-resolution features from 1m bars.

    Returns a DataFrame indexed like df_1m with:
      - Fast features (short-window, 1m bar resolution)
      - Slow features (1h resampled, forward-filled, slow_ prefix)
      - 4h features (4h resampled, forward-filled, tf4h_ prefix)
    """
    ts_col = "open_time" if "open_time" in df_1m.columns else "timestamp"
    ts = df_1m[ts_col].values.astype(np.int64)

    # 1. Fast features on 1m bars
    fast_df = _compute_fast_features(df_1m)

    if not include_slow and not include_4h:
        fast_df["close"] = df_1m["close"].values.astype(np.float64)
        return fast_df

    # 2. Resample to 1h and compute slow features
    if include_slow:
        df_1h = resample_to_hourly(df_1m)
        _has_v11 = _Path("data_files/macro_daily.csv").exists()
        feat_1h = compute_features_batch(symbol, df_1h, include_v11=_has_v11)

        # Map 1h features back to 1m index via forward-fill
        hour_ms = 3_600_000
        hour_keys_1m = ts // hour_ms
        hour_keys_1h = df_1h["open_time"].values.astype(np.int64) // hour_ms

        # Build lookup: hour_key -> index in feat_1h (use PREVIOUS completed hour)
        hour_to_idx = {}
        for i, hk in enumerate(hour_keys_1h):
            hour_to_idx[int(hk)] = i

        # Map each 1m bar to the PREVIOUS completed 1h bar (no look-ahead)
        row_map = np.full(len(ts), -1, dtype=np.int64)
        for i, hk in enumerate(hour_keys_1m):
            prev_hk = int(hk) - 1
            idx = hour_to_idx.get(prev_hk, -1)
            if idx >= 0:
                row_map[i] = idx
            elif i > 0 and row_map[i - 1] >= 0:
                row_map[i] = row_map[i - 1]  # forward-fill

        for slow_name, src_col in _SLOW_SOURCE_MAP.items():
            if src_col in feat_1h.columns:
                src_vals = feat_1h[src_col].values.astype(np.float64)
                mapped = np.where(row_map >= 0, src_vals[row_map], np.nan)
                fast_df[slow_name] = mapped
            else:
                fast_df[slow_name] = np.nan

    # 3. 4h features
    if include_4h and include_slow:
        tf4h = compute_4h_features(df_1h)

        # Map 4h features to 1m using same hour mapping mechanism
        for col in SLOW_4H_FEATURE_NAMES:
            if col in tf4h.columns:
                src_vals = tf4h[col].values.astype(np.float64)
                mapped = np.where(row_map >= 0, src_vals[row_map], np.nan)
                fast_df[col] = mapped
            else:
                fast_df[col] = np.nan

    fast_df["close"] = df_1m["close"].values.astype(np.float64)
    return fast_df


def get_all_feature_names() -> List[str]:
    """Return all multi-resolution feature names (fast + slow + 4h)."""
    return list(FAST_FEATURE_NAMES) + list(SLOW_FEATURE_NAMES) + list(SLOW_4H_FEATURE_NAMES)
