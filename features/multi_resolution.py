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
from typing import List

import numpy as np
import pandas as pd

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features

logger = logging.getLogger(__name__)

from _quant_hotpath import cpp_compute_fast_1m_features, cpp_fast_1m_feature_names

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
    """Compute short-window features directly on 1m bars via Rust."""
    return _compute_fast_features_cpp(df_1m)


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
