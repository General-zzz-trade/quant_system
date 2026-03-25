"""Batch feature computation — C++ accelerated with Python fallback.

Replaces the slow iterrows() loop in compute_oos_features() with a single
C++ call that processes all bars at once.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

from _quant_hotpath import cpp_compute_all_features, cpp_feature_names  # noqa: E402
from _quant_hotpath import cpp_compute_4h_features as _cpp_4h  # noqa: E402
from _quant_hotpath import (  # noqa: E402
    cpp_4h_feature_names,
    cpp_fast_1m_feature_names,
    cpp_compute_fast_1m_features,
    rust_extract_orderbook_features,
)

# Feature name lists from Rust for cross-validation with Python definitions
RUST_4H_FEATURE_NAMES: tuple[str, ...] = tuple(cpp_4h_feature_names())
RUST_1M_FEATURE_NAMES: tuple[str, ...] = tuple(cpp_fast_1m_feature_names())

# Rust orderbook feature extraction — used by tick-level collectors
extract_orderbook_features = rust_extract_orderbook_features

# Rust 1-minute feature computation — used by HFT signal path
compute_fast_1m_features = cpp_compute_fast_1m_features

TF4H_FEATURE_NAMES = (
    "tf4h_ret_1",            # 4h return (= 4h lookback)
    "tf4h_ret_3",            # 12h return
    "tf4h_ret_6",            # 24h return
    "tf4h_rsi_14",           # 56h RSI
    "tf4h_macd_hist",        # 4h MACD histogram
    "tf4h_bb_pctb_20",       # 4h Bollinger %B
    "tf4h_atr_norm_14",      # 4h normalized ATR
    "tf4h_vol_20",           # 4h realized volatility
    "tf4h_close_vs_ma20",    # close vs 80h MA
    "tf4h_mean_reversion_20",  # 4h mean-reversion z-score
)


def compute_4h_features(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLCV to 4h, compute 10 slow features, ffill back to 1h.

    Args:
        df_1h: DataFrame with columns: open_time (or timestamp), open, high, low, close, volume.

    Returns:
        DataFrame with same index as df_1h, containing TF4H_FEATURE_NAMES columns.
    """
    ts_col = "open_time" if "open_time" in df_1h.columns else "timestamp"
    ts = df_1h[ts_col].values.astype(np.int64)

    raw = _cpp_4h(
        np.ascontiguousarray(ts),
        np.ascontiguousarray(df_1h["open"].values, dtype=np.float64),
        np.ascontiguousarray(df_1h["high"].values, dtype=np.float64),
        np.ascontiguousarray(df_1h["low"].values, dtype=np.float64),
        np.ascontiguousarray(df_1h["close"].values, dtype=np.float64),
        np.ascontiguousarray(df_1h["volume"].values, dtype=np.float64),
    )
    result = pd.DataFrame(raw, columns=list(TF4H_FEATURE_NAMES), dtype=np.float64)
    result = result.ffill()
    return result


def _load_schedule(path: Path, ts_col: str, val_col: str) -> Dict[int, float]:
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row[ts_col])] = float(row[val_col])
    return schedule


def _load_spot_closes(symbol: str) -> Dict[int, float]:
    path = Path(f"data_files/{symbol}_spot_1h.csv")
    closes: Dict[int, float] = {}
    if not path.exists():
        return closes
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_col = "open_time" if "open_time" in row else "timestamp"
            closes[int(row[ts_col])] = float(row["close"])
    return closes


def _load_fgi_schedule() -> Dict[int, float]:
    path = Path("data_files/fear_greed_index.csv")
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row["timestamp"])] = float(row["value"])
    return schedule


def _parse_ts_ms(raw: str) -> int:
    """Parse timestamp string — supports both epoch ms and ISO 8601."""
    if "T" in raw or "-" in raw:
        from datetime import datetime
        dt = datetime.fromisoformat(raw)
        return int(dt.timestamp() * 1000)
    return int(raw)


def _load_iv_schedule(symbol: str) -> Dict[int, float]:
    path = Path(f"data_files/{symbol}_deribit_iv.csv")
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_col = "timestamp" if "timestamp" in row else "open_time"
            val_col = "implied_vol" if "implied_vol" in row else "mark_iv"
            if val_col not in row:
                continue
            schedule[_parse_ts_ms(row[ts_col])] = float(row[val_col])
    return schedule


def _load_pcr_schedule(symbol: str) -> Dict[int, float]:
    path = Path(f"data_files/{symbol}_deribit_pcr.csv")
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_col = "timestamp" if "timestamp" in row else "open_time"
            val_col = "put_call_ratio" if "put_call_ratio" in row else "pcr"
            if val_col not in row:
                continue
            schedule[_parse_ts_ms(row[ts_col])] = float(row[val_col])
    return schedule


def _dict_to_sorted_array(d: Dict[int, float]) -> np.ndarray:
    """Convert {timestamp: value} dict to sorted (M, 2) numpy array."""
    if not d:
        return np.empty((0, 2), dtype=np.float64)
    items = sorted(d.items())
    return np.array(items, dtype=np.float64)


def compute_features_batch(
    symbol: str,
    df: pd.DataFrame,
    *,
    include_iv: bool = False,
    include_onchain: bool = False,
    include_v11: bool = False,
) -> pd.DataFrame:
    """Compute enriched features for all bars in df.

    C++ accelerated replacement for compute_oos_features().
    Falls back to Python if C++ module not available.
    """
    # Extract bar data as numpy arrays
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    timestamps = df[ts_col].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    opens = df["open"].values.astype(np.float64) if "open" in df.columns else closes.copy()
    highs = df["high"].values.astype(np.float64) if "high" in df.columns else closes.copy()
    lows = df["low"].values.astype(np.float64) if "low" in df.columns else closes.copy()
    volumes = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.zeros_like(closes)
    trades = df["trades"].values.astype(np.float64) if "trades" in df.columns else np.zeros_like(closes)
    tbv = (df["taker_buy_volume"].values.astype(np.float64)
           if "taker_buy_volume" in df.columns else np.zeros_like(closes))
    qv = df["quote_volume"].values.astype(np.float64) if "quote_volume" in df.columns else np.zeros_like(closes)
    tbqv = (df["taker_buy_quote_volume"].values.astype(np.float64)
            if "taker_buy_quote_volume" in df.columns else np.zeros_like(closes))

    # Replace NaN with 0 for volume-like fields
    for arr in [volumes, trades, tbv, qv, tbqv]:
        np.nan_to_num(arr, copy=False)

    # Load schedule data
    funding = _load_schedule(
        Path(f"data_files/{symbol}_funding.csv"), "timestamp", "funding_rate")
    oi = _load_schedule(
        Path(f"data_files/{symbol}_open_interest.csv"), "timestamp", "sum_open_interest")
    ls = _load_schedule(
        Path(f"data_files/{symbol}_ls_ratio.csv"), "timestamp", "long_short_ratio")
    spot_closes = _load_spot_closes(symbol)
    fgi_schedule = _load_fgi_schedule()

    # Convert schedules to sorted (M, 2) arrays
    funding_arr = _dict_to_sorted_array(funding)
    oi_arr = _dict_to_sorted_array(oi)
    ls_arr = _dict_to_sorted_array(ls)
    spot_arr = _dict_to_sorted_array(spot_closes)
    fgi_arr = _dict_to_sorted_array(fgi_schedule)

    # IV schedules (optional)
    empty_sched = np.empty((0, 2), dtype=np.float64)
    if include_iv:
        iv_arr = _dict_to_sorted_array(_load_iv_schedule(symbol))
        pcr_arr = _dict_to_sorted_array(_load_pcr_schedule(symbol))
    else:
        iv_arr = empty_sched
        pcr_arr = empty_sched

    # On-chain schedule: (M, 7) [ts, FlowIn, FlowOut, Supply, Addr, Tx, HR]
    if include_onchain:
        onchain_arr = _load_onchain_schedule(symbol)
    else:
        onchain_arr = np.empty((0, 7), dtype=np.float64)

    # V11 schedules
    if include_v11:
        liq_arr = _load_liq_schedule(symbol)
        mempool_arr = _load_mempool_schedule()
        macro_arr = _load_macro_schedule_arr()
    else:
        liq_arr = np.empty((0, 4), dtype=np.float64)
        mempool_arr = np.empty((0, 4), dtype=np.float64)
        macro_arr = np.empty((0, 4), dtype=np.float64)

    # Call C++ engine
    result = cpp_compute_all_features(
        timestamps, opens, highs, lows, closes, volumes,
        trades, tbv, qv, tbqv,
        funding_arr, oi_arr, ls_arr, spot_arr, fgi_arr,
        iv_arr, pcr_arr, onchain_arr,
        liq_arr, mempool_arr, macro_arr,
    )

    # Wrap as DataFrame
    feat_names = cpp_feature_names()
    feat_df = pd.DataFrame(result, columns=feat_names, index=df.index)

    # Replace NaN features with NaN (already done by C++ via NaN sentinel)
    feat_df["close"] = closes

    # Multi-timeframe 4h features (not in C++ engine)
    tf4h = compute_4h_features(df)
    for col in TF4H_FEATURE_NAMES:
        if col in tf4h.columns:
            feat_df[col] = tf4h[col].values

    # V14: BTC Dominance features (BTC/ETH ratio)
    _add_dominance_features(symbol, feat_df, closes)

    # V15: Interaction & statistical features (IC-screened 2026-03-18)
    _add_v15_features(feat_df)

    # V16: Orderbook proxy + IV spread + liquidation features
    _add_v16_features(symbol, feat_df, closes, highs, lows, volumes, tbv)

    # V17: On-chain features (TxTfrCnt, AdrActCnt, exchange flow)
    _add_v17_onchain_features(symbol, feat_df, timestamps)

    # V21: Cross-market features (SPY/QQQ/VIX/TLT/USO/COIN from Yahoo Finance)
    _add_cross_market_features(feat_df, timestamps)

    # V22: Deribit IV features (DVOL-based: iv_level, iv_rank_30d, iv_change_1d, etc.)
    _add_iv_features(symbol, feat_df, timestamps, closes)

    # V23: Stablecoin supply features (DeFiLlama: total supply change, z-score, dominance)
    _add_stablecoin_features(feat_df, timestamps)

    # V24: ETF volume/flow features (Yahoo Finance: IBIT/GBTC/ETHA dollar volume)
    _add_etf_volume_features(feat_df, timestamps)

    return feat_df



# V15-V24 feature functions extracted to batch_features_extra.py
from features.batch_features_extra import (  # noqa: E402
    _add_v15_features,
    _add_v16_features,
    _add_v17_onchain_features,
    _add_cross_market_features,
    _add_dominance_features,
    _add_iv_features,
    _add_stablecoin_features,
    _add_etf_volume_features,
    _load_liq_schedule,
    _load_mempool_schedule,
    _load_macro_schedule_arr,
    _load_onchain_schedule,
)
