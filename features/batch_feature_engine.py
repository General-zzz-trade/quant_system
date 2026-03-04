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
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from features._quant_rolling import cpp_compute_all_features, cpp_feature_names
    _USING_CPP = True
except ImportError:
    _USING_CPP = False
    logger.warning("C++ feature engine not available, using Python fallback")


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
            schedule[int(row[ts_col])] = float(row[val_col])
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
            schedule[int(row[ts_col])] = float(row[val_col])
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
) -> pd.DataFrame:
    """Compute enriched features for all bars in df.

    C++ accelerated replacement for compute_oos_features().
    Falls back to Python if C++ module not available.
    """
    if not _USING_CPP:
        from scripts.backtest_alpha_v8 import compute_oos_features
        return compute_oos_features(symbol, df)

    # Extract bar data as numpy arrays
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    timestamps = df[ts_col].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    opens = df["open"].values.astype(np.float64) if "open" in df.columns else closes.copy()
    highs = df["high"].values.astype(np.float64) if "high" in df.columns else closes.copy()
    lows = df["low"].values.astype(np.float64) if "low" in df.columns else closes.copy()
    volumes = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.zeros_like(closes)
    trades = df["trades"].values.astype(np.float64) if "trades" in df.columns else np.zeros_like(closes)
    tbv = df["taker_buy_volume"].values.astype(np.float64) if "taker_buy_volume" in df.columns else np.zeros_like(closes)
    qv = df["quote_volume"].values.astype(np.float64) if "quote_volume" in df.columns else np.zeros_like(closes)
    tbqv = df["taker_buy_quote_volume"].values.astype(np.float64) if "taker_buy_quote_volume" in df.columns else np.zeros_like(closes)

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

    # Call C++ engine
    result = cpp_compute_all_features(
        timestamps, opens, highs, lows, closes, volumes,
        trades, tbv, qv, tbqv,
        funding_arr, oi_arr, ls_arr, spot_arr, fgi_arr,
        iv_arr, pcr_arr, onchain_arr,
    )

    # Wrap as DataFrame
    feat_names = cpp_feature_names()
    feat_df = pd.DataFrame(result, columns=feat_names, index=df.index)

    # Replace NaN features with NaN (already done by C++ via NaN sentinel)
    feat_df["close"] = closes

    return feat_df


def _load_onchain_schedule(symbol: str) -> np.ndarray:
    """Load on-chain metrics as (M, 7) array: [ts, FlowIn, FlowOut, Supply, Addr, Tx, HR]."""
    path = Path(f"data_files/{symbol}_onchain.csv")
    if not path.exists():
        return np.empty((0, 7), dtype=np.float64)

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row.get("timestamp", 0))
            flow_in = float(row.get("FlowInExUSD", "nan"))
            flow_out = float(row.get("FlowOutExUSD", "nan"))
            supply = float(row.get("SplyExNtv", "nan"))
            addr = float(row.get("AdrActCnt", "nan"))
            tx = float(row.get("TxTfrCnt", "nan"))
            hr = float(row.get("HashRate", "nan"))
            rows.append([ts, flow_in, flow_out, supply, addr, tx, hr])

    if not rows:
        return np.empty((0, 7), dtype=np.float64)
    arr = np.array(rows, dtype=np.float64)
    # Sort by timestamp
    arr = arr[arr[:, 0].argsort()]
    return arr
