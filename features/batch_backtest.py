"""C++ accelerated backtest wrapper with Python fallback.

Usage:
    from features.batch_backtest import run_backtest_fast, pred_to_signal_fast

If the C++ extension is available, uses cpp_run_backtest / cpp_pred_to_signal.
Otherwise falls back to the Python implementation in backtest_alpha_v8.py.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from _quant_hotpath import cpp_run_backtest, cpp_pred_to_signal
    _BT_CPP = True
except ImportError:
    _BT_CPP = False


def pred_to_signal_fast(
    y_pred: np.ndarray,
    deadzone: float = 0.5,
    min_hold: int = 24,
    zscore_window: int = 720,
    zscore_warmup: int = 168,
) -> np.ndarray:
    if _BT_CPP:
        return np.asarray(cpp_pred_to_signal(
            y_pred.astype(np.float64, copy=False),
            deadzone, min_hold, zscore_window, zscore_warmup))
    from scripts.backtest_alpha_v8 import _pred_to_signal
    return _pred_to_signal(y_pred, deadzone=deadzone, min_hold=min_hold,
                           zscore_window=zscore_window)


def run_backtest_fast(
    timestamps: np.ndarray,
    closes: np.ndarray,
    y_pred: np.ndarray,
    *,
    volumes: Optional[np.ndarray] = None,
    vol_20: Optional[np.ndarray] = None,
    bear_probs: Optional[np.ndarray] = None,
    vol_values: Optional[np.ndarray] = None,
    funding_rates: Optional[np.ndarray] = None,
    funding_ts: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not _BT_CPP:
        raise ImportError("C++ backtest engine not available; use Python fallback")

    cfg = config or {}
    config_json = json.dumps(cfg)

    ts = timestamps.astype(np.int64, copy=False)
    cl = closes.astype(np.float64, copy=False)
    yp = y_pred.astype(np.float64, copy=False)
    vo = volumes.astype(np.float64, copy=False) if volumes is not None else np.empty(0, dtype=np.float64)
    v20 = vol_20.astype(np.float64, copy=False) if vol_20 is not None else np.empty(0, dtype=np.float64)
    bp = bear_probs.astype(np.float64, copy=False) if bear_probs is not None else np.empty(0, dtype=np.float64)
    vv = vol_values.astype(np.float64, copy=False) if vol_values is not None else np.empty(0, dtype=np.float64)
    fr = funding_rates.astype(np.float64, copy=False) if funding_rates is not None else np.empty(0, dtype=np.float64)
    ft = funding_ts.astype(np.int64, copy=False) if funding_ts is not None else np.empty(0, dtype=np.int64)

    return dict(cpp_run_backtest(ts, cl, vo, v20, yp, bp, vv, fr, ft, config_json))
