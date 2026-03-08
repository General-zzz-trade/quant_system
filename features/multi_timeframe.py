"""Multi-timeframe features — resample 1h OHLCV to 4h and compute slow technicals.

Produces 10 features that capture slower regime signals invisible to 1h models.
All features are forward-filled back to the original 1h index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _quant_hotpath import cpp_compute_4h_features as _cpp_4h

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
