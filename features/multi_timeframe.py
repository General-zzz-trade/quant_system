"""Multi-timeframe features — resample 1h OHLCV to 4h and compute slow technicals.

Produces 10 features that capture slower regime signals invisible to 1h models.
All features are forward-filled back to the original 1h index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from features._quant_rolling import cpp_compute_4h_features as _cpp_4h
    _MTF_CPP = True
except ImportError:
    _MTF_CPP = False

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


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (iterative, handles NaN start)."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1)
    started = False
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            continue
        if not started:
            out[i] = arr[i]
            started = True
        else:
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    cumsum = np.nancumsum(arr)
    out[window - 1:] = cumsum[window - 1:]
    out[window:] -= cumsum[:-window]
    out[window - 1:] /= window
    # Invalidate positions where we don't have enough data
    for i in range(window - 1):
        out[i] = np.nan
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation (ddof=1)."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        w = arr[i - window + 1:i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) >= window // 2:
            out[i] = np.std(valid, ddof=1)
    return out


def compute_4h_features(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLCV to 4h, compute 10 slow features, ffill back to 1h.

    Args:
        df_1h: DataFrame with columns: open_time (or timestamp), open, high, low, close, volume.

    Returns:
        DataFrame with same index as df_1h, containing TF4H_FEATURE_NAMES columns.
    """
    ts_col = "open_time" if "open_time" in df_1h.columns else "timestamp"
    ts = df_1h[ts_col].values.astype(np.int64)

    if _MTF_CPP:
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

    # Group into 4h bars by flooring timestamp
    four_hours_ms = 4 * 3600 * 1000
    group_keys = ts // four_hours_ms

    # Aggregate OHLCV into 4h bars
    df_work = pd.DataFrame({
        "group": group_keys,
        "open": df_1h["open"].values.astype(np.float64),
        "high": df_1h["high"].values.astype(np.float64),
        "low": df_1h["low"].values.astype(np.float64),
        "close": df_1h["close"].values.astype(np.float64),
        "volume": df_1h["volume"].values.astype(np.float64),
    })

    agg = df_work.groupby("group", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )

    close_4h = agg["close"].values
    high_4h = agg["high"].values
    low_4h = agg["low"].values
    n4 = len(close_4h)

    # --- Compute features on 4h bars ---

    # Returns
    ret_1 = np.full(n4, np.nan)
    ret_3 = np.full(n4, np.nan)
    ret_6 = np.full(n4, np.nan)
    for i in range(1, n4):
        ret_1[i] = close_4h[i] / close_4h[i - 1] - 1.0
    for i in range(3, n4):
        ret_3[i] = close_4h[i] / close_4h[i - 3] - 1.0
    for i in range(6, n4):
        ret_6[i] = close_4h[i] / close_4h[i - 6] - 1.0

    # RSI-14
    pct = np.full(n4, np.nan)
    for i in range(1, n4):
        pct[i] = close_4h[i] / close_4h[i - 1] - 1.0
    gains = np.where(~np.isnan(pct) & (pct > 0), pct, 0.0)
    losses = np.where(~np.isnan(pct) & (pct < 0), -pct, 0.0)
    avg_gain = _ema(gains, 14)
    avg_loss = _ema(losses, 14)
    rsi_14 = np.full(n4, np.nan)
    for i in range(n4):
        if not np.isnan(avg_gain[i]) and not np.isnan(avg_loss[i]):
            if avg_loss[i] < 1e-15:
                rsi_14[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi_14[i] = 100.0 - 100.0 / (1.0 + rs)

    # MACD histogram (12, 26, 9)
    ema12 = _ema(close_4h, 12)
    ema26 = _ema(close_4h, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    macd_hist = macd_line - signal_line
    # Normalize by close
    for i in range(n4):
        if close_4h[i] > 0 and not np.isnan(macd_hist[i]):
            macd_hist[i] /= close_4h[i]

    # Bollinger %B (20, 2)
    ma20 = _sma(close_4h, 20)
    std20 = _rolling_std(close_4h, 20)
    bb_pctb = np.full(n4, np.nan)
    for i in range(n4):
        if not np.isnan(ma20[i]) and not np.isnan(std20[i]) and std20[i] > 1e-15:
            upper = ma20[i] + 2.0 * std20[i]
            lower = ma20[i] - 2.0 * std20[i]
            bb_pctb[i] = (close_4h[i] - lower) / (upper - lower)

    # ATR normalized (14)
    tr = np.full(n4, np.nan)
    for i in range(1, n4):
        tr[i] = max(
            high_4h[i] - low_4h[i],
            abs(high_4h[i] - close_4h[i - 1]),
            abs(low_4h[i] - close_4h[i - 1]),
        )
    atr_raw = _ema(tr, 14)
    atr_norm = np.full(n4, np.nan)
    for i in range(n4):
        if not np.isnan(atr_raw[i]) and close_4h[i] > 0:
            atr_norm[i] = atr_raw[i] / close_4h[i]

    # Realized volatility (20-bar rolling std of returns)
    vol_20 = _rolling_std(pct, 20)

    # Close vs MA20
    close_vs_ma20 = np.full(n4, np.nan)
    for i in range(n4):
        if not np.isnan(ma20[i]) and ma20[i] > 0:
            close_vs_ma20[i] = close_4h[i] / ma20[i] - 1.0

    # Mean reversion z-score (close - ma20) / std20
    mean_rev = np.full(n4, np.nan)
    for i in range(n4):
        if not np.isnan(ma20[i]) and not np.isnan(std20[i]) and std20[i] > 1e-15:
            mean_rev[i] = (close_4h[i] - ma20[i]) / std20[i]

    # --- Build 4h feature DataFrame ---
    feat_4h = pd.DataFrame({
        "tf4h_ret_1": ret_1,
        "tf4h_ret_3": ret_3,
        "tf4h_ret_6": ret_6,
        "tf4h_rsi_14": rsi_14,
        "tf4h_macd_hist": macd_hist,
        "tf4h_bb_pctb_20": bb_pctb,
        "tf4h_atr_norm_14": atr_norm,
        "tf4h_vol_20": vol_20,
        "tf4h_close_vs_ma20": close_vs_ma20,
        "tf4h_mean_reversion_20": mean_rev,
    }, index=agg.index)  # index = group_keys

    # --- Map back to 1h rows: use PREVIOUS completed 4h bar (no look-ahead) ---
    # A 4h bar is only "completed" after its last 1h bar closes.
    # So a 1h row in 4h group G can only see features from group G-1.
    # This prevents look-ahead: group G's close uses data up to the
    # last 1h bar in G, which is in the future for earlier bars in G.
    agg_index_arr = agg.index.values
    group_to_idx = {int(g): i for i, g in enumerate(agg_index_arr)}

    row_mapping = np.array([
        group_to_idx.get(int(g) - 1, -1) for g in group_keys
    ])

    result_data = {}
    for col in TF4H_FEATURE_NAMES:
        col_4h = feat_4h[col].values
        col_1h = np.where(row_mapping >= 0, col_4h[row_mapping], np.nan)
        result_data[col] = col_1h

    result = pd.DataFrame(result_data, dtype=np.float64)
    result = result.ffill()

    return result
