"""Shared signal post-processing helpers for research/backtest scripts.

These are thin Python wrappers. The canonical implementations live in
ext/rust/src/constraint_pipeline.rs. The batch Rust path (cpp_pred_to_signal)
calls the same shared functions. When Rust is unavailable, these pure-Python
fallbacks produce identical results.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from _quant_hotpath import cpp_pred_to_signal as _rust_pred_to_signal
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def rolling_zscore(pred: np.ndarray, window: int = 720, warmup: int = 180) -> np.ndarray:
    """Causal rolling z-score for prediction streams."""
    n = len(pred)
    z = np.zeros(n)
    buf: list[float] = []
    for i in range(n):
        buf.append(float(pred[i]))
        if len(buf) > window:
            buf.pop(0)
        if len(buf) < warmup:
            continue
        arr = np.array(buf)
        std = float(np.std(arr))
        if std < 1e-12:
            z[i] = 0.0
        else:
            z[i] = (float(pred[i]) - float(np.mean(arr))) / std
    return z


def _compute_bear_mask(closes: np.ndarray, ma_window: int = 480) -> np.ndarray:
    """Return boolean mask: True where close <= SMA(ma_window).

    Warmup bars (first ma_window) return False (allow trading),
    matching live check_monthly_gate() which returns True during warmup.
    """
    n = len(closes)
    mask = np.zeros(n, dtype=bool)
    if n < ma_window:
        return mask  # All False = allow trading during warmup
    cs = np.cumsum(closes)
    # First ma_window bars: mask stays False (allow trading)
    for i in range(ma_window, n):
        ma = (cs[i] - cs[i - ma_window]) / ma_window
        mask[i] = closes[i] <= ma
    return mask


def _apply_monthly_gate(
    signal: np.ndarray,
    closes: np.ndarray,
    ma_window: int = 480,
) -> np.ndarray:
    """Zero out signal when close <= SMA(ma_window)."""
    if len(signal) != len(closes):
        raise ValueError("signal and closes must have same length")
    out = signal.copy()
    out[_compute_bear_mask(closes, ma_window)] = 0.0
    return out


def _apply_trend_hold(
    signal: np.ndarray,
    trend_vals: np.ndarray,
    trend_threshold: float,
    max_hold: int,
) -> np.ndarray:
    """Extend long positions when trend is still favorable."""
    out = signal.copy()
    hold_count = 0
    for i in range(len(out)):
        if out[i] > 0:
            hold_count += 1
        elif i > 0 and out[i - 1] > 0 and out[i] == 0:
            tv = trend_vals[i] if i < len(trend_vals) else float("nan")
            if (not np.isnan(tv)
                    and tv > trend_threshold
                    and hold_count < max_hold):
                out[i] = out[i - 1]
                hold_count += 1
            else:
                hold_count = 0
        else:
            hold_count = 0
    return out


def _apply_vol_target(
    signal: np.ndarray,
    vol_vals: np.ndarray,
    vol_target: Optional[float],
) -> np.ndarray:
    """Scale non-zero signals by target_vol / realized_vol, capped at 1.0."""
    out = signal.copy()
    if vol_target is None:
        return out
    for i in range(len(out)):
        if out[i] != 0.0 and not np.isnan(vol_vals[i]) and vol_vals[i] > 1e-8:
            out[i] *= min(vol_target / vol_vals[i], 1.0)
    return out


def _enforce_min_hold(signal: np.ndarray, min_hold: int) -> np.ndarray:
    """Apply min-hold to an already discretized signal path."""
    if len(signal) == 0:
        return signal.copy()

    held = np.zeros_like(signal)
    held[0] = signal[0]
    hold_count = 1
    for i in range(1, len(signal)):
        if hold_count < min_hold:
            held[i] = held[i - 1]
            hold_count += 1
        else:
            held[i] = signal[i]
            if signal[i] != held[i - 1]:
                hold_count = 1
            else:
                hold_count += 1
    return held


def pred_to_signal(
    y_pred: np.ndarray,
    *,
    deadzone: float = 0.5,
    min_hold: int = 24,
    zscore_window: int = 720,
    zscore_warmup: int = 180,
    long_only: bool = False,
    trend_follow: bool = False,
    trend_values: Optional[np.ndarray] = None,
    trend_threshold: float = 0.0,
    max_hold: int = 120,
) -> np.ndarray:
    """Full constraint pipeline: z-score → discretize → min-hold → trend-hold.

    Delegates to Rust (cpp_pred_to_signal) when available, which calls the
    same shared constraint_pipeline.rs functions as the live/tick paths.
    Falls back to pure Python with identical semantics.
    """
    if _HAS_RUST:
        tv = list(trend_values) if trend_values is not None else []
        return np.array(_rust_pred_to_signal(
            list(y_pred), deadzone, min_hold, zscore_window, zscore_warmup,
            long_only, trend_follow, tv, trend_threshold, max_hold,
        ))

    # Pure Python fallback
    z = rolling_zscore(y_pred, window=zscore_window, warmup=zscore_warmup)
    if long_only:
        z = np.maximum(z, 0.0)
    raw = np.where(z > deadzone, 1.0, np.where(z < -deadzone, -1.0, 0.0))
    signal = _enforce_min_hold(raw, min_hold)
    if trend_follow and trend_values is not None:
        signal = _apply_trend_hold(signal, trend_values, trend_threshold, max_hold)
    return signal


def should_exit_position(
    *,
    position: float,
    z_value: float,
    held_bars: int,
    min_hold: int,
    max_hold: int,
    reversal_threshold: float = -0.3,
    deadzone_fade: float = 0.2,
) -> bool:
    """Shared exit rule for discrete z-score strategies."""
    if held_bars >= max_hold:
        return True
    if held_bars >= min_hold:
        return position * z_value < reversal_threshold or abs(z_value) < deadzone_fade
    return False
