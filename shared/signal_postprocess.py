"""Shared signal post-processing helpers for research/backtest scripts.

The canonical constraint pipeline lives in decision/rust/constraint_pipeline.rs.
The batch Rust path (cpp_pred_to_signal) is the only implementation — there is
no Python fallback. Pure-Python helpers (rolling_zscore, _compute_bear_mask,
etc.) remain for research/backtest scripts that need them independently.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from _quant_hotpath import cpp_pred_to_signal as _rust_pred_to_signal


def rolling_zscore(pred: np.ndarray, window: int = 720, warmup: int = 180) -> np.ndarray:
    """Causal rolling z-score for prediction streams.

    Note on bar/hourly equivalence: when bars are 1h, bar-level z-score
    equals hourly z-score (window=720 bars = 720 hours = 30 days).
    For sub-hourly bars, the effective time window shrinks proportionally
    (e.g., 5-min bars: 720 bars = 60 hours, not 720 hours), which inflates
    the z-score relative to the hourly baseline.
    """
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
    """Extend long/short positions when trend is still favorable."""
    out = signal.copy()
    hold_count = 0
    for i in range(len(out)):
        if out[i] != 0:
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
        elif i > 0 and out[i - 1] < 0 and out[i] == 0:
            tv = trend_vals[i] if i < len(trend_vals) else float("nan")
            if (not np.isnan(tv)
                    and tv < -trend_threshold
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


def _enforce_hold_single_pass(
    raw: np.ndarray,
    min_hold: int,
    trend_follow: bool = False,
    trend_values: Optional[np.ndarray] = None,
    trend_threshold: float = 0.0,
    max_hold: int = 120,
) -> np.ndarray:
    """Single-pass min-hold + trend-hold — mirrors Rust enforce_hold_array exactly.

    The old two-pass approach (_enforce_min_hold then _apply_trend_hold) had
    divergent hold_count semantics: the second pass started its own counter,
    allowing trend extension beyond what Rust permits.

    This single-pass version uses the same logic as enforce_hold_step in
    decision/rust/constraint_pipeline.rs.
    """
    n = len(raw)
    if n == 0:
        return raw.copy()

    signal = np.zeros(n)
    signal[0] = raw[0]
    hold_count = 1

    for i in range(1, n):
        desired = raw[i]
        prev = signal[i - 1]

        # Min-hold lockout
        if hold_count < min_hold:
            signal[i] = prev
            hold_count += 1
            continue

        # Trend hold: extend long when raw goes flat but trend is up
        if (trend_follow and desired == 0.0 and prev > 0.0
                and hold_count < max_hold):
            tv = trend_values[i] if (trend_values is not None
                                     and i < len(trend_values)) else float("nan")
            if not np.isnan(tv) and tv > trend_threshold:
                signal[i] = prev
                hold_count += 1
                continue

        # Short trend hold: extend short when raw goes flat but trend is down
        if (trend_follow and desired == 0.0 and prev < 0.0
                and hold_count < max_hold):
            tv = trend_values[i] if (trend_values is not None
                                     and i < len(trend_values)) else float("nan")
            if not np.isnan(tv) and tv < -trend_threshold:
                signal[i] = prev
                hold_count += 1
                continue

        # Allow change
        if desired != prev:
            signal[i] = desired
            hold_count = 1
        else:
            signal[i] = desired
            hold_count += 1

    return signal


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

    Delegates to Rust cpp_pred_to_signal, which calls the same shared
    constraint_pipeline.rs functions as the live/tick paths.
    """
    tv = list(trend_values) if trend_values is not None else []
    return np.array(_rust_pred_to_signal(
        list(y_pred), deadzone, min_hold, zscore_window, zscore_warmup,
        long_only, trend_follow, tv, trend_threshold, max_hold,
    ))


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
