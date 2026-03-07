# alpha/signal_transform.py
"""Canonical pred→signal conversion used by backtest, walk-forward, and live bridge.

This is the single source of truth for prediction-to-signal transformation.
LiveInferenceBridge._apply_constraints mirrors this logic for streaming/live use.
"""
from __future__ import annotations

import numpy as np

try:
    from _quant_hotpath import cpp_pred_to_signal as _cpp_pred_to_signal
    _HAS_CPP_SIGNAL = True
except ImportError:
    _HAS_CPP_SIGNAL = False


def pred_to_signal(
    y_pred: np.ndarray,
    target_mode: str = "",
    deadzone: float = 0.5,
    min_hold: int = 24,
    zscore_window: int = 720,
) -> np.ndarray:
    """Convert raw predictions to discrete positions {-1, 0, +1} with min hold.

    Uses rolling-window z-score normalization (causal — no lookahead).
    Each bar's z-score is computed using the last ``zscore_window`` predictions.

    Args:
        y_pred: Raw model predictions.
        target_mode: "binary" or continuous.
        deadzone: z-score threshold to enter a position.
        min_hold: Minimum bars to hold before allowing signal change.
        zscore_window: Rolling window size for z-score (default: 720 = 30 days).
    """
    # C++ fast path (handles rolling z-score + min-hold in single pass)
    if _HAS_CPP_SIGNAL and target_mode != "binary":
        y_c = np.ascontiguousarray(y_pred, dtype=np.float64)
        result = _cpp_pred_to_signal(y_c, deadzone, min_hold, zscore_window, min(168, zscore_window))
        return np.asarray(result, dtype=np.float64)

    # Step 1: raw discrete signal from predictions
    if target_mode == "binary":
        centered = y_pred - 0.5
        raw = np.sign(centered)
        raw = np.where(np.abs(centered) < 0.02, 0.0, raw)
    else:
        # Rolling-window z-score: causal, adapts to recent distribution
        n = len(y_pred)
        raw = np.zeros(n)
        buf = np.empty(zscore_window)
        buf_idx = 0
        buf_count = 0
        for i in range(n):
            buf[buf_idx] = y_pred[i]
            buf_idx = (buf_idx + 1) % zscore_window
            buf_count = min(buf_count + 1, zscore_window)
            if buf_count < min(168, zscore_window):
                continue  # warmup: need at least 168 bars (1 week)
            window = buf[:buf_count] if buf_count < zscore_window else buf
            mu = np.mean(window)
            std = np.std(window)
            if std < 1e-12:
                continue
            z = (y_pred[i] - mu) / std
            if z > deadzone:
                raw[i] = 1.0
            elif z < -deadzone:
                raw[i] = -1.0

    # Step 2: enforce minimum holding period
    signal = np.zeros_like(raw)
    signal[0] = raw[0]
    hold_count = 1
    for i in range(1, len(raw)):
        if hold_count < min_hold:
            signal[i] = signal[i - 1]
            hold_count += 1
        else:
            signal[i] = raw[i]
            if raw[i] != signal[i - 1]:
                hold_count = 1
            else:
                hold_count += 1
    return signal


def enforce_min_hold(signal: np.ndarray, min_hold: int = 24) -> np.ndarray:
    """Enforce minimum holding period on discrete signal {-1, 0, +1}."""
    out = np.zeros_like(signal)
    out[0] = signal[0]
    hold = 1
    for i in range(1, len(signal)):
        if hold < min_hold:
            out[i] = out[i - 1]
            hold += 1
        else:
            out[i] = signal[i]
            if signal[i] != out[i - 1]:
                hold = 1
            else:
                hold += 1
    return out
