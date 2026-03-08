# alpha/signal_transform.py
"""Canonical pred->signal conversion used by backtest, walk-forward, and live bridge."""
from __future__ import annotations

import numpy as np

from _quant_hotpath import cpp_pred_to_signal as _cpp_pred_to_signal


def pred_to_signal(
    y_pred: np.ndarray,
    target_mode: str = "",
    deadzone: float = 0.5,
    min_hold: int = 24,
    zscore_window: int = 720,
) -> np.ndarray:
    """Convert raw predictions to discrete positions {-1, 0, +1} with min hold."""
    if target_mode == "binary":
        centered = y_pred - 0.5
        raw = np.sign(centered)
        raw = np.where(np.abs(centered) < 0.02, 0.0, raw)
        return enforce_min_hold(raw, min_hold)

    y_c = np.ascontiguousarray(y_pred, dtype=np.float64)
    result = _cpp_pred_to_signal(y_c, deadzone, min_hold, zscore_window, min(168, zscore_window))
    return np.asarray(result, dtype=np.float64)


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
