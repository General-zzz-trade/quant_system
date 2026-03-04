"""NN utility functions for sliding window preparation and target alignment."""
from __future__ import annotations

import numpy as np


def make_sliding_windows(X_2d: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """Convert (N, F) feature matrix to (N - seq_len + 1, seq_len, F) sliding windows."""
    N, F = X_2d.shape
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if seq_len > N:
        raise ValueError(f"seq_len ({seq_len}) > N ({N})")
    out_len = N - seq_len + 1
    out = np.empty((out_len, seq_len, F), dtype=X_2d.dtype)
    for i in range(out_len):
        out[i] = X_2d[i:i + seq_len]
    return out


def align_target(y: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """Align target to sliding windows: y[seq_len - 1:].

    Each window's target corresponds to the last row in that window.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    return y[seq_len - 1:]
