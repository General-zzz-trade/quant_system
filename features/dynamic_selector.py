"""Dynamic feature selection based on rolling IC.

Selects top-K features by rolling IC before each walk-forward fold.
Uses OLS residualization from research.orthogonalize for marginal IC.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from _quant_hotpath import (
    cpp_greedy_ic_select_np as _cpp_greedy_np,
    cpp_rolling_ic_select as _cpp_rolling_ic,
    cpp_spearman_ic_select as _cpp_spearman_ic,
    cpp_icir_select as _cpp_icir,
    cpp_stable_icir_select as _cpp_stable_icir,
    cpp_feature_icir_report as _cpp_icir_report,
)


def rolling_ic_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 20,
    ic_window: int = 500,
) -> List[str]:
    """Select top-K features by absolute rolling IC on most recent window."""
    n_samples, n_features = X.shape
    if n_samples < 50:
        return list(feature_names[:top_k])

    X_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    indices = _cpp_rolling_ic(X_c, y_c, top_k=top_k, ic_window=ic_window)
    return [feature_names[i] for i in indices]


def greedy_ic_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 20,
) -> List[str]:
    """Greedy forward selection maximizing marginal IC."""
    n_samples, n_features = X.shape
    if n_samples < 50 or n_features == 0:
        return list(feature_names[:top_k])

    X_c = np.ascontiguousarray(np.nan_to_num(X, nan=0.0), dtype=np.float64)
    y_c = np.ascontiguousarray(np.nan_to_num(y, nan=0.0), dtype=np.float64)
    indices = _cpp_greedy_np(X_c, y_c, top_k)
    return [feature_names[i] for i in indices]


def icir_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 20,
    ic_window: int = 200,
    n_windows: int = 5,
    min_icir: float = 0.3,
    max_consecutive_negative: int = 3,
) -> List[str]:
    """Select top-K features by IC Information Ratio (ICIR)."""
    n_samples, n_features = X.shape
    total_needed = ic_window * n_windows
    if n_samples < total_needed or n_samples < 50:
        return list(feature_names[:top_k])

    X_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    indices = _cpp_icir(X_c, y_c, top_k=top_k, ic_window=ic_window,
                        n_windows=n_windows, min_icir=min_icir,
                        max_consec_neg=max_consecutive_negative)
    selected = [feature_names[i] for i in indices]
    if len(selected) < top_k:
        already = set(selected)
        fallback = spearman_ic_select(X, y, feature_names, top_k=top_k)
        for name in fallback:
            if name not in already:
                selected.append(name)
                already.add(name)
            if len(selected) >= top_k:
                break
    return selected


def compute_feature_icir_report(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    ic_window: int = 200,
    n_windows: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Compute ICIR diagnostic report for all features."""
    n_samples, n_features = X.shape
    total_needed = ic_window * n_windows
    if n_samples < total_needed:
        return {}

    X_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    raw = _cpp_icir_report(X_c, y_c, ic_window=ic_window, n_windows=n_windows)
    arr = np.asarray(raw, dtype=np.float64).reshape(n_features, 5)
    report: Dict[str, Dict[str, float]] = {}
    for j in range(n_features):
        report[feature_names[j]] = {
            "mean_ic": float(arr[j, 0]),
            "std_ic": float(arr[j, 1]),
            "icir": float(arr[j, 2]),
            "max_consec_neg": float(arr[j, 3]),
            "pct_positive": float(arr[j, 4]),
        }
    return report


def stable_icir_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 20,
    ic_window: int = 200,
    n_windows: int = 5,
    min_icir: float = 0.3,
    min_stable_folds: int = 4,
    sign_consistency_threshold: float = 0.8,
) -> List[str]:
    """Select features by ICIR with stability gate and sign consistency."""
    n_samples, n_features = X.shape
    total_needed = ic_window * n_windows
    if n_samples < total_needed or n_samples < 50:
        return greedy_ic_select(X, y, feature_names, top_k=top_k)

    X_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    indices = _cpp_stable_icir(X_c, y_c, top_k=top_k, ic_window=ic_window,
                                n_windows=n_windows, min_icir=min_icir,
                                min_stable_folds=min_stable_folds,
                                sign_consistency=sign_consistency_threshold)
    if len(indices) == 0:
        return greedy_ic_select(X, y, feature_names, top_k=top_k)
    return [feature_names[i] for i in indices]


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Rank data (average method). Avoids scipy dependency."""
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(ranks[order[i:j]])
            ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank IC between two arrays (no scipy dependency)."""
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_ic_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 20,
    ic_window: int = 500,
) -> List[str]:
    """Select top-K features by absolute Spearman rank IC."""
    n_samples, n_features = X.shape
    if n_samples < 50:
        return list(feature_names[:top_k])

    X_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    indices = _cpp_spearman_ic(X_c, y_c, top_k=top_k, ic_window=ic_window)
    return [feature_names[i] for i in indices]
