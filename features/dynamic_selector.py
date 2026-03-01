"""Dynamic feature selection based on rolling IC.

Selects top-K features by rolling IC before each walk-forward fold.
Uses OLS residualization from research.orthogonalize for marginal IC.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from features._quant_rolling import cpp_greedy_ic_select_np as _cpp_greedy_np
    _GREEDY_CPP = True
except ImportError:
    _GREEDY_CPP = False


def rolling_ic_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 20,
    ic_window: int = 500,
) -> List[str]:
    """Select top-K features by absolute rolling IC on most recent window.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array (n_samples,)
        feature_names: Feature names corresponding to columns
        top_k: Number of features to select
        ic_window: Number of recent bars to compute IC over

    Returns:
        List of selected feature names
    """
    n_samples, n_features = X.shape
    if n_samples < 50:
        return list(feature_names[:top_k])

    # Use the most recent `ic_window` bars
    start = max(0, n_samples - ic_window)
    X_recent = X[start:]
    y_recent = y[start:]

    ic_scores: List[tuple] = []
    for j in range(n_features):
        col = X_recent[:, j]
        valid = ~np.isnan(col) & ~np.isnan(y_recent)
        if valid.sum() < 30:
            ic_scores.append((feature_names[j], 0.0))
            continue
        x_clean = col[valid]
        y_clean = y_recent[valid]
        if np.std(x_clean) < 1e-12 or np.std(y_clean) < 1e-12:
            ic_scores.append((feature_names[j], 0.0))
            continue
        ic = float(np.corrcoef(x_clean, y_clean)[0, 1])
        ic_scores.append((feature_names[j], abs(ic)))

    # Sort by absolute IC descending
    ic_scores.sort(key=lambda x: -x[1])
    selected = [name for name, _ in ic_scores[:top_k]]
    return selected


def greedy_ic_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 20,
) -> List[str]:
    """Greedy forward selection maximizing marginal IC.

    Starts with the highest-IC feature, then iteratively adds the feature
    that has the highest IC after residualizing against already-selected.
    Avoids multicollinearity.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array (n_samples,)
        feature_names: Feature names
        top_k: Max features to select

    Returns:
        Ordered list of selected feature names
    """
    n_samples, n_features = X.shape
    if n_samples < 50 or n_features == 0:
        return list(feature_names[:top_k])

    if _GREEDY_CPP:
        X_c = np.ascontiguousarray(np.nan_to_num(X, nan=0.0), dtype=np.float64)
        y_c = np.ascontiguousarray(np.nan_to_num(y, nan=0.0), dtype=np.float64)
        indices = _cpp_greedy_np(X_c, y_c, top_k)
        return [feature_names[i] for i in indices]

    # Python fallback
    X_clean = np.nan_to_num(X, nan=0.0)
    y_clean = np.nan_to_num(y, nan=0.0)

    selected: List[int] = []
    remaining = set(range(n_features))

    for _ in range(min(top_k, n_features)):
        best_ic = -1.0
        best_idx = -1

        for j in remaining:
            if len(selected) == 0:
                col = X_clean[:, j]
                if np.std(col) < 1e-12:
                    continue
                ic = abs(float(np.corrcoef(col, y_clean)[0, 1]))
            else:
                X_sel = X_clean[:, selected]
                col = X_clean[:, j]
                try:
                    coef, _, _, _ = np.linalg.lstsq(X_sel, col, rcond=None)
                    residual = col - X_sel @ coef
                except np.linalg.LinAlgError:
                    residual = col
                if np.std(residual) < 1e-12:
                    continue
                ic = abs(float(np.corrcoef(residual, y_clean)[0, 1]))

            if ic > best_ic:
                best_ic = ic
                best_idx = j

        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [feature_names[i] for i in selected]


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Rank data (average method). Avoids scipy dependency."""
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    # Average ties
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
    """Select top-K features by IC Information Ratio (ICIR).

    ICIR = mean(|IC|) / std(IC) — measures IC stability across windows.
    Features with unstable IC (flipping signs) or low ICIR are filtered out.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target array (n_samples,).
        feature_names: Feature names corresponding to columns.
        top_k: Number of features to select.
        ic_window: Size of each non-overlapping IC window.
        n_windows: Number of windows to use (from end of data).
        min_icir: Minimum ICIR threshold for inclusion.
        max_consecutive_negative: Max consecutive negative-IC windows before discard.

    Returns:
        List of selected feature names, sorted by descending ICIR.
    """
    n_samples, n_features = X.shape
    total_needed = ic_window * n_windows
    if n_samples < total_needed or n_samples < 50:
        return list(feature_names[:top_k])

    # Cut n_windows non-overlapping windows from end
    start_offset = n_samples - total_needed
    windows = []
    for w in range(n_windows):
        ws = start_offset + w * ic_window
        we = ws + ic_window
        windows.append((ws, we))

    results: List[tuple] = []
    for j in range(n_features):
        ics = []
        for ws, we in windows:
            col = X[ws:we, j]
            tgt = y[ws:we]
            valid = ~np.isnan(col) & ~np.isnan(tgt)
            if valid.sum() < 30:
                ics.append(0.0)
                continue
            x_clean = col[valid]
            y_clean = tgt[valid]
            if np.std(x_clean) < 1e-12 or np.std(y_clean) < 1e-12:
                ics.append(0.0)
                continue
            ic = _spearman_ic(x_clean, y_clean)
            ics.append(ic)

        ic_arr = np.array(ics)

        # Check consecutive negative windows
        max_neg = 0
        cur_neg = 0
        for ic_val in ic_arr:
            if ic_val < 0:
                cur_neg += 1
                max_neg = max(max_neg, cur_neg)
            else:
                cur_neg = 0

        if max_neg >= max_consecutive_negative:
            continue

        ic_std = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0
        ic_mean_abs = float(np.mean(np.abs(ic_arr)))

        if ic_std < 1e-12:
            icir = ic_mean_abs * 100.0 if ic_mean_abs > 0 else 0.0
        else:
            icir = ic_mean_abs / ic_std

        if icir < min_icir:
            continue

        results.append((feature_names[j], icir))

    results.sort(key=lambda x: -x[1])
    selected = [name for name, _ in results[:top_k]]

    # If too few pass filters, relax and fill from absolute IC
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
    """Compute ICIR diagnostic report for all features.

    Returns:
        Dict mapping feature name to {mean_ic, std_ic, icir, max_consec_neg, pct_positive}.
    """
    n_samples, n_features = X.shape
    total_needed = ic_window * n_windows
    if n_samples < total_needed:
        return {}

    start_offset = n_samples - total_needed
    windows = []
    for w in range(n_windows):
        ws = start_offset + w * ic_window
        we = ws + ic_window
        windows.append((ws, we))

    report: Dict[str, Dict[str, float]] = {}
    for j in range(n_features):
        ics = []
        for ws, we in windows:
            col = X[ws:we, j]
            tgt = y[ws:we]
            valid = ~np.isnan(col) & ~np.isnan(tgt)
            if valid.sum() < 30:
                ics.append(0.0)
                continue
            x_clean = col[valid]
            y_clean = tgt[valid]
            if np.std(x_clean) < 1e-12 or np.std(y_clean) < 1e-12:
                ics.append(0.0)
                continue
            ic = _spearman_ic(x_clean, y_clean)
            ics.append(ic)

        ic_arr = np.array(ics)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0
        ic_mean_abs = float(np.mean(np.abs(ic_arr)))
        icir = ic_mean_abs / ic_std if ic_std > 1e-12 else (ic_mean_abs * 100.0 if ic_mean_abs > 0 else 0.0)

        max_neg = 0
        cur_neg = 0
        for ic_val in ic_arr:
            if ic_val < 0:
                cur_neg += 1
                max_neg = max(max_neg, cur_neg)
            else:
                cur_neg = 0

        pct_pos = float(np.mean(ic_arr > 0))

        report[feature_names[j]] = {
            "mean_ic": ic_mean,
            "std_ic": ic_std,
            "icir": icir,
            "max_consec_neg": float(max_neg),
            "pct_positive": pct_pos,
        }

    return report


def spearman_ic_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 20,
    ic_window: int = 500,
) -> List[str]:
    """Select top-K features by absolute Spearman rank IC.

    Uses rank correlation instead of Pearson — more robust to fat-tailed
    crypto returns and outlier features.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target array (n_samples,).
        feature_names: Feature names corresponding to columns.
        top_k: Number of features to select.
        ic_window: Number of recent bars to compute IC over.

    Returns:
        List of selected feature names, sorted by descending |rank IC|.
    """
    n_samples, n_features = X.shape
    if n_samples < 50:
        return list(feature_names[:top_k])

    start = max(0, n_samples - ic_window)
    X_recent = X[start:]
    y_recent = y[start:]

    ic_scores: List[tuple] = []
    for j in range(n_features):
        col = X_recent[:, j]
        valid = ~np.isnan(col) & ~np.isnan(y_recent)
        if valid.sum() < 30:
            ic_scores.append((feature_names[j], 0.0))
            continue
        x_clean = col[valid]
        y_clean = y_recent[valid]
        if np.std(x_clean) < 1e-12 or np.std(y_clean) < 1e-12:
            ic_scores.append((feature_names[j], 0.0))
            continue
        rx = _rankdata(x_clean)
        ry = _rankdata(y_clean)
        ic = float(np.corrcoef(rx, ry)[0, 1])
        ic_scores.append((feature_names[j], abs(ic)))

    ic_scores.sort(key=lambda x: -x[1])
    return [name for name, _ in ic_scores[:top_k]]
