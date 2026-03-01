"""Dynamic feature selection based on rolling IC.

Selects top-K features by rolling IC before each walk-forward fold.
Uses OLS residualization from research.orthogonalize for marginal IC.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


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

    # Replace NaN with 0 for computation
    X_clean = np.nan_to_num(X, nan=0.0)
    y_clean = np.nan_to_num(y, nan=0.0)

    selected: List[int] = []
    remaining = set(range(n_features))

    for _ in range(min(top_k, n_features)):
        best_ic = -1.0
        best_idx = -1

        for j in remaining:
            if len(selected) == 0:
                # Raw IC
                col = X_clean[:, j]
                if np.std(col) < 1e-12:
                    continue
                ic = abs(float(np.corrcoef(col, y_clean)[0, 1]))
            else:
                # Residualize against selected features
                X_sel = X_clean[:, selected]
                col = X_clean[:, j]
                # Simple OLS residual: col - X_sel @ (X_sel'X_sel)^-1 X_sel'col
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
