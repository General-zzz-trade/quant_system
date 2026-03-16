"""ModelTrainer — walk-forward cross-validation and training orchestration.

Usage:
    trainer = ModelTrainer(model=XGBAlphaModel(), feature_names=["sma_20", "rsi_14"])
    results = trainer.walk_forward_train(X, y, n_splits=5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class TrainableModel(Protocol):
    """Protocol for models that can be trained."""
    name: str
    def fit(self, X: Any, y: Any, **kwargs: Any) -> Dict[str, float]: ...
    def save(self, path: str | Path) -> None: ...


@dataclass
class FoldResult:
    """Result from a single train/val fold."""
    fold: int
    train_size: int
    val_size: int
    metrics: Dict[str, float]


@dataclass
class ModelTrainer:
    """Walk-forward training orchestrator."""

    model: TrainableModel
    out_dir: Optional[Path] = None

    def walk_forward_train(
        self,
        X: Any,
        y: Any,
        *,
        n_splits: int = 5,
        min_train_size: int = 100,
        expanding: bool = True,
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[FoldResult]:
        """Walk-forward cross-validation.

        Splits data chronologically into n_splits folds.
        Each fold trains on past data and validates on the next chunk.

        Args:
            expanding: If True, training window expands. If False, fixed window.
        """
        try:
            import numpy as np  # noqa: F401
        except ImportError as e:
            raise RuntimeError("numpy required: pip install numpy") from e

        n = len(X)
        chunk_size = n // (n_splits + 1)

        if chunk_size < min_train_size:
            logger.warning("Not enough data for %d splits (chunk=%d)", n_splits, chunk_size)
            n_splits = max(1, n // min_train_size - 1)
            chunk_size = n // (n_splits + 1)

        results: List[FoldResult] = []

        for fold in range(n_splits):
            if expanding:
                train_end = chunk_size * (fold + 1)
            else:
                train_start = chunk_size * fold
                train_end = chunk_size * (fold + 1)

            val_start = train_end
            val_end = min(val_start + chunk_size, n)

            if expanding:
                X_train = X[:train_end]
                y_train = y[:train_end]
            else:
                X_train = X[train_start:train_end]
                y_train = y[train_start:train_end]

            X_val = X[val_start:val_end]
            y[val_start:val_end]

            if len(X_val) == 0:
                break

            kwargs = fit_kwargs or {}
            metrics = self.model.fit(X_train, y_train, **kwargs)

            results.append(FoldResult(
                fold=fold,
                train_size=len(X_train),
                val_size=len(X_val),
                metrics=metrics,
            ))

            logger.info(
                "Fold %d/%d: train=%d val=%d metrics=%s",
                fold + 1, n_splits, len(X_train), len(X_val), metrics,
            )

        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.out_dir / f"{self.model.name}_final.pkl")

        return results
