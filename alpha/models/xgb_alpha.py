"""XGBoost alpha model — gradient-boosted decision trees for signal prediction.

Implements AlphaModel Protocol. Requires: pip install xgboost scikit-learn

Features dict must contain numeric values keyed by feature name.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Signal:
    symbol: str
    ts: datetime
    side: str
    strength: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class XGBAlphaModel:
    """XGBoost-based alpha model.

    Predicts return direction: positive → long, negative → short.

    The model must be trained first via fit() or loaded via load().
    """

    name: str = "xgb_alpha"
    feature_names: Sequence[str] = ()
    threshold: float = 0.0
    _model: Any = None

    def predict(
        self, *, symbol: str, ts: datetime, features: Dict[str, Any],
    ) -> Optional[_Signal]:
        if self._model is None:
            return None

        try:
            import numpy as np
        except ImportError:
            return None

        x = np.array([[features.get(f, float("nan")) for f in self.feature_names]])

        # Handle both XGBRegressor (sklearn API) and raw Booster
        try:
            import xgboost as xgb
        except ImportError:
            return None
        if isinstance(self._model, xgb.Booster):
            dm = xgb.DMatrix(x, feature_names=list(self.feature_names))
            pred = self._model.predict(dm)[0]
        else:
            pred = self._model.predict(x)[0]

        if pred > self.threshold:
            return _Signal(symbol=symbol, ts=ts, side="long", strength=min(abs(pred), 1.0))
        elif pred < -self.threshold:
            return _Signal(symbol=symbol, ts=ts, side="short", strength=min(abs(pred), 1.0))
        return _Signal(symbol=symbol, ts=ts, side="flat", strength=0.0)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        params: Optional[Dict[str, Any]] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = 0,
        embargo_bars: int = 0,
        val_size: float = 0.2,
        sample_weight: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Train the model. X: 2D array of features, y: 1D array of returns.

        Returns training metrics dict.
        """
        try:
            import xgboost as xgb  # type: ignore[import-untyped]
            import numpy as np
        except ImportError as e:
            raise RuntimeError("Missing deps: pip install xgboost scikit-learn") from e

        n = len(X)
        split = int(n * (1.0 - val_size))
        train_end = max(split - embargo_bars, 1)
        val_start = split
        X_train, X_val = X[:train_end], X[val_start:]
        y_train, y_val = y[:train_end], y[val_start:]
        w_train = sample_weight[:train_end] if sample_weight is not None else None

        default_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }
        if params:
            default_params.update(params)

        if early_stopping_rounds > 0:
            default_params["early_stopping_rounds"] = early_stopping_rounds

        model = xgb.XGBRegressor(**default_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self._model = model

        val_pred = model.predict(X_val)
        mse = float(np.mean((val_pred - y_val) ** 2))
        direction_acc = float(np.mean(np.sign(val_pred) == np.sign(y_val)))

        metrics: Dict[str, float] = {
            "val_mse": mse,
            "direction_accuracy": direction_acc,
        }
        if early_stopping_rounds > 0:
            metrics["best_iteration"] = float(getattr(model, "best_iteration", n_estimators))

        logger.info("XGB trained: val_mse=%.6f, direction_accuracy=%.4f", mse, direction_acc)
        return metrics

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "features": self.feature_names}, f)
        from infra.model_signing import sign_file
        sign_file(path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)
        from infra.model_signing import load_verified_pickle
        data = load_verified_pickle(path)
        self._model = data["model"]
        if "features" in data:
            object.__setattr__(self, "feature_names", data["features"])
