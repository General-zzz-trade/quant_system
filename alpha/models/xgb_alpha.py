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

        x = np.array([[features.get(f, 0.0) for f in self.feature_names]])
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
    ) -> Dict[str, float]:
        """Train the model. X: 2D array of features, y: 1D array of returns.

        Returns training metrics dict.
        """
        try:
            import xgboost as xgb  # type: ignore[import-untyped]
            import numpy as np
            from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("Missing deps: pip install xgboost scikit-learn") from e

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        default_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }
        if params:
            default_params.update(params)

        model = xgb.XGBRegressor(**default_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        self._model = model

        val_pred = model.predict(X_val)
        mse = float(np.mean((val_pred - y_val) ** 2))
        direction_acc = float(np.mean(np.sign(val_pred) == np.sign(y_val)))

        logger.info("XGB trained: val_mse=%.6f, direction_accuracy=%.4f", mse, direction_acc)
        return {"val_mse": mse, "direction_accuracy": direction_acc}

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "features": self.feature_names}, f)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        if "features" in data:
            object.__setattr__(self, "feature_names", data["features"])
