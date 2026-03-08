"""LightGBM alpha model — fast gradient-boosted trees for signal prediction.

Implements AlphaModel Protocol. Requires: pip install lightgbm scikit-learn
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
class LGBMAlphaModel:
    """LightGBM-based alpha model.

    Faster training than XGBoost, handles categorical features natively.
    Uses RustTreePredictor for inference when .json companion file exists.
    """

    name: str = "lgbm_alpha"
    feature_names: Sequence[str] = ()
    threshold: float = 0.0
    _model: Any = None
    _is_classifier: bool = False
    _rust_predictor: Any = None

    def predict(
        self, *, symbol: str, ts: datetime, features: Dict[str, Any],
    ) -> Optional[_Signal]:
        if self._rust_predictor is None and self._model is None:
            return None

        if self._rust_predictor is not None:
            pred = self._rust_predictor.predict_dict(features)
        elif self._is_classifier:
            x = [[features.get(f, float("nan")) for f in self.feature_names]]
            prob = self._model.predict_proba(x)[0, 1]
            pred = prob - 0.5
        else:
            x = [[features.get(f, float("nan")) for f in self.feature_names]]
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
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        early_stopping_rounds: int = 0,
        embargo_bars: int = 0,
        val_size: float = 0.2,
        sample_weight: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Train the model. Returns training metrics.

        Args:
            early_stopping_rounds: Stop if val metric doesn't improve for N rounds. 0 = disabled.
            embargo_bars: Gap between train/val to prevent information leakage.
            val_size: Fraction of data for validation (taken from end, time-series safe).
        """
        try:
            import lightgbm as lgb
            import numpy as np
        except ImportError as e:
            raise RuntimeError("Missing deps: pip install lightgbm scikit-learn") from e

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
            "num_leaves": num_leaves,
            "objective": "regression",
            "verbosity": -1,
        }
        if params:
            default_params.update(params)

        callbacks = []
        if early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))

        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks if callbacks else None,
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
            metrics["best_iteration"] = float(model.best_iteration_)

        logger.info(
            "LGBM trained: val_mse=%.6f, direction_accuracy=%.4f, best_iter=%s",
            mse, direction_acc,
            model.best_iteration_ if early_stopping_rounds > 0 else "N/A",
        )
        return metrics

    def fit_classifier(
        self,
        X: Any,
        y_binary: Any,
        *,
        params: Optional[Dict[str, Any]] = None,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        early_stopping_rounds: int = 0,
        embargo_bars: int = 0,
        val_size: float = 0.2,
        sample_weight: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Train binary classifier. Output is prob - 0.5 (centered at 0).

        Args:
            y_binary: Binary target (1 if vol_normalized_return > 0, else 0).
        """
        try:
            import lightgbm as lgb
            import numpy as np
        except ImportError as e:
            raise RuntimeError("Missing deps: pip install lightgbm") from e

        n = len(X)
        split = int(n * (1.0 - val_size))
        train_end = max(split - embargo_bars, 1)
        val_start = split
        X_train, X_val = X[:train_end], X[val_start:]
        y_train, y_val = y_binary[:train_end], y_binary[val_start:]
        w_train = sample_weight[:train_end] if sample_weight is not None else None

        default_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
        }
        if params:
            default_params.update(params)

        callbacks = []
        if early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))

        model = lgb.LGBMClassifier(**default_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks if callbacks else None,
        )

        self._model = model
        self._is_classifier = True

        val_prob = model.predict_proba(X_val)[:, 1]
        val_pred_centered = val_prob - 0.5
        direction_acc = float(np.mean(
            (val_pred_centered > 0).astype(int) == y_val.astype(int)
        ))
        logloss = float(-np.mean(
            y_val * np.log(np.clip(val_prob, 1e-15, 1.0))
            + (1 - y_val) * np.log(np.clip(1 - val_prob, 1e-15, 1.0))
        ))

        metrics: Dict[str, float] = {
            "val_logloss": logloss,
            "direction_accuracy": direction_acc,
        }
        if early_stopping_rounds > 0:
            metrics["best_iteration"] = float(model.best_iteration_)

        logger.info(
            "LGBM classifier: logloss=%.6f, dir_acc=%.4f, best_iter=%s",
            logloss, direction_acc,
            model.best_iteration_ if early_stopping_rounds > 0 else "N/A",
        )
        return metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "features": self.feature_names,
                "is_classifier": self._is_classifier,
            }, f)
        from infra.model_signing import sign_file
        sign_file(path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        from infra.model_signing import load_verified_pickle  # HMAC-signed artifacts
        data = load_verified_pickle(path)
        self._model = data["model"]
        if "features" in data:
            object.__setattr__(self, "feature_names", data["features"])
        self._is_classifier = data.get("is_classifier", False)
        self._try_load_rust(path)

    def _try_load_rust(self, pkl_path: Path) -> None:
        json_path = pkl_path.with_suffix(".json")
        if not json_path.exists():
            return
        try:
            from _quant_hotpath import RustTreePredictor
            self._rust_predictor = RustTreePredictor.load(str(json_path))
            logger.info("Rust native inference: %s", self._rust_predictor.info())
        except Exception as e:
            logger.warning("Rust predictor load failed, using Python: %s", e)
