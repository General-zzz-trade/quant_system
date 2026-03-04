"""LSTM alpha model — recurrent neural network for time-series prediction.

EXPERIMENTAL: Not validated in production. Requires: pip install torch

Uses sliding windows of features to predict next-period returns.
"""
from __future__ import annotations

import copy
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Signal:
    symbol: str
    ts: datetime
    side: str
    strength: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LSTMAlphaModel:
    """LSTM-based alpha model for time-series return prediction.

    Maintains a sliding window of feature vectors internally.
    """

    name: str = "lstm_alpha"
    feature_names: Sequence[str] = ()
    seq_len: int = 20
    hidden_size: int = 64
    num_layers: int = 2
    threshold: float = 0.0
    _model: Any = None
    _window: List[List[float]] = field(default_factory=list)

    def predict(
        self, *, symbol: str, ts: datetime, features: Dict[str, Any],
    ) -> Optional[_Signal]:
        if self._model is None:
            return None

        row = [features.get(f, 0.0) for f in self.feature_names]
        self._window.append(row)
        if len(self._window) > self.seq_len:
            self._window.pop(0)
        if len(self._window) < self.seq_len:
            return None

        try:
            import torch
        except ImportError:
            return None

        self._model.eval()
        with torch.no_grad():
            x = torch.tensor([self._window], dtype=torch.float32)
            pred = self._model(x).item()

        if pred > self.threshold:
            return _Signal(symbol=symbol, ts=ts, side="long", strength=min(abs(pred), 1.0))
        elif pred < -self.threshold:
            return _Signal(symbol=symbol, ts=ts, side="short", strength=min(abs(pred), 1.0))
        return _Signal(symbol=symbol, ts=ts, side="flat", strength=0.0)

    def build_model(self, input_size: int) -> None:
        """Build the LSTM model architecture."""
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise RuntimeError("torch not installed: pip install torch") from e

        class _LSTMNet(nn.Module):
            def __init__(self, input_sz: int, hidden: int, layers: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_sz, hidden, layers, batch_first=True, dropout=0.1 if layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                _, (h, _) = self.lstm(x)
                return self.fc(h[-1]).squeeze(-1)

        self._model = _LSTMNet(input_size, self.hidden_size, self.num_layers)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        patience: int = 5,
        embargo: int = 24,
    ) -> Dict[str, float]:
        """Train the LSTM model.

        X: 3D array (n_samples, seq_len, n_features)
        y: 1D array (n_samples,) of target returns
        patience: early stopping patience (epochs without improvement)
        embargo: gap between train and val to prevent leakage
        """
        try:
            import torch
            import torch.nn as nn
            import numpy as np
        except ImportError as e:
            raise RuntimeError("torch/numpy required") from e

        if self._model is None:
            self.build_model(X.shape[2])

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Split train/val with embargo gap
        split = int(len(X) * 0.8)
        train_end = max(split - embargo, 1)
        X_train, X_val = X_tensor[:train_end], X_tensor[split:]
        y_train, y_val = y_tensor[:train_end], y_tensor[split:]

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self._model.train()
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(epochs):
            # Mini-batch training
            perm = torch.randperm(len(X_train))

            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_train[idx], y_train[idx]

                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            self._model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info("LSTM early stop at epoch %d", epoch + 1)
                    break

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Final metrics
        self._model.eval()
        with torch.no_grad():
            val_pred = self._model(X_val).numpy()
            val_true = y_val.numpy()
            direction_acc = float(np.mean(np.sign(val_pred) == np.sign(val_true)))

        logger.info("LSTM trained: val_loss=%.6f, direction_acc=%.4f", best_val_loss, direction_acc)
        return {"val_loss": best_val_loss, "direction_accuracy": direction_acc}

    def predict_batch(self, X_3d: Any, batch_size: int = 256) -> Any:
        """Batch prediction on 3D input. Returns 1D numpy array."""
        try:
            import torch
            import numpy as np
        except ImportError as e:
            raise RuntimeError("torch/numpy required") from e

        if self._model is None:
            raise RuntimeError("Model not built/loaded")

        self._model.eval()
        X_tensor = torch.tensor(X_3d, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                preds.append(self._model(batch).numpy())
        return np.concatenate(preds)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            torch.save({
                "model_state": self._model.state_dict() if self._model else None,
                "features": list(self.feature_names),
                "config": {
                    "seq_len": self.seq_len,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                },
            }, path)
        except ImportError:
            with open(path, "wb") as f:
                pickle.dump({"model": None, "features": self.feature_names}, f)

    def load(self, path: str | Path) -> None:
        try:
            import torch
            data = torch.load(path, weights_only=False)
            if data.get("config"):
                cfg = data["config"]
                n_features = len(data.get("features", self.feature_names))
                self.build_model(n_features)
                if data.get("model_state") and self._model:
                    self._model.load_state_dict(data["model_state"])
            if "features" in data:
                object.__setattr__(self, "feature_names", tuple(data["features"]))
        except ImportError:
            pass
