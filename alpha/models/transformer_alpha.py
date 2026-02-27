"""Temporal Fusion Transformer alpha model — attention-based time series prediction.

EXPERIMENTAL: Not validated in production. Requires: pip install torch
"""
from __future__ import annotations

import logging
import math
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
class TransformerAlphaModel:
    """Transformer-based alpha model for time-series prediction.

    Uses positional encoding + multi-head attention over feature sequences.
    """

    name: str = "transformer_alpha"
    feature_names: Sequence[str] = ()
    seq_len: int = 20
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
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
        """Build the Transformer model architecture."""
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise RuntimeError("torch not installed: pip install torch") from e

        class _PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, max_len: int = 500) -> None:
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.pe[:, :x.size(1)]

        class _TransformerNet(nn.Module):
            def __init__(
                self, input_sz: int, d_model: int, n_heads: int, n_layers: int,
            ) -> None:
                super().__init__()
                self.input_proj = nn.Linear(input_sz, d_model)
                self.pos_enc = _PositionalEncoding(d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, batch_first=True,
                    dim_feedforward=d_model * 4, dropout=0.1,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.fc = nn.Linear(d_model, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.input_proj(x)
                x = self.pos_enc(x)
                x = self.encoder(x)
                x = x[:, -1, :]  # Take last time step
                return self.fc(x).squeeze(-1)

        self._model = _TransformerNet(input_size, self.d_model, self.n_heads, self.n_layers)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        epochs: int = 50,
        lr: float = 0.0001,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Train the model. X: (n_samples, seq_len, n_features), y: (n_samples,)."""
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

        split = int(len(X) * 0.8)
        X_train, X_val = X_tensor[:split], X_tensor[split:]
        y_train, y_val = y_tensor[:split], y_tensor[split:]

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            self._model.train()
            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                pred = self._model(X_train[idx])
                loss = criterion(pred, y_train[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = criterion(self._model(X_val), y_val).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        self._model.eval()
        with torch.no_grad():
            val_pred = self._model(X_val).numpy()
            val_true = y_val.numpy()
            direction_acc = float(np.mean(np.sign(val_pred) == np.sign(val_true)))

        logger.info("Transformer trained: val_loss=%.6f, acc=%.4f", best_val_loss, direction_acc)
        return {"val_loss": best_val_loss, "direction_accuracy": direction_acc}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            torch.save({
                "model_state": self._model.state_dict() if self._model else None,
                "features": list(self.feature_names),
            }, path)
        except ImportError:
            pass

    def load(self, path: str | Path) -> None:
        try:
            import torch
            data = torch.load(path, weights_only=False)
            if "features" in data:
                object.__setattr__(self, "feature_names", tuple(data["features"]))
        except ImportError:
            pass
