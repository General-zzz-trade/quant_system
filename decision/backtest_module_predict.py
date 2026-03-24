"""Prediction helpers for MLSignalDecisionModule (backtest).

Extracted from backtest_module.py to keep it under 500 lines.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional

import numpy as np


@dataclass
class _ZScoreBuf:
    """Rolling z-score normalization (causal, backward-looking only)."""
    window: int = 720
    warmup: int = 180
    _buf: Deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._buf = deque(maxlen=self.window)

    def push(self, value: float) -> float:
        self._buf.append(value)
        if len(self._buf) < self.warmup:
            return 0.0
        arr = np.array(self._buf)
        std = float(np.std(arr))
        if std < 1e-12:
            return 0.0
        return (value - float(np.mean(arr))) / std

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.warmup


def _resolve_primary_horizon_config(config: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    horizon_models = config.get("horizon_models") or []
    if not horizon_models:
        return None
    primary_horizon = config.get("primary_horizon")
    if primary_horizon is not None:
        for hm in horizon_models:
            if hm.get("horizon") == primary_horizon:
                return hm
    return horizon_models[0]


def _resolve_primary_model_artifacts(
    model_dir: Path,
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    primary = _resolve_primary_horizon_config(config)
    if primary is not None:
        lgbm_file = primary.get("lgbm")
        if not lgbm_file:
            return None
        artifacts: Dict[str, Any] = {
            "lgbm": model_dir / str(lgbm_file),
            "xgb": model_dir / str(primary.get("xgb")) if primary.get("xgb") else None,
            "ridge": model_dir / str(primary.get("ridge")) if primary.get("ridge") else None,
            "features": list(primary.get("features", [])),
            "ridge_features": primary.get("ridge_features"),
        }
        return artifacts if artifacts["lgbm"].exists() else None

    lgbm_path = model_dir / "lgbm_v8.pkl"
    if not lgbm_path.exists():
        return None
    return {
        "lgbm": lgbm_path,
        "xgb": model_dir / "xgb_v8.pkl" if (model_dir / "xgb_v8.pkl").exists() else None,
        "ridge": None,
        "features": None,
        "ridge_features": None,
    }
