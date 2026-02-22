# alpha/training
"""Alpha model training — offline model fitting and hyperparameter tuning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence


@dataclass(frozen=True)
class TrainingConfig:
    """训练配置。"""
    model_name: str
    symbols: tuple[str, ...] = ()
    lookback_days: int = 365
    validation_pct: float = 0.2
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    """训练结果。"""
    model_name: str
    train_score: float = 0.0
    val_score: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    artifact_path: str = ""
