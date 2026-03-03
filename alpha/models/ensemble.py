# alpha/models/ensemble.py
"""Ensemble alpha model — weighted average of multiple sub-models."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
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
class EnsembleAlphaModel:
    """Weighted average of multiple AlphaModel predictions.

    Each sub-model predicts independently; raw scores are averaged with weights.
    The ensemble produces a single Signal, so LiveInferenceBridge sees one model.
    """

    name: str = "ensemble"
    sub_models: Sequence[Any] = ()
    weights: Sequence[float] = ()

    def predict(
        self, *, symbol: str, ts: datetime, features: Dict[str, Any],
    ) -> Optional[_Signal]:
        if not self.sub_models:
            return None

        total_weight = 0.0
        weighted_score = 0.0

        for model, w in zip(self.sub_models, self.weights):
            sig = model.predict(symbol=symbol, ts=ts, features=features)
            if sig is None:
                continue
            score = sig.strength
            if sig.side == "short":
                score = -score
            elif sig.side == "flat":
                score = 0.0
            weighted_score += w * score
            total_weight += w

        if total_weight == 0.0:
            return None

        avg = weighted_score / total_weight

        if avg > 0:
            return _Signal(symbol=symbol, ts=ts, side="long", strength=min(abs(avg), 1.0))
        elif avg < 0:
            return _Signal(symbol=symbol, ts=ts, side="short", strength=min(abs(avg), 1.0))
        return _Signal(symbol=symbol, ts=ts, side="flat", strength=0.0)
