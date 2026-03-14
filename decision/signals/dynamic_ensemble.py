"""Dynamic ensemble signal with Sharpe-weighted model allocation.

NOTE: This module is not currently imported by any production path.
It may be used by research scripts or tests only. Consider archiving
if no longer needed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from math import exp
from typing import Any, Deque, Dict, Sequence, Tuple

from _quant_hotpath import rust_rolling_sharpe
from decision.types import SignalResult
from decision.signals.base import SignalModel


@dataclass
class DynamicEnsembleSignal:
    """Combines multiple signals with dynamically adjusted weights based on rolling Sharpe."""

    models: Sequence[Tuple[SignalModel, Decimal]]
    name: str = "dynamic_ensemble"
    lookback: int = 60
    min_weight: Decimal = Decimal("0.05")

    _weights: Dict[str, Decimal] = field(default_factory=dict, repr=False)
    _returns: Dict[str, Deque[float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for model, w in self.models:
            self._weights[model.name] = w
            self._returns[model.name] = deque(maxlen=self.lookback)

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        score = Decimal("0")
        conf = Decimal("0")
        metas: Dict[str, Any] = {}
        for model, _ in self.models:
            w = self._weights.get(model.name, Decimal("0"))
            r = model.compute(snapshot, symbol)
            metas[model.name] = {
                "side": r.side,
                "score": str(r.score),
                "weight": str(w),
            }
            score += r.score * w
            conf += r.confidence * abs(w)

        side = "flat"
        if score > 0:
            side = "buy"
        elif score < 0:
            side = "sell"
        return SignalResult(symbol=symbol, side=side, score=score, confidence=conf, meta=metas)

    def update_weights(self, returns_by_model: Dict[str, Sequence[float]]) -> None:
        """Update rolling returns and recompute weights via softmax on Sharpe ratios."""
        for name, rets in returns_by_model.items():
            if name not in self._returns:
                continue
            for r in rets:
                self._returns[name].append(r)

        sharpes: Dict[str, float] = {}
        for name, ret_deque in self._returns.items():
            sharpes[name] = rust_rolling_sharpe(list(ret_deque))

        self._reweight(sharpes)

    def _reweight(self, sharpes: Dict[str, float]) -> None:
        """Softmax-like reweighting based on Sharpe ratios."""
        names = [m.name for m, _ in self.models]
        vals = [sharpes.get(n, 0.0) for n in names]

        # Softmax with temperature=1
        max_v = max(vals) if vals else 0.0
        exps = [exp(v - max_v) for v in vals]
        total = sum(exps)
        if total == 0:
            return

        raw_weights = [e / total for e in exps]

        # Apply min_weight floor: set floor, redistribute excess from above-floor
        min_w = float(self.min_weight)
        n_models = len(raw_weights)
        floored = list(raw_weights)

        # Iteratively apply floor
        for _ in range(n_models):
            below = [i for i, w in enumerate(floored) if w < min_w]
            if not below:
                break
            deficit = sum(min_w - floored[i] for i in below)
            above = [i for i in range(n_models) if i not in below]
            if not above:
                break
            above_total = sum(floored[i] for i in above)
            for i in below:
                floored[i] = min_w
            for i in above:
                floored[i] -= deficit * (floored[i] / above_total) if above_total > 0 else 0

        # Final normalize to ensure sum=1
        total_f = sum(floored)
        normalized = [Decimal(str(round(w / total_f, 6))) for w in floored]

        for name, w in zip(names, normalized):
            self._weights[name] = w

    def get_weights(self) -> Dict[str, Decimal]:
        return dict(self._weights)
