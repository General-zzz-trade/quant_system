"""WeightedEnsembleSignal — combine multiple signals into one score.

NOTE: This module is not currently imported by any production path.
It is re-exported from decision.signals.__init__ but no production
code imports it. It may be used by research scripts or tests only.
Consider archiving if no longer needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Sequence, Tuple

from decision.types import SignalResult
from decision.signals.base import SignalModel


@dataclass(frozen=True, slots=True)
class WeightedEnsembleSignal:
    """Combine multiple signals into one score (deterministic)."""

    models: Sequence[Tuple[SignalModel, Decimal]]
    name: str = "ensemble"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        score = Decimal("0")
        conf = Decimal("0")
        metas = {}
        for m, w in self.models:
            r = m.compute(snapshot, symbol)
            metas[m.name] = {"side": r.side, "score": str(r.score), "confidence": str(r.confidence)}
            score += r.score * w
            conf += r.confidence * abs(w)
        side = "flat"
        if score > 0:
            side = "buy"
        elif score < 0:
            side = "sell"
        return SignalResult(symbol=symbol, side=side, score=score, confidence=conf, meta=metas)
