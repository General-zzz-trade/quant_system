from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional

from decision.types import SignalResult, SignalSide
from strategy.signals.ml.features_contract import FeaturesContract


@dataclass(frozen=True, slots=True)
class ModelRunnerSignal:
    """Inference-only placeholder. Expects `features['ml_score']` or similar."""

    score_key: str = "ml_score"
    contract: Optional[FeaturesContract] = None
    name: str = "ml_runner"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        feats = getattr(snapshot, "features", None)
        if not isinstance(feats, Mapping):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        if self.contract is not None:
            ok, missing = self.contract.validate(feats)
            if not ok:
                return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"),
                    meta={"missing": missing})
        raw = feats.get(self.score_key)
        try:
            s = Decimal(str(raw))
        except Exception:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        side: SignalSide = "flat"
        if s > 0:
            side = "buy"
        elif s < 0:
            side = "sell"
        return SignalResult(symbol=symbol, side=side, score=s, confidence=Decimal("0.7"), meta={"score": str(s)})
