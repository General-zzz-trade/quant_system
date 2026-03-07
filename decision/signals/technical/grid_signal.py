from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional

from decision.market_access import get_decimal_attr
from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class GridSignal:
    grid_spacing: Decimal = Decimal("0.01")
    levels: int = 5
    name: str = "grid_signal"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        feats = getattr(snapshot, "features", None)
        if not isinstance(feats, Mapping):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        # Get close price: first try features, then snapshot.market.close
        raw_close = feats.get("close")
        if raw_close is None:
            market = getattr(snapshot, "market", None)
            raw_close = get_decimal_attr(market, "close", "last_price") if market is not None else None

        if raw_close is None:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        try:
            close = Decimal(str(raw_close))
        except (InvalidOperation, Exception):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        # Get reference price from features, fallback to close itself
        raw_ref = feats.get("grid_ref_price")
        try:
            ref = Decimal(str(raw_ref)) if raw_ref is not None else close
        except (InvalidOperation, Exception):
            ref = close

        if ref == Decimal("0"):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        try:
            delta = (close - ref) / ref
            if self.grid_spacing == Decimal("0"):
                return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
            levels_moved = delta / self.grid_spacing
        except (InvalidOperation, Exception):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        meta = {
            "close": str(close),
            "ref": str(ref),
            "levels_moved": str(round(levels_moved, 4)),
            "grid_spacing": str(self.grid_spacing),
        }

        if levels_moved > Decimal("0.5"):
            score = min(Decimal("1"), Decimal(str(round(levels_moved, 4))))
            # Confidence scales with number of levels moved, capped at 1
            confidence_raw = abs(levels_moved) / Decimal(str(self.levels))
            confidence = min(Decimal("1"), confidence_raw)
            return SignalResult(
                symbol=symbol,
                side="sell",
                score=score,
                confidence=confidence,
                meta=meta,
            )

        if levels_moved < Decimal("-0.5"):
            score = min(Decimal("1"), Decimal(str(round(abs(levels_moved), 4))))
            confidence_raw = abs(levels_moved) / Decimal(str(self.levels))
            confidence = min(Decimal("1"), confidence_raw)
            return SignalResult(
                symbol=symbol,
                side="buy",
                score=score,
                confidence=confidence,
                meta=meta,
            )

        return SignalResult(
            symbol=symbol,
            side="flat",
            score=Decimal("0"),
            confidence=Decimal("0.5"),
            meta=meta,
        )
