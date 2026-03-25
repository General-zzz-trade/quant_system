from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from decision.market_access import get_decimal_attr
from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class BollingerBandSignal:
    upper_key: str = "bb_upper"
    lower_key: str = "bb_lower"
    middle_key: str = "bb_middle"
    close_key: str = "close"
    name: str = "bollinger_band"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        feats = getattr(snapshot, "features", None)
        if not isinstance(feats, Mapping):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        # Get close price: first try features, then snapshot.market.close
        raw_close = feats.get(self.close_key)
        if raw_close is None:
            market = getattr(snapshot, "market", None)
            raw_close = get_decimal_attr(market, "close", "last_price") if market is not None else None

        raw_upper = feats.get(self.upper_key)
        raw_lower = feats.get(self.lower_key)
        raw_middle = feats.get(self.middle_key)

        if any(v is None for v in (raw_close, raw_upper, raw_lower, raw_middle)):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        try:
            close = Decimal(str(raw_close))
            upper = Decimal(str(raw_upper))
            lower = Decimal(str(raw_lower))
            middle = Decimal(str(raw_middle))
        except (InvalidOperation, Exception):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        meta = {
            "close": str(close),
            "upper": str(upper),
            "lower": str(lower),
            "middle": str(middle),
        }

        if close > upper:
            return SignalResult(
                symbol=symbol,
                side="sell",
                score=Decimal("-1"),
                confidence=Decimal("0.8"),
                meta=meta,
            )

        if close < lower:
            return SignalResult(
                symbol=symbol,
                side="buy",
                score=Decimal("1"),
                confidence=Decimal("0.8"),
                meta=meta,
            )

        # Flat: score based on position relative to middle
        try:
            band_half = upper - middle
            if band_half == Decimal("0"):
                score = Decimal("0")
            else:
                raw_score = -(close - middle) / band_half
                # Clamp to [-1, 1]
                if raw_score > Decimal("1"):
                    score = Decimal("1")
                elif raw_score < Decimal("-1"):
                    score = Decimal("-1")
                else:
                    score = raw_score
        except (InvalidOperation, Exception):
            score = Decimal("0")

        return SignalResult(
            symbol=symbol,
            side="flat",
            score=score,
            confidence=Decimal("0.5"),
            meta=meta,
        )
