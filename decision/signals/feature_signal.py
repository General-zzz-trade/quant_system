# decision/signals/feature_signal.py
"""FeatureSignal — consumes LiveFeatureComputer output to generate trading signals."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Optional

from decision.types import SignalResult
from features.live_computer import LiveFeatureComputer


@dataclass
class FeatureSignal:
    """Signal model that uses LiveFeatureComputer to generate trading signals.

    Combines momentum and volatility features into a composite score.
    Implements SignalModel protocol: compute(snapshot, symbol) -> SignalResult.

    Logic:
    - momentum > threshold → buy signal
    - momentum < -threshold → sell signal
    - High volatility reduces confidence
    - VWAP ratio > 1 confirms buy bias; < 1 confirms sell bias
    """

    name: str = "feature_signal"
    computer: LiveFeatureComputer = field(default_factory=LiveFeatureComputer)
    momentum_threshold: float = 0.001
    vol_penalty_factor: float = 2.0
    vwap_weight: float = 0.3

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        """Compute signal from snapshot market data."""
        # Extract price from snapshot
        markets = getattr(snapshot, "markets", {})
        mkt = markets.get(symbol)
        if mkt is None:
            return self._neutral(symbol)

        close = getattr(mkt, "close", None) or getattr(mkt, "last_price", None)
        if close is None:
            return self._neutral(symbol)

        volume = getattr(mkt, "volume", 0.0)

        # Feed bar to computer
        features = self.computer.on_bar(
            symbol, close=float(close), volume=float(volume or 0),
        )

        # Need minimum data
        if features.ma_fast is None or features.ma_slow is None:
            return self._neutral(symbol)

        # Base score from momentum
        momentum = features.momentum or 0.0
        if abs(momentum) < self.momentum_threshold:
            side = "flat"
            raw_score = 0.0
        elif momentum > 0:
            side = "buy"
            raw_score = min(momentum / self.momentum_threshold, 5.0) / 5.0
        else:
            side = "sell"
            raw_score = max(momentum / self.momentum_threshold, -5.0) / 5.0

        # VWAP confirmation
        vwap_bonus = 0.0
        if features.vwap_ratio is not None:
            vwap_dev = features.vwap_ratio - 1.0
            if (side == "buy" and vwap_dev > 0) or (side == "sell" and vwap_dev < 0):
                vwap_bonus = self.vwap_weight * abs(vwap_dev)
            elif side != "flat":
                vwap_bonus = -self.vwap_weight * abs(vwap_dev) * 0.5

        score = raw_score + vwap_bonus

        # Confidence: reduced by high volatility
        confidence = 1.0
        if features.volatility is not None and features.volatility > 0:
            vol_penalty = min(features.volatility * self.vol_penalty_factor * 100, 0.8)
            confidence = max(1.0 - vol_penalty, 0.2)

        return SignalResult(
            symbol=symbol,
            side=side,
            score=Decimal(str(round(score, 6))),
            confidence=Decimal(str(round(confidence, 4))),
            meta={
                "momentum": round(momentum, 6),
                "volatility": round(features.volatility, 6) if features.volatility else None,
                "vwap_ratio": round(features.vwap_ratio, 6) if features.vwap_ratio else None,
                "ma_fast": round(features.ma_fast, 4) if features.ma_fast else None,
                "ma_slow": round(features.ma_slow, 4) if features.ma_slow else None,
            },
        )

    @staticmethod
    def _neutral(symbol: str) -> SignalResult:
        return SignalResult(
            symbol=symbol, side="flat",
            score=Decimal("0"), confidence=Decimal("0"),
        )
