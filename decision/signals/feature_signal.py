# decision/signals/feature_signal.py
"""FeatureSignal — consumes LiveFeatureComputer output to generate trading signals.

NOTE: This module is not currently imported by any production path.
It may be used by research scripts or tests only. Consider archiving
if no longer needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from _quant_hotpath import rust_compute_feature_signal
from decision.market_access import get_float_attr
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

        close = get_float_attr(mkt, "close", "last_price")
        if close is None:
            return self._neutral(symbol)

        volume = get_float_attr(mkt, "volume") or 0.0

        # Feed bar to computer
        features = self.computer.on_bar(
            symbol, close=close, volume=volume,
        )

        # Need minimum data
        if features.ma_fast is None or features.ma_slow is None:
            return self._neutral(symbol)

        momentum = features.momentum or 0.0
        volatility = features.volatility or 0.0
        vwap_ratio = features.vwap_ratio if features.vwap_ratio is not None else 1.0

        side, score, confidence = rust_compute_feature_signal(
            momentum, volatility, vwap_ratio,
            self.momentum_threshold, self.vol_penalty_factor, self.vwap_weight,
        )

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
