"""Multi-factor strategy — combines momentum, value, quality, and volatility factors.

Generates cross-sectional signals by scoring and ranking assets
across multiple factor dimensions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class FactorScore:
    """Single factor score for a symbol."""
    symbol: str
    factor: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class CompositeSignal:
    """Combined multi-factor signal."""
    symbol: str
    composite_score: float
    factor_scores: Dict[str, float]
    rank: int
    side: str  # "long", "short", "flat"


def _zscore_values(values: Mapping[str, float]) -> Dict[str, float]:
    """Z-score normalize a dict of symbol → value."""
    if len(values) < 2:
        return {k: 0.0 for k in values}
    vals = list(values.values())
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    if std == 0:
        return {k: 0.0 for k in values}
    return {k: (v - mean) / std for k, v in values.items()}


def momentum_factor(
    returns: Mapping[str, Sequence[float]],
    lookback: int = 60,
    skip: int = 5,
) -> Dict[str, float]:
    """Momentum factor: cumulative return over lookback, skipping recent days."""
    scores: Dict[str, float] = {}
    for sym, rets in returns.items():
        if len(rets) < lookback:
            continue
        window = rets[-(lookback):-(skip) if skip > 0 else len(rets)]
        cum = 1.0
        for r in window:
            cum *= (1 + r)
        scores[sym] = cum - 1.0
    return _zscore_values(scores)


def volatility_factor(
    returns: Mapping[str, Sequence[float]],
    window: int = 20,
) -> Dict[str, float]:
    """Low-volatility factor: inverse of realized volatility (lower vol = higher score)."""
    scores: Dict[str, float] = {}
    for sym, rets in returns.items():
        if len(rets) < window:
            continue
        recent = rets[-window:]
        mean = sum(recent) / len(recent)
        var = sum((r - mean) ** 2 for r in recent) / len(recent)
        vol = math.sqrt(var)
        scores[sym] = -vol  # Negative so lower vol ranks higher
    return _zscore_values(scores)


def mean_reversion_factor(
    returns: Mapping[str, Sequence[float]],
    short_window: int = 5,
) -> Dict[str, float]:
    """Short-term reversal factor: negative of recent return."""
    scores: Dict[str, float] = {}
    for sym, rets in returns.items():
        if len(rets) < short_window:
            continue
        recent = rets[-short_window:]
        cum = 1.0
        for r in recent:
            cum *= (1 + r)
        scores[sym] = -(cum - 1.0)  # Negative: buy losers, sell winners
    return _zscore_values(scores)


class MultiFactorStrategy:
    """Combines multiple factor scores into a composite signal."""

    def __init__(
        self,
        *,
        factor_weights: Optional[Dict[str, float]] = None,
        long_threshold: float = 1.0,
        short_threshold: float = -1.0,
    ) -> None:
        self._weights = factor_weights or {
            "momentum": 0.4,
            "volatility": 0.3,
            "mean_reversion": 0.3,
        }
        self._long_threshold = long_threshold
        self._short_threshold = short_threshold

    def compute_signals(
        self,
        returns: Mapping[str, Sequence[float]],
    ) -> List[CompositeSignal]:
        """Compute multi-factor composite signals for all symbols."""
        factors: Dict[str, Dict[str, float]] = {}

        if "momentum" in self._weights:
            factors["momentum"] = momentum_factor(returns)
        if "volatility" in self._weights:
            factors["volatility"] = volatility_factor(returns)
        if "mean_reversion" in self._weights:
            factors["mean_reversion"] = mean_reversion_factor(returns)

        # Compute composite score
        all_symbols = set()
        for f_scores in factors.values():
            all_symbols.update(f_scores.keys())

        composites: Dict[str, float] = {}
        factor_details: Dict[str, Dict[str, float]] = {}

        for sym in all_symbols:
            score = 0.0
            details: Dict[str, float] = {}
            for fname, fscores in factors.items():
                fscore = fscores.get(sym, 0.0)
                weight = self._weights.get(fname, 0.0)
                score += weight * fscore
                details[fname] = fscore
            composites[sym] = score
            factor_details[sym] = details

        # Rank
        sorted_syms = sorted(composites.keys(), key=lambda s: composites[s], reverse=True)

        signals: List[CompositeSignal] = []
        for rank, sym in enumerate(sorted_syms):
            score = composites[sym]
            if score > self._long_threshold:
                side = "long"
            elif score < self._short_threshold:
                side = "short"
            else:
                side = "flat"

            signals.append(CompositeSignal(
                symbol=sym,
                composite_score=score,
                factor_scores=factor_details[sym],
                rank=rank + 1,
                side=side,
            ))

        return signals
