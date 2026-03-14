"""Multi-timeframe signal ensemble — fuses signals across 1m/5m/15m/1h."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from strategies.multi_timeframe.aggregator import AggregatedBar, BarAggregator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: str
    direction: int  # +1 long, -1 short, 0 flat
    strength: float  # 0.0 to 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnsembleSignal:
    """Fused signal across multiple timeframes."""
    direction: int  # +1 long, -1 short, 0 flat
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0 — agreement across timeframes
    components: Tuple[TimeframeSignal, ...]
    meta: Dict[str, Any] = field(default_factory=dict)


# Signal generator protocol: (bar) -> TimeframeSignal
SignalGenerator = Callable[[AggregatedBar], Optional[TimeframeSignal]]


@dataclass(frozen=True, slots=True)
class TimeframeConfig:
    """Configuration for a single timeframe in the ensemble."""
    timeframe: str
    weight: float
    signal_fn: SignalGenerator


class MultiTimeframeEnsemble:
    """Fuses signals from multiple timeframes into a single trading signal.

    Architecture:
    1. Feed 1m bars via on_bar()
    2. BarAggregator produces higher-timeframe bars
    3. Each timeframe runs its signal_fn on completed bars
    4. Signals are fused via weighted voting

    Fusion methods:
    - weighted_vote: direction = sign(sum(direction * weight * strength))
    - majority: direction = majority direction across timeframes
    - cascade: higher timeframe acts as filter for lower timeframe signals

    Usage:
        ensemble = MultiTimeframeEnsemble(
            symbol="BTCUSDT",
            configs=[
                TimeframeConfig("5m", weight=0.3, signal_fn=momentum_5m),
                TimeframeConfig("15m", weight=0.3, signal_fn=trend_15m),
                TimeframeConfig("1h", weight=0.4, signal_fn=regime_1h),
            ],
        )
        signal = ensemble.on_bar(ts, open, high, low, close, volume)
    """

    def __init__(
        self,
        symbol: str,
        configs: Sequence[TimeframeConfig],
        fusion_method: str = "weighted_vote",
        min_agreement: float = 0.5,
    ) -> None:
        self._symbol = symbol
        self._configs = list(configs)
        self._fusion_method = fusion_method
        self._min_agreement = min_agreement

        timeframes = [c.timeframe for c in configs]
        self._aggregator = BarAggregator(timeframes=timeframes, symbol=symbol)

        # Signal generators indexed by timeframe
        self._generators: Dict[str, SignalGenerator] = {c.timeframe: c.signal_fn for c in configs}
        self._weights: Dict[str, float] = {c.timeframe: c.weight for c in configs}

        # Last signal from each timeframe
        self._last_signals: Dict[str, TimeframeSignal] = {}

    def on_bar(
        self,
        ts: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> Optional[EnsembleSignal]:
        """Feed a 1m bar. Returns EnsembleSignal if any timeframe produced a new signal."""
        completed = self._aggregator.on_bar(ts, open, high, low, close, volume)

        updated = False
        for tf, bar in completed.items():
            gen = self._generators.get(tf)
            if gen is None:
                continue
            signal = gen(bar)
            if signal is not None:
                self._last_signals[tf] = signal
                updated = True

        if not updated or not self._last_signals:
            return None

        return self._fuse()

    def _fuse(self) -> EnsembleSignal:
        """Fuse signals from all timeframes with available data."""
        if self._fusion_method == "cascade":
            return self._fuse_cascade()
        elif self._fusion_method == "majority":
            return self._fuse_majority()
        else:
            return self._fuse_weighted_vote()

    def _fuse_weighted_vote(self) -> EnsembleSignal:
        """Weighted vote: direction = sign(sum(direction * weight * strength))."""
        total_score = 0.0
        total_weight = 0.0
        components: list[TimeframeSignal] = []

        for cfg in self._configs:
            sig = self._last_signals.get(cfg.timeframe)
            if sig is None:
                continue
            total_score += sig.direction * cfg.weight * sig.strength
            total_weight += cfg.weight
            components.append(sig)

        if total_weight == 0:
            return EnsembleSignal(
                direction=0, strength=0.0, confidence=0.0,
                components=tuple(components),
            )

        normalized = total_score / total_weight
        direction = 1 if normalized > 0.01 else (-1 if normalized < -0.01 else 0)
        strength = min(abs(normalized), 1.0)

        # Confidence: fraction of timeframes agreeing with the final direction
        agreeing = sum(1 for s in components if s.direction == direction and direction != 0)
        confidence = agreeing / len(components) if components else 0.0

        return EnsembleSignal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            components=tuple(components),
            meta={"method": "weighted_vote", "raw_score": normalized},
        )

    def _fuse_majority(self) -> EnsembleSignal:
        """Majority vote: direction = most common direction."""
        components: list[TimeframeSignal] = list(self._last_signals.values())
        if not components:
            return EnsembleSignal(direction=0, strength=0.0, confidence=0.0, components=())

        votes = {1: 0.0, -1: 0.0, 0: 0.0}
        for sig in components:
            votes[sig.direction] += sig.strength

        direction = max(votes, key=lambda d: votes[d])
        total_strength = sum(s.strength for s in components)
        strength = votes[direction] / total_strength if total_strength > 0 else 0.0

        agreeing = sum(1 for s in components if s.direction == direction)
        confidence = agreeing / len(components)

        return EnsembleSignal(
            direction=direction, strength=strength, confidence=confidence,
            components=tuple(components), meta={"method": "majority"},
        )

    def _fuse_cascade(self) -> EnsembleSignal:
        """Cascade: highest timeframe acts as regime filter.

        Only pass lower-timeframe signals that agree with the highest timeframe direction.
        """
        components: list[TimeframeSignal] = list(self._last_signals.values())
        if not components:
            return EnsembleSignal(direction=0, strength=0.0, confidence=0.0, components=())

        # Find highest timeframe signal (last in configs = highest weight)
        regime_tf = self._configs[-1].timeframe
        regime_sig = self._last_signals.get(regime_tf)

        if regime_sig is None or regime_sig.direction == 0:
            return EnsembleSignal(
                direction=0, strength=0.0, confidence=0.0,
                components=tuple(components), meta={"method": "cascade", "filter": "no_regime"},
            )

        # Filter lower timeframe signals
        agreeing = [s for s in components if s.direction == regime_sig.direction]
        if not agreeing:
            return EnsembleSignal(
                direction=0, strength=0.0, confidence=0.0,
                components=tuple(components), meta={"method": "cascade", "filter": "disagreement"},
            )

        avg_strength = sum(s.strength for s in agreeing) / len(agreeing)
        confidence = len(agreeing) / len(components)

        return EnsembleSignal(
            direction=regime_sig.direction,
            strength=min(avg_strength, 1.0),
            confidence=confidence,
            components=tuple(components),
            meta={"method": "cascade", "regime_tf": regime_tf},
        )

    @property
    def last_signals(self) -> Dict[str, TimeframeSignal]:
        return dict(self._last_signals)

    @property
    def aggregator(self) -> BarAggregator:
        return self._aggregator
