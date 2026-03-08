"""Adaptive ensemble signal — auto-tunes weights based on recent performance.

Supports three methods:
- ic_weighted: weight by rolling IC (signal-return correlation)
- inverse_vol: weight by inverse signal variance (stability)
- ridge: refit stacking ensemble on recent data
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Sequence

from _quant_hotpath import rust_adaptive_ensemble_calibrate
from decision.types import SignalResult
from decision.signals.base import SignalModel


@dataclass(frozen=True, slots=True)
class AdaptiveEnsembleConfig:
    """Configuration for adaptive weight calibration."""

    lookback_bars: int = 200
    recalibrate_every: int = 50
    method: str = "ic_weighted"  # "ic_weighted" | "inverse_vol" | "ridge"
    shrinkage: float = 0.3  # toward equal weight


class AdaptiveEnsembleSignal:
    """Ensemble signal that periodically recalibrates weights.

    Satisfies SignalModel protocol.
    """

    def __init__(
        self,
        signals: Sequence[SignalModel],
        config: AdaptiveEnsembleConfig = AdaptiveEnsembleConfig(),
    ) -> None:
        self.name = "adaptive_ensemble"
        self._signals = list(signals)
        self._config = config

        # Weight state
        k = len(signals)
        equal_w = 1.0 / max(k, 1)
        self._weights: Dict[str, float] = {s.name: equal_w for s in signals}

        # History buffers for recalibration
        self._score_history: Dict[str, List[float]] = {s.name: [] for s in signals}
        self._return_history: List[float] = []
        self._bars_since_calibration: int = 0

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        """Compute weighted ensemble signal."""
        scores: Dict[str, SignalResult] = {}
        weighted_score = Decimal("0")
        weighted_conf = Decimal("0")

        for sig in self._signals:
            result = sig.compute(snapshot, symbol)
            scores[sig.name] = result
            w = Decimal(str(self._weights.get(sig.name, 0.0)))
            weighted_score += result.score * w
            weighted_conf += result.confidence * abs(w)

        side = "flat"
        if weighted_score > 0:
            side = "buy"
        elif weighted_score < 0:
            side = "sell"

        meta = {
            "weights": dict(self._weights),
            "components": {
                name: {"side": r.side, "score": str(r.score)}
                for name, r in scores.items()
            },
        }

        return SignalResult(
            symbol=symbol, side=side,
            score=weighted_score, confidence=weighted_conf,
            meta=meta,
        )

    def record(self, signal_scores: Mapping[str, float], realized_return: float) -> None:
        """Record signal scores and realized return for calibration."""
        for name, score in signal_scores.items():
            if name in self._score_history:
                buf = self._score_history[name]
                buf.append(score)
                if len(buf) > self._config.lookback_bars:
                    buf[:] = buf[-self._config.lookback_bars:]

        self._return_history.append(realized_return)
        if len(self._return_history) > self._config.lookback_bars:
            self._return_history[:] = self._return_history[-self._config.lookback_bars:]

        self._bars_since_calibration += 1
        if self._bars_since_calibration >= self._config.recalibrate_every:
            self.recalibrate()

    def recalibrate(self) -> None:
        """Recalibrate weights based on recent signal performance."""
        self._bars_since_calibration = 0

        if len(self._return_history) < 20:
            return

        self._weights = rust_adaptive_ensemble_calibrate(
            self._config.method,
            self._score_history,
            self._return_history,
            self._config.shrinkage,
            self._config.lookback_bars,
        )

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)
