"""Regime-adaptive model switching — dynamically adjusts model weights based on regime.

Uses the existing regime detection module (HMM, volatility, trend, composite)
to select or weight different alpha models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class AlphaModelLike(Protocol):
    """Minimal alpha model protocol for regime switching."""
    name: str
    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class RegimeWeight:
    """Model weight for a specific regime."""
    model_name: str
    regime: str
    weight: float


@dataclass
class RegimeModelSwitcher:
    """Selects or weights alpha models based on current market regime.

    Maintains a weight matrix: regime × model → weight.
    """

    models: Dict[str, AlphaModelLike] = field(default_factory=dict)
    regime_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    current_regime: str = "normal"

    def register_model(
        self,
        model: AlphaModelLike,
        regime_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Register an alpha model with per-regime weights."""
        self.models[model.name] = model
        if regime_weights:
            for regime, weight in regime_weights.items():
                self.regime_weights.setdefault(regime, {})[model.name] = weight

    def set_regime(self, regime: str) -> None:
        """Update current regime (called by regime detector)."""
        if regime != self.current_regime:
            logger.info("Regime switch: %s → %s", self.current_regime, regime)
            self.current_regime = regime

    def get_active_weights(self) -> Dict[str, float]:
        """Get current model weights based on active regime."""
        weights = self.regime_weights.get(self.current_regime, {})
        if not weights:
            # Equal weight fallback
            n = len(self.models)
            return {name: 1.0 / n for name in self.models} if n > 0 else {}
        return dict(weights)

    def predict_ensemble(
        self,
        *,
        symbol: str,
        ts: datetime,
        features: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run all models and return weighted ensemble prediction."""
        weights = self.get_active_weights()
        if not weights:
            return None

        results: Dict[str, Any] = {}
        weighted_score = 0.0
        total_weight = 0.0

        for name, weight in weights.items():
            model = self.models.get(name)
            if model is None or weight <= 0:
                continue

            try:
                signal = model.predict(symbol=symbol, ts=ts, features=features)
                if signal is None:
                    continue

                side = getattr(signal, "side", "flat")
                strength = getattr(signal, "strength", 0.0)

                direction = {"long": 1.0, "short": -1.0, "flat": 0.0}.get(side, 0.0)
                weighted_score += direction * strength * weight
                total_weight += weight

                results[name] = {"side": side, "strength": strength, "weight": weight}
            except Exception as e:
                logger.warning("Model %s failed: %s", name, e)

        if total_weight > 0:
            avg_score = weighted_score / total_weight
            ensemble_side = "long" if avg_score > 0.1 else "short" if avg_score < -0.1 else "flat"
            results["_ensemble"] = {
                "side": ensemble_side,
                "score": avg_score,
                "regime": self.current_regime,
            }

        return results
