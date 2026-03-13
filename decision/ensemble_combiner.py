# decision/ensemble_combiner.py
"""Multi-timeframe ensemble combiner.

Fuses signals from multiple LiveInferenceBridge instances (potentially
running on different timeframes) into a single ml_score.

Conservative conflict policy: when bridges disagree on direction -> flat.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EnsembleCombiner:
    """Combines signals from multiple inference bridges.

    Each bridge is identified by a name (e.g., "h1_gate_v2", "m15_btc").
    Bridges are called sequentially; their ml_score outputs are fused.

    Conflict policy:
      - All bridges agree on direction -> weighted average
      - Any disagreement (mixed signs) -> 0.0 (flat)
      - Only one bridge produces signal -> pass through with its weight

    The combiner implements the same ``enrich(symbol, ts, features)``
    interface as ``LiveInferenceBridge``, so it is a drop-in replacement
    anywhere a bridge is expected (FeatureComputeHook, coordinator, etc.).
    """

    bridges: List[Tuple[str, Any]]  # (name, LiveInferenceBridge)
    weights: Optional[List[float]] = None
    conflict_policy: str = "flat"  # "flat" or "average"
    score_key: str = "ml_score"
    _weight_overrides: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = [1.0] * len(self.bridges)
        if len(self.weights) != len(self.bridges):
            raise ValueError(
                f"weights length {len(self.weights)} != bridges length {len(self.bridges)}"
            )

    # ------------------------------------------------------------------
    # Core interface (matches LiveInferenceBridge.enrich)
    # ------------------------------------------------------------------

    def enrich(
        self,
        symbol: str,
        ts: Optional[datetime],
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all bridges and combine their scores into features[score_key]."""
        scores: List[Tuple[str, float, float]] = []  # (name, score, weight)

        for i, (name, bridge) in enumerate(self.bridges):
            # Each bridge writes to self.score_key in features
            bridge.enrich(symbol, ts, features)
            score = features.get(self.score_key, 0.0)

            # Get effective weight (allow runtime overrides from attribution feedback)
            w = self._weight_overrides.get(name, self.weights[i])
            scores.append((name, float(score), w))

            # Store per-bridge score for attribution / monitoring
            features[f"ml_score_{name}"] = score

        if not scores:
            features[self.score_key] = 0.0
            return features

        # Filter out zero-weight bridges
        active = [(n, s, w) for n, s, w in scores if w > 0]

        if not active:
            features[self.score_key] = 0.0
            return features

        # Check for directional conflict
        signs = set()
        for _, s, _ in active:
            if s > 0:
                signs.add(1)
            elif s < 0:
                signs.add(-1)

        if len(signs) > 1 and self.conflict_policy == "flat":
            # Directional conflict -> go flat (conservative)
            combined = 0.0
            logger.debug(
                "Ensemble conflict for %s: %s -> flat",
                symbol,
                [(n, f"{s:.4f}") for n, s, _ in active],
            )
        else:
            # Weighted average
            total_weight = sum(w for _, _, w in active)
            if total_weight > 0:
                combined = sum(s * w for _, s, w in active) / total_weight
            else:
                combined = 0.0

        features[self.score_key] = combined

        logger.debug(
            "Ensemble %s: %s -> combined=%.4f",
            symbol,
            [(n, f"{s:.4f}", f"w={w:.2f}") for n, s, w in scores],
            combined,
        )

        return features

    # ------------------------------------------------------------------
    # Dynamic weight management (for Direction 18 attribution feedback)
    # ------------------------------------------------------------------

    def update_weight(self, bridge_name: str, weight: float) -> None:
        """Update weight for a specific bridge (used by attribution feedback)."""
        self._weight_overrides[bridge_name] = max(0.0, weight)
        logger.info("Ensemble weight updated: %s -> %.3f", bridge_name, weight)

    def get_weights(self) -> Dict[str, float]:
        """Return current effective weights per bridge."""
        result = {}
        for i, (name, _) in enumerate(self.bridges):
            result[name] = self._weight_overrides.get(name, self.weights[i])
        return result

    # ------------------------------------------------------------------
    # Checkpoint / restore (mirrors LiveInferenceBridge interface)
    # ------------------------------------------------------------------

    def checkpoint(self) -> Dict[str, Any]:
        """Serialize combiner state (delegates to each bridge)."""
        bridge_states = {}
        for name, bridge in self.bridges:
            bridge_states[name] = bridge.checkpoint()
        return {
            "bridge_states": bridge_states,
            "weight_overrides": dict(self._weight_overrides),
        }

    def restore(self, data: Dict[str, Any]) -> None:
        """Restore combiner state from checkpoint."""
        bridge_states = data.get("bridge_states", {})
        for name, bridge in self.bridges:
            if name in bridge_states:
                bridge.restore(bridge_states[name])
        self._weight_overrides = dict(data.get("weight_overrides", {}))

    # ------------------------------------------------------------------
    # Model update (forward to all sub-bridges)
    # ------------------------------------------------------------------

    def update_models(self, models: Sequence[Any]) -> None:
        """Forward model update to all bridges (used by SIGHUP reload)."""
        for name, bridge in self.bridges:
            if hasattr(bridge, "update_models"):
                bridge.update_models(models)
        logger.info(
            "EnsembleCombiner: forwarded update_models to %d bridges",
            len(self.bridges),
        )

    def update_params(
        self,
        symbol: str,
        *,
        deadzone: Optional[float] = None,
        min_hold: Optional[int] = None,
        max_hold: Optional[int] = None,
        long_only: Optional[bool] = None,
    ) -> None:
        """Forward param update to all bridges."""
        for name, bridge in self.bridges:
            if hasattr(bridge, "update_params"):
                bridge.update_params(
                    symbol,
                    deadzone=deadzone,
                    min_hold=min_hold,
                    max_hold=max_hold,
                    long_only=long_only,
                )
