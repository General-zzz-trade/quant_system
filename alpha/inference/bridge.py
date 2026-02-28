# alpha/inference/bridge.py
"""LiveInferenceBridge — connects InferenceEngine to the feature pipeline."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence

from alpha.base import AlphaModel
from alpha.inference import InferenceEngine

logger = logging.getLogger(__name__)


class LiveInferenceBridge:
    """Bridges feature computation to ML model inference.

    Called by FeatureComputeHook after features are computed.
    Runs all registered models and injects ml_score into the features dict.

    Usage:
        bridge = LiveInferenceBridge(models=[lgbm_model])
        features = bridge.enrich(symbol, ts, features_dict)
        # features now contains 'ml_score' key
    """

    def __init__(
        self,
        models: Sequence[AlphaModel],
        *,
        score_key: str = "ml_score",
    ) -> None:
        self._engine = InferenceEngine(models=list(models))
        self._score_key = score_key

    def enrich(
        self,
        symbol: str,
        ts: Optional[datetime],
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run inference and add ml_score to features dict."""
        if ts is None:
            ts = datetime.utcnow()

        results = self._engine.run(symbol=symbol, ts=ts, features=features)

        for r in results:
            if r.error is not None:
                logger.warning(
                    "Inference error for %s/%s: %s",
                    r.model_name, symbol, r.error,
                )
                continue
            if r.signal is not None:
                score = r.signal.strength
                if r.signal.side == "short":
                    score = -score
                elif r.signal.side == "flat":
                    score = 0.0
                features[self._score_key] = score
                logger.debug(
                    "Inference %s/%s: side=%s strength=%.4f latency=%.1fms",
                    r.model_name, symbol, r.signal.side, r.signal.strength, r.latency_ms,
                )

        return features
