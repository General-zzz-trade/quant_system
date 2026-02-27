# engine/feature_hook.py
"""FeatureComputeHook — computes features from market events for the pipeline."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


class FeatureComputeHook:
    """Bridges LiveFeatureComputer into the engine pipeline.

    Called before pipeline.apply() to compute features from market events.
    Features are injected into PipelineInput.features and flow through
    to StateSnapshot.features for downstream ML signal consumption.

    Optionally integrates LiveInferenceBridge to run ML models and inject
    ml_score into the features dict.
    """

    def __init__(self, computer: Any, inference_bridge: Any = None) -> None:
        self._computer = computer
        self._inference = inference_bridge
        self._last_features: Dict[str, Dict[str, Any]] = {}

    def on_event(self, event: Any) -> Optional[Mapping[str, Any]]:
        """Compute features if this is a market event. Returns features dict or None."""
        kind = getattr(event, "event_type", None)
        if kind is not None:
            kind_val = getattr(kind, "value", kind)
            if "market" not in str(kind_val).lower():
                sym = getattr(event, "symbol", None)
                if sym and sym in self._last_features:
                    return dict(self._last_features[sym])
                return None

        symbol = getattr(event, "symbol", None)
        close = getattr(event, "close", None)
        if symbol is None or close is None:
            return None

        try:
            close_f = float(close)
        except (TypeError, ValueError):
            return None

        volume = float(getattr(event, "volume", 0) or 0)
        high = float(getattr(event, "high", 0) or 0)
        low = float(getattr(event, "low", 0) or 0)

        self._computer.on_bar(
            symbol, close=close_f, volume=volume, high=high, low=low,
        )

        features = self._computer.get_features_dict(symbol)
        features["close"] = close_f
        features["volume"] = volume

        if self._inference is not None:
            ts = getattr(event, "ts", None)
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    ts = None
            features = self._inference.enrich(symbol, ts, features)

        self._last_features[symbol] = features
        return features
