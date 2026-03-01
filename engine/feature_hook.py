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

    def __init__(self, computer: Any, inference_bridge: Any = None,
                 funding_rate_source: Any = None,
                 cross_asset_computer: Any = None,
                 oi_source: Any = None,
                 ls_ratio_source: Any = None) -> None:
        self._computer = computer
        self._inference = inference_bridge
        self._funding_rate_source = funding_rate_source
        self._cross_asset = cross_asset_computer
        self._oi_source = oi_source
        self._ls_ratio_source = ls_ratio_source
        self._last_features: Dict[str, Dict[str, Any]] = {}
        # Check once which extra params computer.on_bar accepts
        import inspect
        sig = inspect.signature(computer.on_bar)
        self._pass_open = "open_" in sig.parameters
        self._pass_hour = "hour" in sig.parameters
        self._pass_funding = "funding_rate" in sig.parameters
        self._pass_trades = "trades" in sig.parameters
        self._pass_oi = "open_interest" in sig.parameters

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
        open_ = float(getattr(event, "open", 0) or 0)

        bar_kwargs = {"close": close_f, "volume": volume, "high": high, "low": low}
        if self._pass_open:
            bar_kwargs["open_"] = open_

        if self._pass_hour:
            ts = getattr(event, "ts", None)
            if isinstance(ts, datetime):
                bar_kwargs["hour"] = ts.hour
                bar_kwargs["dow"] = ts.weekday()

        if self._pass_funding and self._funding_rate_source is not None:
            rate = self._funding_rate_source()
            if rate is not None:
                bar_kwargs["funding_rate"] = rate

        if self._pass_trades:
            bar_kwargs["trades"] = float(getattr(event, "trades", 0) or 0)
            bar_kwargs["taker_buy_volume"] = float(getattr(event, "taker_buy_volume", 0) or 0)
            bar_kwargs["quote_volume"] = float(getattr(event, "quote_volume", 0) or 0)

        if self._pass_oi:
            if self._oi_source is not None:
                oi_val = self._oi_source()
                if oi_val is not None:
                    bar_kwargs["open_interest"] = oi_val
            if self._ls_ratio_source is not None:
                ls_val = self._ls_ratio_source()
                if ls_val is not None:
                    bar_kwargs["ls_ratio"] = ls_val

        self._computer.on_bar(symbol, **bar_kwargs)

        features = self._computer.get_features_dict(symbol)
        features["close"] = close_f
        features["volume"] = volume

        if self._cross_asset is not None:
            funding_rate = bar_kwargs.get("funding_rate")
            self._cross_asset.on_bar(symbol, close=close_f, funding_rate=funding_rate)
            cross_feats = self._cross_asset.get_features(symbol)
            features.update(cross_feats)

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
