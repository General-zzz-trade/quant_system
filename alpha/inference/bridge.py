# alpha/inference/bridge.py
"""LiveInferenceBridge — connects InferenceEngine to the feature pipeline."""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Union

from alpha.base import AlphaModel
from alpha.inference import InferenceEngine

logger = logging.getLogger(__name__)


class LiveInferenceBridge:
    """Bridges feature computation to ML model inference.

    Called by FeatureComputeHook after features are computed.
    Runs all registered models and injects ml_score into the features dict.

    Supports optional signal constraints to match backtest behavior:
      - min_hold_bars: minimum bars before position change (per-symbol)
      - long_only_symbols: clip short signals to 0 (per-symbol)
      - deadzone: z-score threshold for discretization

    When min_hold_bars is None (default), raw float scores pass through unchanged.
    """

    def __init__(
        self,
        models: Sequence[AlphaModel],
        *,
        score_key: str = "ml_score",
        metrics_exporter: Any = None,
        min_hold_bars: Optional[Dict[str, int]] = None,
        long_only_symbols: Optional[Set[str]] = None,
        deadzone: Union[float, Dict[str, float]] = 0.5,
        trend_follow: bool = False,
        trend_indicator: str = "tf4h_close_vs_ma20",
        trend_threshold: float = 0.0,
        max_hold: int = 120,
        monthly_gate: bool = False,
        monthly_gate_window: Union[int, Dict[str, int]] = 480,
        bear_model: Optional[AlphaModel] = None,
        bear_thresholds: Optional[Sequence[tuple]] = None,
        vol_target: Union[None, float, Dict[str, Optional[float]]] = None,
        vol_feature: Union[str, Dict[str, str]] = "atr_norm_14",
    ) -> None:
        self._engine = InferenceEngine(models=list(models))
        self._score_key = score_key
        self._metrics = metrics_exporter
        self._min_hold_bars = min_hold_bars or {}
        self._long_only_symbols = long_only_symbols or set()
        self._deadzone = deadzone
        self._trend_follow = trend_follow
        self._trend_indicator = trend_indicator
        self._trend_threshold = trend_threshold
        self._max_hold = max_hold
        self._monthly_gate = monthly_gate
        self._monthly_gate_window = monthly_gate_window
        self._bear_model = bear_model
        if bear_thresholds is not None:
            thresholds_only = [t[0] for t in bear_thresholds]
            if thresholds_only != sorted(thresholds_only, reverse=True):
                raise ValueError(
                    f"bear_thresholds must be in descending order by threshold, "
                    f"got: {bear_thresholds}"
                )
        self._bear_thresholds = bear_thresholds
        self._vol_target = vol_target
        self._vol_feature = vol_feature
        # Per-symbol state for min_hold enforcement
        self._position: Dict[str, float] = {}
        self._hold_counter: Dict[str, int] = {}
        # Per-symbol close history for monthly gate
        self._close_history: Dict[str, deque] = {}

    def update_models(self, models: Sequence[AlphaModel]) -> None:
        self._engine.set_models(models)
        self._position.clear()
        self._hold_counter.clear()
        self._close_history.clear()
        logger.info("Models hot-swapped: %d model(s)", len(models))

    @staticmethod
    def _resolve(param, symbol: str, default=None):
        """Resolve a scalar-or-dict parameter to a per-symbol value."""
        if isinstance(param, dict):
            return param.get(symbol, default)
        return param if param is not None else default

    def _apply_constraints(
        self, symbol: str, raw_score: float, features: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Apply long_only clip, discretization, min_hold, and trend_hold constraints."""
        min_hold = self._min_hold_bars.get(symbol)
        if min_hold is None:
            # No constraints — raw float passthrough (backward-compatible)
            return raw_score

        # Long-only clip
        score = raw_score
        if symbol in self._long_only_symbols:
            score = max(0.0, score)

        # Discretize: z > deadzone → +1, z < -deadzone → -1, else 0
        dz = self._resolve(self._deadzone, symbol, 0.5)
        if score > dz:
            desired = 1.0
        elif score < -dz:
            desired = -1.0
        else:
            desired = 0.0

        # Min-hold enforcement
        prev_pos = self._position.get(symbol, 0.0)
        hold_count = self._hold_counter.get(symbol, min_hold)

        if hold_count < min_hold:
            # Still in holding period — keep previous position
            self._hold_counter[symbol] = hold_count + 1
            return prev_pos

        # Trend hold: when model says exit (desired=0) but trend is still favorable,
        # keep position up to max_hold bars
        if (
            self._trend_follow
            and desired == 0.0
            and prev_pos > 0.0
            and features is not None
            and hold_count < self._max_hold
        ):
            trend_val = features.get(self._trend_indicator)
            if trend_val is not None and trend_val > self._trend_threshold:
                self._hold_counter[symbol] = hold_count + 1
                return prev_pos

        # Holding period expired — allow change
        if desired != prev_pos:
            self._position[symbol] = desired
            self._hold_counter[symbol] = 1
        else:
            self._hold_counter[symbol] = hold_count + 1
        return desired

    def _check_monthly_gate(self, symbol: str, close: float) -> bool:
        """Return True if signal is allowed (close > MA), False if gated."""
        if not self._monthly_gate:
            return True
        w = self._resolve(self._monthly_gate_window, symbol, 480)
        hist = self._close_history.get(symbol)
        if hist is None:
            hist = deque(maxlen=w)
            self._close_history[symbol] = hist
        hist.append(close)
        if len(hist) < w:
            return False  # not enough data yet — conservative: gate off
        ma = sum(hist) / w
        return close > ma

    def enrich(
        self,
        symbol: str,
        ts: Optional[datetime],
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run inference and add ml_score to features dict."""
        if ts is None:
            ts = datetime.utcnow()

        # Track close for monthly gate
        close_val = features.get("close")
        gate_ok = True
        if close_val is not None:
            gate_ok = self._check_monthly_gate(symbol, float(close_val))

        results = self._engine.run(symbol=symbol, ts=ts, features=features)

        for r in results:
            if r.error is not None:
                logger.warning(
                    "Inference error for %s/%s: %s",
                    r.model_name, symbol, r.error,
                )
                if self._metrics is not None:
                    self._metrics.inc_counter(
                        "inference_errors_total",
                        labels={"model": r.model_name, "symbol": symbol},
                    )
                continue
            if r.signal is not None:
                score = r.signal.strength
                if r.signal.side == "short":
                    score = -score
                elif r.signal.side == "flat":
                    score = 0.0

                score = self._apply_constraints(symbol, score, features)

                # Monthly gate: bear regime — run bear model or go flat
                if not gate_ok:
                    if self._bear_model is not None:
                        bear_sig = self._bear_model.predict(
                            symbol=symbol, ts=ts, features=features)
                        if bear_sig is not None and bear_sig.side == "long":
                            # Graded bear scoring by probability thresholds
                            if self._bear_thresholds:
                                prob = bear_sig.strength
                                score = 0.0
                                for thresh, s in self._bear_thresholds:
                                    if prob > thresh:
                                        score = s
                                        break
                            else:
                                score = -1.0
                        else:
                            score = 0.0
                    elif score != 0.0:
                        score = 0.0
                    # Sync hold state on regime switch
                    if score != self._position.get(symbol, 0.0):
                        self._position[symbol] = score
                        self._hold_counter[symbol] = 1

                # Vol-adaptive sizing
                vt = self._resolve(self._vol_target, symbol)
                if score != 0.0 and vt is not None:
                    vf = self._resolve(self._vol_feature, symbol, "atr_norm_14")
                    vol_val = features.get(vf)
                    if vol_val is not None and vol_val > 1e-8:
                        scale = min(vt / float(vol_val), 1.0)
                        score *= scale

                features[self._score_key] = score

                logger.debug(
                    "Inference %s/%s: side=%s strength=%.4f latency=%.1fms",
                    r.model_name, symbol, r.signal.side, r.signal.strength, r.latency_ms,
                )
                if self._metrics is not None:
                    self._metrics.observe_histogram(
                        "inference_latency_seconds",
                        r.latency_ms / 1000.0,
                        labels={"model": r.model_name},
                    )
                    self._metrics.set_gauge(
                        "ml_score",
                        score,
                        labels={"model": r.model_name, "symbol": symbol},
                    )

        return features
