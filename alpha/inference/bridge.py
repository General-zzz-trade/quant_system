# alpha/inference/bridge.py
"""LiveInferenceBridge — connects InferenceEngine to the feature pipeline."""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Union

from _quant_hotpath import RustInferenceBridge as _RustBridge
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
        short_model: Optional[AlphaModel] = None,
        short_score_key: str = "ml_short_score",
        vol_target: Union[None, float, Dict[str, Optional[float]]] = None,
        vol_feature: Union[str, Dict[str, str]] = "atr_norm_14",
        zscore_window: int = 720,
        zscore_warmup: int = 168,
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
        self._short_model = short_model
        self._short_score_key = short_score_key
        self._vol_target = vol_target
        self._vol_feature = vol_feature
        self._zscore_window = zscore_window
        self._zscore_warmup = zscore_warmup
        # Rust kernel for per-symbol signal processing state
        self._rust = _RustBridge(zscore_window, zscore_warmup, 480)

    def checkpoint(self) -> dict:
        """Serialize bridge state for persistence."""
        return dict(self._rust.checkpoint())

    def restore(self, data: dict) -> None:
        """Restore bridge state from checkpoint."""
        self._rust.restore(data)
        logger.info("Bridge state restored")

    def update_models(self, models: Sequence[AlphaModel]) -> None:
        self._engine.set_models(models)
        self._rust.reset()
        logger.info("Models hot-swapped: %d model(s)", len(models))

    @staticmethod
    def _resolve(param, symbol: str, default=None):
        """Resolve a scalar-or-dict parameter to a per-symbol value."""
        if isinstance(param, dict):
            return param.get(symbol, default)
        return param if param is not None else default

    def _apply_constraints(
        self, symbol: str, raw_score: float, features: Optional[Dict[str, Any]] = None,
        ts: Optional[datetime] = None,
    ) -> float:
        """Apply z-score normalization, long_only clip, discretization, min_hold, and trend_hold."""
        min_hold = self._min_hold_bars.get(symbol, 0)
        if not min_hold:
            return raw_score

        hour_key = int((ts or datetime.now(timezone.utc)).timestamp()) // 3600
        dz = self._resolve(self._deadzone, symbol, 0.5)
        long_only = symbol in self._long_only_symbols

        # Trend indicator value
        trend_val = float("nan")
        if self._trend_follow and features is not None:
            tv = features.get(self._trend_indicator)
            if tv is not None:
                trend_val = float(tv)

        return self._rust.apply_constraints(
            symbol, raw_score, hour_key, dz, min_hold, long_only,
            self._trend_follow, trend_val, self._trend_threshold, self._max_hold,
        )

    def _check_monthly_gate(self, symbol: str, close: float, ts: Optional[datetime] = None) -> bool:
        """Return True if signal is allowed (close > MA), False if gated."""
        if not self._monthly_gate:
            return True
        w = self._resolve(self._monthly_gate_window, symbol, 480)
        hour_key = int((ts or datetime.now(timezone.utc)).timestamp()) // 3600
        return self._rust.check_monthly_gate(symbol, close, hour_key, w)

    def enrich(
        self,
        symbol: str,
        ts: Optional[datetime],
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run inference and add ml_score to features dict."""
        if ts is None:
            ts = datetime.now(timezone.utc)

        # Track close for monthly gate (hourly resolution)
        close_val = features.get("close")
        gate_ok = True
        if close_val is not None:
            gate_ok = self._check_monthly_gate(symbol, float(close_val), ts)

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

                score = self._apply_constraints(symbol, score, features, ts)

                # Monthly gate: bear regime — run bear model or go flat
                if not gate_ok:
                    if self._bear_model is not None:
                        # NaN guard: skip bear model if features contain NaN
                        _has_nan = any(
                            isinstance(v, float) and math.isnan(v)
                            for v in features.values()
                        )
                        if _has_nan:
                            logger.warning("NaN in features for %s, skipping bear model", symbol)
                            score = 0.0
                        else:
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
                    if score != self._rust.get_position(symbol):
                        self._rust.set_position(symbol, score, 1)

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
                    "Inference %s/%s: side=%s strength=%.4f score=%.4f latency=%.1fms",
                    r.model_name, symbol, r.signal.side, r.signal.strength, score, r.latency_ms,
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

        # ── Independent short model (runs unconditionally, no regime gate) ──
        if self._short_model is not None:
            _has_nan = any(
                isinstance(v, float) and math.isnan(v)
                for v in features.values()
            )
            if _has_nan:
                logger.warning("NaN in features for %s, skipping short model", symbol)
                features[self._short_score_key] = 0.0
            else:
                short_sig = self._short_model.predict(
                    symbol=symbol, ts=ts, features=features)
                short_score = 0.0
                if short_sig is not None and short_sig.strength is not None:
                    raw = short_sig.strength
                    if short_sig.side == "short":
                        raw = -raw
                    elif short_sig.side == "flat":
                        raw = 0.0

                    min_hold = self._min_hold_bars.get(symbol, 0)
                    hour_key = int(ts.timestamp()) // 3600
                    dz = self._resolve(self._deadzone, symbol, 0.5)
                    short_score = self._rust.process_short_signal(
                        symbol, raw, hour_key, dz, min_hold)

                    # Vol-adaptive sizing for short
                    vt = self._resolve(self._vol_target, symbol)
                    if short_score != 0.0 and vt is not None:
                        vf = self._resolve(self._vol_feature, symbol, "atr_norm_14")
                        vol_val = features.get(vf)
                        if vol_val is not None and vol_val > 1e-8:
                            scale = min(vt / float(vol_val), 1.0)
                            short_score *= scale

                features[self._short_score_key] = short_score

                if self._metrics is not None:
                    self._metrics.set_gauge(
                        "ml_short_score",
                        short_score,
                        labels={"model": "short", "symbol": symbol},
                    )

        return features
