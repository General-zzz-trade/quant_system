from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .base import RegimeLabel
from .trend import TrendRegimeDetector
from .volatility import VolatilityRegimeDetector

try:
    from _quant_hotpath import RustCompositeRegimeDetector as _RustCompositeDetector
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass(frozen=True, slots=True)
class CompositeRegimeLabel:
    """Combined volatility + trend regime label."""

    vol: str  # "low_vol" / "normal_vol" / "high_vol" / "crisis"
    trend: str  # "strong_up" / "weak_up" / "ranging" / "weak_down" / "strong_down"

    @property
    def is_favorable(self) -> bool:
        """Strong trend + low/normal vol = favorable for trading."""
        return "strong" in self.trend and self.vol in ("low_vol", "normal_vol")

    @property
    def is_crisis(self) -> bool:
        return self.vol == "crisis"


class CompositeRegimeDetector:
    """Combines volatility and trend detectors into a composite regime.

    Returns a RegimeLabel whose value encodes both dimensions
    (e.g. "strong_up|normal_vol") and whose meta contains
    the full CompositeRegimeLabel.

    When no custom sub-detectors are injected and Rust is available,
    uses RustCompositeRegimeDetector for the fast path.
    """

    name = "composite"

    def __init__(
        self,
        *,
        vol_detector: VolatilityRegimeDetector | None = None,
        trend_detector: TrendRegimeDetector | None = None,
    ) -> None:
        self.vol_detector = vol_detector or VolatilityRegimeDetector()
        self.trend_detector = trend_detector or TrendRegimeDetector()
        # Only use Rust when no custom detectors injected
        if _HAS_RUST and vol_detector is None and trend_detector is None:
            self._rust_detector: Any = _RustCompositeDetector()
        else:
            self._rust_detector = None

    def detect(
        self, *, symbol: str, ts: datetime, features: Dict[str, Any]
    ) -> Optional[RegimeLabel]:
        # Use Rust fast path if available and no custom sub-detectors
        if self._rust_detector is not None:
            feat_f64 = {
                k: float(v)
                for k, v in features.items()
                if v is not None
                and isinstance(v, (int, float))
                and math.isfinite(float(v))
            }
            result = self._rust_detector.detect(feat_f64)
            if result is None:
                return None
            # Map RustRegimeResult back to RegimeLabel
            composite = CompositeRegimeLabel(
                vol=result.vol_label, trend=result.trend_label
            )
            meta: Dict[str, Any] = {
                "composite": composite,
                "is_favorable": result.is_favorable,
                "is_crisis": result.is_crisis,
                "vol_label": result.vol_label,    # plain string (not RegimeLabel) for forward compat
                "trend_label": result.trend_label, # plain string (not RegimeLabel) for forward compat
            }
            return RegimeLabel(
                name=self.name,
                ts=ts,
                value=result.value,
                score=result.score,
                meta=meta,
            )

        # Python fallback path (used when custom detectors are injected)
        vol_label = self.vol_detector.detect(symbol=symbol, ts=ts, features=features)
        trend_label = self.trend_detector.detect(symbol=symbol, ts=ts, features=features)

        if vol_label is None and trend_label is None:
            return None

        vol_val = vol_label.value if vol_label is not None else "normal_vol"
        trend_val = trend_label.value if trend_label is not None else "ranging"

        composite = CompositeRegimeLabel(vol=vol_val, trend=trend_val)

        # Combined score: average of both, weighted toward vol in crisis
        vol_score = vol_label.score if vol_label is not None else 0.5
        trend_score = trend_label.score if trend_label is not None else 0.5

        if composite.is_crisis:
            score = 0.8 * vol_score + 0.2 * trend_score
        else:
            score = 0.5 * vol_score + 0.5 * trend_score

        value = f"{trend_val}|{vol_val}"

        py_meta: Dict[str, Any] = {
            "composite": composite,
            "vol_label": vol_label,
            "trend_label": trend_label,
            "is_favorable": composite.is_favorable,
            "is_crisis": composite.is_crisis,
        }

        return RegimeLabel(name=self.name, ts=ts, value=value, score=score, meta=py_meta)
