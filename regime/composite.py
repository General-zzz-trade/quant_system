from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .base import RegimeLabel

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

    Delegates to RustCompositeRegimeDetector (default path).
    When custom sub-detectors are injected, falls back to Python.
    """

    name = "composite"

    def __init__(
        self,
        *,
        vol_detector: object | None = None,
        trend_detector: object | None = None,
    ) -> None:
        from .trend import TrendRegimeDetector
        from .volatility import VolatilityRegimeDetector

        self.vol_detector = vol_detector or VolatilityRegimeDetector()
        self.trend_detector = trend_detector or TrendRegimeDetector()
        if _HAS_RUST and vol_detector is None and trend_detector is None:
            self._rust_detector: Any = _RustCompositeDetector()
        else:
            self._rust_detector = None

    def detect(
        self, *, symbol: str, ts: datetime, features: Dict[str, Any]
    ) -> Optional[RegimeLabel]:
        if self._rust_detector is not None:
            result = self._detect_rust(ts, features)
            if result is not None:
                return result
            # Rust returned None (insufficient bars or missing features) — fall through
        return self._detect_python(symbol, ts, features)

    def _detect_rust(
        self, ts: datetime, features: Dict[str, Any]
    ) -> Optional[RegimeLabel]:
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
        composite = CompositeRegimeLabel(
            vol=result.vol_label, trend=result.trend_label
        )
        return RegimeLabel(
            name=self.name, ts=ts, value=result.value, score=result.score,
            meta={
                "composite": composite,
                "is_favorable": result.is_favorable,
                "is_crisis": result.is_crisis,
                "vol_label": result.vol_label,
                "trend_label": result.trend_label,
            },
        )

    def _detect_python(
        self, symbol: str, ts: datetime, features: Dict[str, Any]
    ) -> Optional[RegimeLabel]:
        """Python fallback — used when custom detectors injected or Rust returns None."""
        vol_label = self.vol_detector.detect(symbol=symbol, ts=ts, features=features)
        trend_label = self.trend_detector.detect(symbol=symbol, ts=ts, features=features)

        if vol_label is None and trend_label is None:
            return None

        vol_val = vol_label.value if vol_label else "normal_vol"
        trend_val = trend_label.value if trend_label else "ranging"
        composite = CompositeRegimeLabel(vol=vol_val, trend=trend_val)

        vol_score = vol_label.score if vol_label else 0.5
        trend_score = trend_label.score if trend_label else 0.5
        score = (0.8 * vol_score + 0.2 * trend_score) if composite.is_crisis else (0.5 * vol_score + 0.5 * trend_score)

        return RegimeLabel(
            name=self.name, ts=ts, value=f"{trend_val}|{vol_val}", score=score,
            meta={"composite": composite, "vol_label": vol_label,
                  "trend_label": trend_label, "is_favorable": composite.is_favorable,
                  "is_crisis": composite.is_crisis},
        )
