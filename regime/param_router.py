from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .composite import CompositeRegimeLabel

try:
    from _quant_hotpath import (
        RustRegimeParamRouter as _RustRouter,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass(frozen=True, slots=True)
class RegimeParams:
    """Trading parameters for a given regime."""

    deadzone: float
    min_hold: int
    max_hold: int
    position_scale: float  # 0.0 to 1.0


# Key: (trend, vol).  "*" matches any value in that dimension.
DEFAULT_PARAMS: Dict[Tuple[str, str], RegimeParams] = {
    # Strong trend + low/normal vol: aggressive
    ("strong_up", "low_vol"): RegimeParams(0.3, 18, 60, 1.0),
    ("strong_up", "normal_vol"): RegimeParams(0.5, 18, 60, 0.8),
    ("strong_down", "low_vol"): RegimeParams(0.3, 18, 60, 1.0),
    ("strong_down", "normal_vol"): RegimeParams(0.5, 18, 60, 0.8),
    # Weak trend + normal vol: moderate
    ("weak_up", "normal_vol"): RegimeParams(0.8, 24, 96, 0.6),
    ("weak_down", "normal_vol"): RegimeParams(0.8, 24, 96, 0.6),
    # Weak trend + low vol
    ("weak_up", "low_vol"): RegimeParams(0.5, 24, 96, 0.7),
    ("weak_down", "low_vol"): RegimeParams(0.5, 24, 96, 0.7),
    # Ranging
    ("ranging", "low_vol"): RegimeParams(1.0, 24, 96, 0.5),
    ("ranging", "normal_vol"): RegimeParams(1.2, 24, 96, 0.4),
    ("ranging", "high_vol"): RegimeParams(1.5, 24, 48, 0.3),
    # Crisis: minimal trading (wildcard trend)
    ("*", "crisis"): RegimeParams(2.5, 48, 96, 0.1),
    # High vol fallback (wildcard trend)
    ("*", "high_vol"): RegimeParams(1.5, 24, 60, 0.3),
}

# Absolute fallback
_FALLBACK = RegimeParams(1.0, 24, 96, 0.5)


class RegimeParamRouter:
    """Maps CompositeRegimeLabel to trading parameters.

    Lookup order:
      1. Exact match (trend, vol)
      2. Wildcard match ("*", vol)
      3. Wildcard match (trend, "*")
      4. Fallback defaults

    When no custom params/fallback are injected and Rust is available,
    delegates to RustRegimeParamRouter for the fast path.
    """

    def __init__(
        self,
        params: Dict[Tuple[str, str], RegimeParams] | None = None,
        fallback: RegimeParams | None = None,
    ) -> None:
        self.params = dict(params) if params is not None else dict(DEFAULT_PARAMS)
        self.fallback = fallback or _FALLBACK
        # Use Rust when no custom params injected
        if _HAS_RUST and params is None and fallback is None:
            self._rust_router: object | None = _RustRouter()
        else:
            self._rust_router = None

    def route(self, regime: CompositeRegimeLabel) -> RegimeParams:
        """Return trading parameters for the given regime."""
        if self._rust_router is not None:
            rust_p = self._rust_router.route(regime.trend, regime.vol)  # type: ignore[union-attr]
            return RegimeParams(
                deadzone=rust_p.deadzone,
                min_hold=rust_p.min_hold,
                max_hold=rust_p.max_hold,
                position_scale=rust_p.position_scale,
            )

        # Python fallback path (used when custom params are injected)
        # 1. Exact match
        key = (regime.trend, regime.vol)
        if key in self.params:
            return self.params[key]

        # 2. Wildcard on trend
        wild_trend = ("*", regime.vol)
        if wild_trend in self.params:
            return self.params[wild_trend]

        # 3. Wildcard on vol
        wild_vol = (regime.trend, "*")
        if wild_vol in self.params:
            return self.params[wild_vol]

        # 4. Fallback
        return self.fallback
