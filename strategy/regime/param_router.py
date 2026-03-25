from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from strategy.regime.composite import CompositeRegimeLabel

try:
    from _quant_hotpath import RustRegimeParamRouter as _RustRouter
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
# Kept as module-level export for tests/research; Rust has identical table.
DEFAULT_PARAMS: Dict[Tuple[str, str], RegimeParams] = {
    ("strong_up", "low_vol"): RegimeParams(0.3, 18, 60, 1.0),
    ("strong_up", "normal_vol"): RegimeParams(0.5, 18, 60, 0.8),
    ("strong_down", "low_vol"): RegimeParams(0.3, 18, 60, 1.0),
    ("strong_down", "normal_vol"): RegimeParams(0.5, 18, 60, 0.8),
    ("weak_up", "normal_vol"): RegimeParams(0.8, 24, 96, 0.6),
    ("weak_down", "normal_vol"): RegimeParams(0.8, 24, 96, 0.6),
    ("weak_up", "low_vol"): RegimeParams(0.5, 24, 96, 0.7),
    ("weak_down", "low_vol"): RegimeParams(0.5, 24, 96, 0.7),
    ("ranging", "low_vol"): RegimeParams(0.6, 18, 72, 0.6),
    ("ranging", "normal_vol"): RegimeParams(0.8, 18, 72, 0.6),
    ("ranging", "high_vol"): RegimeParams(1.2, 24, 48, 0.4),
    ("*", "crisis"): RegimeParams(2.5, 48, 96, 0.1),
    ("*", "high_vol"): RegimeParams(1.5, 24, 60, 0.3),
}

_FALLBACK = RegimeParams(1.0, 24, 96, 0.5)


class RegimeParamRouter:
    """Maps CompositeRegimeLabel to trading parameters.

    Delegates to RustRegimeParamRouter (default path).
    Falls back to Python lookup when custom params are injected.
    """

    def __init__(
        self,
        params: Dict[Tuple[str, str], RegimeParams] | None = None,
        fallback: RegimeParams | None = None,
    ) -> None:
        self._custom_params = dict(params) if params is not None else None
        self.fallback = fallback or _FALLBACK
        if _HAS_RUST and params is None and fallback is None:
            self._rust_router: object | None = _RustRouter()
        else:
            self._rust_router = None

    def route(self, regime: CompositeRegimeLabel) -> RegimeParams:
        """Return trading parameters for the given regime."""
        if self._rust_router is not None:
            rust_p = self._rust_router.route(regime.trend, regime.vol)  # type: ignore[union-attr]
            return RegimeParams(
                deadzone=rust_p.deadzone, min_hold=rust_p.min_hold,
                max_hold=rust_p.max_hold, position_scale=rust_p.position_scale,
            )
        # Python fallback (custom params injected)
        params = self._custom_params or {}
        for key in [(regime.trend, regime.vol), ("*", regime.vol), (regime.trend, "*")]:
            if key in params:
                return params[key]
        return self.fallback
