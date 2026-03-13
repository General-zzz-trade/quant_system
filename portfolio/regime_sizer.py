"""Regime-aware position sizer.

Converts discrete regime labels into continuous position scale factors.
Integrates with AlphaHealthMonitor pattern: returns 0.0-1.0 scale.

Scale mapping:
  - Low vol regime:   1.0x (full position)
  - Medium vol:       0.5-0.8x
  - High vol:         0.2-0.3x (reduced, not zero)
  - Crash (breaker):  0.0x (handled by DrawdownCircuitBreaker)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegimeSizerConfig:
    """Configuration for regime-based position sizing."""
    # Scale factors per volatility regime
    low_vol_scale: float = 1.0
    mid_vol_scale: float = 0.6
    high_vol_scale: float = 0.25

    # Smoothing: blend new scale with previous to avoid whipsawing
    smoothing_alpha: float = 0.3  # EMA blend (0=no smooth, 1=instant)

    # Minimum scale (never go below this unless drawdown breaker kills)
    min_scale: float = 0.1


@dataclass
class RegimePositionSizer:
    """Continuous regime-based position sizing.

    Unlike the binary regime gate (allow/block), this sizer outputs
    a continuous scale factor (0.0-1.0) based on the current volatility
    regime score.

    Usage:
        sizer = RegimePositionSizer()
        sizer.update(symbol, regime_label)
        scale = sizer.position_scale(symbol)  # 0.0-1.0
    """

    config: RegimeSizerConfig = field(default_factory=RegimeSizerConfig)
    _current_scale: Dict[str, float] = field(default_factory=dict)
    _last_regime: Dict[str, str] = field(default_factory=dict)

    def update(self, symbol: str, regime_label: Any) -> float:
        """Update regime state and return new position scale.

        Args:
            symbol: Trading symbol
            regime_label: RegimeLabel from VolatilityRegimeDetector
                         (has .value and .score attributes)

        Returns:
            Position scale factor (0.0-1.0)
        """
        if regime_label is None:
            return self._current_scale.get(symbol, 1.0)

        value = getattr(regime_label, "value", "mid")
        score = float(getattr(regime_label, "score", 0.5))

        # Map regime value to base scale
        if value == "low":
            raw_scale = self.config.low_vol_scale
        elif value == "high":
            raw_scale = self.config.high_vol_scale
        else:  # "mid"
            # Interpolate within mid range based on score
            # score 0.2 (low boundary) -> mid_vol_scale upper end
            # score 1.0 (high boundary) -> mid_vol_scale lower end
            t = max(0.0, min(1.0, (score - 0.2) / 0.8))
            raw_scale = self.config.mid_vol_scale * (1.0 - t * 0.3)

        # Clamp
        raw_scale = max(self.config.min_scale, min(1.0, raw_scale))

        # Smooth with EMA to prevent whipsawing
        prev = self._current_scale.get(symbol, raw_scale)
        alpha = self.config.smoothing_alpha
        smoothed = alpha * raw_scale + (1.0 - alpha) * prev

        self._current_scale[symbol] = smoothed
        self._last_regime[symbol] = value

        return smoothed

    def position_scale(self, symbol: str) -> float:
        """Get current position scale for a symbol."""
        return self._current_scale.get(symbol, 1.0)

    def get_status(self) -> Dict[str, Any]:
        """Return current state for health endpoint."""
        return {
            "scales": dict(self._current_scale),
            "regimes": dict(self._last_regime),
        }
