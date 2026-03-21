# alpha/dynamic_horizon.py
"""Dynamic horizon selection based on market regime (Tier 1b).

Instead of fixed h=24 for all conditions:
  - High volatility / trending: short horizon (h=12) — capture fast moves
  - Low volatility / ranging: long horizon (h=48) — avoid noise
  - Normal: default horizon (h=24)

Regime classification uses ATR percentile and ADX trend strength,
matching the existing CompositeRegimeDetector logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class HorizonConfig:
    """Horizon selection parameters."""
    # Available horizons (must match trained models)
    horizons: tuple[int, ...] = (12, 24)

    # Regime thresholds
    high_vol_percentile: float = 75.0   # vol > P75 → short horizon
    low_vol_percentile: float = 25.0    # vol < P25 → long horizon
    trend_adx_threshold: float = 25.0   # ADX > 25 → trending

    # Horizon mapping
    trending_horizon: int = 12          # fast-moving market
    ranging_horizon: int = 24           # slow, mean-reverting
    default_horizon: int = 24           # fallback

    # Vol lookback for percentile
    vol_lookback: int = 500             # bars for vol percentile


class DynamicHorizonSelector:
    """Select prediction horizon based on current market regime.

    Uses vol percentile + ADX to classify regime, then selects
    the horizon that matches. If the selected horizon has no
    trained model, falls back to nearest available.
    """

    def __init__(self, cfg: HorizonConfig | None = None) -> None:
        self._cfg = cfg or HorizonConfig()
        self._vol_history: list[float] = []
        self._last_horizon: int = self._cfg.default_horizon
        self._regime_label: str = "normal"

    def update(
        self,
        vol: float,
        adx: float = 0.0,
        close_vs_ma: float = 0.0,
    ) -> int:
        """Update regime state and return selected horizon.

        Args:
            vol: Current realized vol (e.g. ATR as % of price).
            adx: ADX(14) value (0-100).
            close_vs_ma: |close/MA20 - 1|, measures trend displacement.

        Returns:
            Selected horizon (e.g., 12 or 24).
        """
        cfg = self._cfg

        # Track vol history for percentile
        self._vol_history.append(vol)
        if len(self._vol_history) > cfg.vol_lookback:
            self._vol_history = self._vol_history[-cfg.vol_lookback:]

        # Classify regime
        if len(self._vol_history) < 50:
            self._regime_label = "warmup"
            self._last_horizon = cfg.default_horizon
            return self._last_horizon

        vol_arr = np.array(self._vol_history)
        vol_pct = _percentile_rank(vol, vol_arr)

        is_high_vol = vol_pct > cfg.high_vol_percentile
        is_low_vol = vol_pct < cfg.low_vol_percentile
        is_trending = adx > cfg.trend_adx_threshold or abs(close_vs_ma) > 0.02

        if is_high_vol and is_trending:
            # Strong trend + high vol → short horizon
            self._regime_label = "trend_vol"
            horizon = cfg.trending_horizon
        elif is_high_vol and not is_trending:
            # High vol but no trend → choppy, use default
            self._regime_label = "high_vol_flat"
            horizon = cfg.default_horizon
        elif is_low_vol:
            # Low vol → ranging, use longer horizon
            self._regime_label = "ranging"
            horizon = cfg.ranging_horizon
        elif is_trending:
            # Normal vol + trend → short horizon
            self._regime_label = "trending"
            horizon = cfg.trending_horizon
        else:
            self._regime_label = "normal"
            horizon = cfg.default_horizon

        # Snap to nearest available horizon
        horizon = _nearest(horizon, cfg.horizons)
        self._last_horizon = horizon
        return horizon

    @property
    def regime_label(self) -> str:
        return self._regime_label

    @property
    def current_horizon(self) -> int:
        return self._last_horizon

    def get_horizon_weights(self) -> Dict[int, float]:
        """Return soft weights for multi-horizon ensemble.

        Instead of hard switching, blend horizons with regime-based weights:
          trending: h=12 weight=0.7, h=24 weight=0.3
          ranging:  h=12 weight=0.3, h=24 weight=0.7
          normal:   equal weights
        """
        cfg = self._cfg
        h_short = min(cfg.horizons) if cfg.horizons else 12
        h_long = max(cfg.horizons) if cfg.horizons else 24

        if self._regime_label in ("trend_vol", "trending"):
            return {h_short: 0.7, h_long: 0.3}
        elif self._regime_label in ("ranging", "high_vol_flat"):
            return {h_short: 0.3, h_long: 0.7}
        else:
            n = len(cfg.horizons)
            return {h: 1.0 / n for h in cfg.horizons}


def _percentile_rank(value: float, arr: np.ndarray) -> float:
    """Percentile rank of value in array (0-100)."""
    return float(np.sum(arr <= value) / len(arr) * 100)


def _nearest(target: int, candidates: Sequence[int]) -> int:
    """Find nearest value in candidates to target."""
    if not candidates:
        return target
    return min(candidates, key=lambda x: abs(x - target))
