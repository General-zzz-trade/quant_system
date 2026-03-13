"""Regime Gate — scale or skip positions based on market regime.

Gate purpose: Feature-based gate — monitors "is the market in a tradeable regime?"
This is intentionally separate from AlphaHealthMonitor (monitoring/alpha_health.py),
which monitors "is the model still predictive?" using IC tracking.
Both gates may independently scale or block positions:
  - RegimeGate: scales down in unfavorable regimes (high vol, low trend)
  - AlphaHealthMonitor: reduces/halts when rolling IC degrades

Uses existing features (bb_width_20, vol_of_vol, adx_14 if available)
to classify regime and output a position scale factor.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Tuple


class RegimeGate:
    """Classifies market regime and returns position scale factor.

    Uses features already computed by RustFeatureEngine, so no extra cost.
    If adx_14 is available, uses ADX-based regime detection.
    Falls back to bb_width + vol_of_vol otherwise.

    Parameters
    ----------
    config : RegimeGateConfig
        Regime gate configuration from V11Config.
    """

    def __init__(self, config):
        from alpha.v11_config import RegimeGateConfig
        self._config: RegimeGateConfig = config
        # Running percentile buffers for bb_width and vol_of_vol
        self._bb_width_buf: deque = deque(maxlen=720)
        self._vol_of_vol_buf: deque = deque(maxlen=720)

    @property
    def _lookback(self) -> int:
        return self._bb_width_buf.maxlen or 720

    def checkpoint(self) -> dict:
        """Serialize buffer state for persistence."""
        return {
            "bb_width_buf": list(self._bb_width_buf),
            "vol_of_vol_buf": list(self._vol_of_vol_buf),
        }

    def restore(self, data: dict) -> None:
        """Restore buffer state from checkpoint."""
        maxlen = self._lookback
        self._bb_width_buf = deque(data.get("bb_width_buf", []), maxlen=maxlen)
        self._vol_of_vol_buf = deque(data.get("vol_of_vol_buf", []), maxlen=maxlen)

    def evaluate(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Classify regime and return (regime_label, position_scale).

        regime_label: "trending" | "ranging" | "ranging_high_vol" | "normal"
        position_scale: 0.0-1.0 multiplier on position size
        """
        if not self._config.enabled:
            return "normal", 1.0

        cfg = self._config

        # Update running buffers
        bb_width = features.get("bb_width_20", 0.0)
        vol_of_vol = features.get("vol_of_vol", 0.0)
        self._bb_width_buf.append(bb_width)
        self._vol_of_vol_buf.append(vol_of_vol)

        # Check ADX if available
        adx = features.get("adx_14")

        if adx is not None:
            # ADX-based regime detection
            if adx > cfg.adx_trend_threshold:
                return "trending", 1.0
            elif adx < cfg.adx_range_threshold:
                # Ranging — check if also high volatility
                if self._is_high_vol(bb_width, vol_of_vol):
                    return self._ranging_high_vol_result()
                return "ranging", 1.0
            else:
                return "normal", 1.0
        else:
            # Fallback: bb_width + vol_of_vol only
            if self._is_high_vol(bb_width, vol_of_vol):
                return self._ranging_high_vol_result()
            return "normal", 1.0

    def _is_high_vol(self, bb_width: float, vol_of_vol: float) -> bool:
        """Check if current bb_width or vol_of_vol is in the high percentile."""
        cfg = self._config

        bb_high = False
        if len(self._bb_width_buf) >= 100:
            sorted_bb = sorted(self._bb_width_buf)
            idx = int(len(sorted_bb) * cfg.bb_width_high_pct / 100)
            idx = min(idx, len(sorted_bb) - 1)
            bb_high = bb_width >= sorted_bb[idx]

        vov_high = False
        if len(self._vol_of_vol_buf) >= 100:
            sorted_vov = sorted(self._vol_of_vol_buf)
            idx = int(len(sorted_vov) * cfg.vol_of_vol_high_pct / 100)
            idx = min(idx, len(sorted_vov) - 1)
            vov_high = vol_of_vol >= sorted_vov[idx]

        return bb_high or vov_high

    def _ranging_high_vol_result(self) -> Tuple[str, float]:
        """Return result for ranging + high volatility regime."""
        cfg = self._config
        if cfg.ranging_high_vol_action == "skip":
            return "ranging_high_vol", 0.0
        else:  # "reduce"
            return "ranging_high_vol", cfg.reduce_factor
