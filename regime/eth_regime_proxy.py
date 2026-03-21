# regime/eth_regime_proxy.py
"""ETH Regime-Adaptive using BTC regime as proxy.

BTC Regime-Adaptive achieved Sharpe 5.52 (+172% over baseline).
ETH with its own regime detection was worse than fixed params.

Hypothesis: BTC regime labels are more reliable (BTC leads crypto market),
and ETH can benefit by using BTC's regime classification while keeping
ETH-specific parameter mapping.

This module:
  1. Takes BTC regime labels from CompositeRegimeDetector
  2. Maps them to ETH-optimized parameters (different from BTC mapping)
  3. Provides position_scale() for gate chain integration

Key differences from BTC param_router:
  - ETH uses tighter deadzones (faster signal, shorter hold)
  - ETH ranging regime uses higher scale (ETH mean-reverts better)
  - ETH crisis response less aggressive (ETH recovers faster)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from regime.composite import CompositeRegimeDetector, CompositeRegimeLabel
from regime.param_router import RegimeParams, RegimeParamRouter

_log = logging.getLogger(__name__)

# ETH-specific parameter mapping (using BTC regime labels)
# Key: (trend, vol) from BTC regime
ETH_REGIME_PARAMS: Dict[Tuple[str, str], RegimeParams] = {
    # Strong BTC trend + low vol: ETH follows BTC strongly → aggressive
    ("strong_up", "low_vol"): RegimeParams(0.3, 14, 48, 1.0),
    ("strong_up", "normal_vol"): RegimeParams(0.4, 14, 48, 0.9),
    ("strong_down", "low_vol"): RegimeParams(0.3, 14, 48, 1.0),
    ("strong_down", "normal_vol"): RegimeParams(0.4, 14, 48, 0.9),
    # Weak BTC trend: ETH may decouple → moderate
    ("weak_up", "normal_vol"): RegimeParams(0.5, 18, 60, 0.7),
    ("weak_down", "normal_vol"): RegimeParams(0.5, 18, 60, 0.7),
    ("weak_up", "low_vol"): RegimeParams(0.4, 18, 60, 0.8),
    ("weak_down", "low_vol"): RegimeParams(0.4, 18, 60, 0.8),
    # BTC ranging: ETH mean-reverts well → keep trading but tighter stops
    ("ranging", "low_vol"): RegimeParams(0.6, 18, 48, 0.6),
    ("ranging", "normal_vol"): RegimeParams(0.8, 18, 48, 0.5),
    ("ranging", "high_vol"): RegimeParams(1.0, 24, 48, 0.4),
    # Crisis: reduce but don't fully stop (ETH recovers faster than BTC)
    ("*", "crisis"): RegimeParams(1.5, 24, 48, 0.2),
    ("*", "high_vol"): RegimeParams(1.0, 18, 48, 0.4),
}

ETH_FALLBACK = RegimeParams(0.5, 18, 60, 0.6)


@dataclass
class ETHRegimeProxyConfig:
    """Configuration for ETH regime proxy."""
    enabled: bool = True
    btc_symbol: str = "BTCUSDT"
    # Use BTC regime labels but ETH param mapping
    use_btc_regime: bool = True
    # Fallback to ETH's own vol if BTC regime unavailable
    vol_lookback: int = 100
    atr_lookback: int = 14


class ETHRegimeProxy:
    """Use BTC regime labels to route ETH trading parameters.

    Wraps CompositeRegimeDetector for BTC and maps to ETH-specific
    RegimeParams via ETH_REGIME_PARAMS table.

    Integration:
      - Create one CompositeRegimeDetector for BTC
      - Feed BTC bars to it (vol, adx, close_vs_ma)
      - This module takes the BTC regime label and returns ETH params

    Usage:
        proxy = ETHRegimeProxy()
        # On each BTC bar:
        proxy.update_btc_regime(btc_vol, btc_adx, btc_close_vs_ma)
        # On each ETH trade decision:
        params = proxy.get_eth_params()
        scale = proxy.position_scale()
    """

    def __init__(
        self,
        cfg: ETHRegimeProxyConfig | None = None,
        btc_regime_detector: CompositeRegimeDetector | None = None,
    ) -> None:
        self._cfg = cfg or ETHRegimeProxyConfig()
        self._btc_detector = btc_regime_detector or CompositeRegimeDetector()
        self._router = RegimeParamRouter(
            params=ETH_REGIME_PARAMS,
            fallback=ETH_FALLBACK,
        )
        self._current_label: Optional[CompositeRegimeLabel] = None
        self._current_params: RegimeParams = ETH_FALLBACK
        self._update_count = 0

    def update_btc_regime(
        self,
        vol: float,
        adx: float = 0.0,
        close_vs_ma: float = 0.0,
        ts: datetime | None = None,
    ) -> CompositeRegimeLabel:
        """Update BTC regime state and recalculate ETH params.

        Args:
            vol: BTC ATR or realized vol (atr_norm_14)
            adx: BTC ADX(14)
            close_vs_ma: BTC |close/MA20 - 1|
            ts: Current timestamp (default: now UTC)

        Returns:
            Current BTC regime label
        """
        features = {
            "atr_norm_14": vol,
            "adx_14": adx,
            "close_vs_ma20": close_vs_ma,
            "vol_20": vol,  # alias
        }
        if ts is None:
            ts = datetime.now(timezone.utc)

        label = self._btc_detector.detect(
            symbol=self._cfg.btc_symbol, ts=ts, features=features,
        )
        if label is not None:
            meta = label.meta or {}
            composite = meta.get("composite")
            if isinstance(composite, CompositeRegimeLabel):
                self._current_label = composite
            else:
                # Fallback: parse from label value "trend|vol"
                parts = label.value.split("|")
                if len(parts) == 2:
                    self._current_label = CompositeRegimeLabel(
                        trend=parts[0], vol=parts[1]
                    )

        if self._current_label is not None:
            self._current_params = self._router.route(self._current_label)

        self._update_count += 1

        if self._update_count % 100 == 0:
            _log.info(
                "ETHRegimeProxy: BTC regime=%s/%s → ETH params dz=%.1f mh=%d scale=%.1f",
                self._current_label.trend if self._current_label else "?",
                self._current_label.vol if self._current_label else "?",
                self._current_params.deadzone, self._current_params.min_hold,
                self._current_params.position_scale,
            )
        return self._current_label

    def get_eth_params(self) -> RegimeParams:
        """Get current ETH trading parameters based on BTC regime."""
        return self._current_params

    def position_scale(self, symbol: str = "") -> float:
        """Position scale for gate chain integration."""
        if not self._cfg.enabled:
            return 1.0
        return self._current_params.position_scale

    @property
    def current_regime(self) -> Optional[CompositeRegimeLabel]:
        return self._current_label

    @property
    def stats(self) -> dict:
        return {
            "enabled": self._cfg.enabled,
            "updates": self._update_count,
            "current_regime": (
                f"{self._current_label.trend}/{self._current_label.vol}"
                if self._current_label else "none"
            ),
            "current_scale": self._current_params.position_scale,
            "current_deadzone": self._current_params.deadzone,
        }
