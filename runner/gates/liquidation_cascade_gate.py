# runner/gates/liquidation_cascade_gate.py
"""Liquidation cascade protection gate.

Detects active liquidation cascades and reduces/blocks new entries:
  - liquidation_volume_zscore_24 > 2.0 → scale down to 0.3x
  - oi_acceleration < -2.0 → scale down to 0.5x (rapid OI unwind)
  - Both triggered → block entirely

Also monitors liquidation_cascade_score (|oi_change_pct| × vol_ma_ratio)
as a secondary confirmation signal.

Purpose: Avoid entering during liquidation events where we'd be
providing exit liquidity to forced sellers, leading to adverse fills.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from runner.gate_chain import GateResult

_log = logging.getLogger(__name__)


@dataclass
class LiquidationCascadeConfig:
    """Configuration for liquidation cascade gate."""
    enabled: bool = True

    # Primary thresholds
    liq_zscore_caution: float = 1.5     # start reducing
    liq_zscore_danger: float = 2.0      # aggressive reduction
    liq_zscore_block: float = 3.0       # block new entries

    # OI acceleration thresholds (negative = OI dropping fast)
    oi_accel_caution: float = -1.5      # mild OI unwind
    oi_accel_danger: float = -2.0       # rapid OI unwind

    # Cascade score threshold (composite signal)
    cascade_score_threshold: float = 2.0  # |oi_change| × vol_ratio elevated

    # Scaling factors
    caution_scale: float = 0.5          # mild cascade → half size
    danger_scale: float = 0.3           # active cascade → minimal size
    oi_unwind_scale: float = 0.5        # OI acceleration warning


class LiquidationCascadeGate:
    """Gate: protect against liquidation cascades.

    Reads from context:
      - liquidation_volume_zscore_24: z-score of liquidation volume
      - oi_acceleration: rate of change of OI change
      - liquidation_cascade_score: composite signal
      - liquidation_imbalance: directional liquidation bias

    Scaling logic:
      liq_zscore > 3.0                      → BLOCK
      liq_zscore > 2.0 AND oi_accel < -2.0  → BLOCK
      liq_zscore > 2.0                      → 0.3x
      liq_zscore > 1.5                      → 0.5x
      oi_accel < -2.0 (alone)               → 0.5x
      Otherwise                             → 1.0x
    """

    name = "LiquidationCascade"

    def __init__(self, cfg: LiquidationCascadeConfig | None = None) -> None:
        self._cfg = cfg or LiquidationCascadeConfig()
        self._total_checks = 0
        self._blocked_count = 0
        self._scaled_count = 0

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        if not self._cfg.enabled:
            return GateResult(allowed=True, scale=1.0)

        self._total_checks += 1
        cfg = self._cfg

        liq_zscore = _safe_float(context.get("liquidation_volume_zscore_24"), 0.0)
        oi_accel = _safe_float(context.get("oi_acceleration"), 0.0)
        cascade_score = _safe_float(context.get("liquidation_cascade_score"), 0.0)
        liq_imbalance = _safe_float(context.get("liquidation_imbalance"), 0.0)

        # Get signal direction for directional analysis
        signal = 0
        meta = getattr(ev, "metadata", None) or {}
        if isinstance(meta, dict):
            signal = int(meta.get("signal", 0))
        if signal == 0:
            signal = int(context.get("signal", 0))

        # Extreme liquidation → block
        if liq_zscore >= cfg.liq_zscore_block:
            self._blocked_count += 1
            reason = f"liq_cascade_block zscore={liq_zscore:.2f}"
            _log.warning("LiquidationCascade BLOCK: %s", reason)
            return GateResult(allowed=False, reason=reason)

        # Danger: high liq + OI unwind → block
        if liq_zscore >= cfg.liq_zscore_danger and oi_accel <= cfg.oi_accel_danger:
            self._blocked_count += 1
            reason = (
                f"liq_cascade_block zscore={liq_zscore:.2f} "
                f"oi_accel={oi_accel:.2f}"
            )
            _log.warning("LiquidationCascade BLOCK: %s", reason)
            return GateResult(allowed=False, reason=reason)

        # Determine scale
        scale = 1.0
        reasons = []

        if liq_zscore >= cfg.liq_zscore_danger:
            scale = min(scale, cfg.danger_scale)
            reasons.append(f"liq_danger={liq_zscore:.2f}")
        elif liq_zscore >= cfg.liq_zscore_caution:
            scale = min(scale, cfg.caution_scale)
            reasons.append(f"liq_caution={liq_zscore:.2f}")

        if oi_accel <= cfg.oi_accel_danger:
            scale = min(scale, cfg.oi_unwind_scale)
            reasons.append(f"oi_unwind={oi_accel:.2f}")
        elif oi_accel <= cfg.oi_accel_caution:
            scale = min(scale, 0.7)
            reasons.append(f"oi_caution={oi_accel:.2f}")

        # Cascade score as additional confirmation
        if cascade_score >= cfg.cascade_score_threshold:
            scale = min(scale, cfg.caution_scale)
            reasons.append(f"cascade_score={cascade_score:.2f}")

        # Directional analysis: if entering same side as liquidation flush,
        # that's contrarian (potentially good), slight boost
        if signal != 0 and abs(liq_imbalance) > 0.3 and scale == 1.0:
            # Liquidation imbalance: positive = more buy liquidations (shorts squeezed)
            # If we're going long and shorts are being squeezed, that's aligned
            contrarian = (
                (signal > 0 and liq_imbalance < -0.3) or  # long after sell liquidations
                (signal < 0 and liq_imbalance > 0.3)       # short after buy liquidations
            )
            if contrarian:
                scale = 1.1  # slight contrarian boost
                reasons.append(f"contrarian_liq imbal={liq_imbalance:.2f}")

        if scale < 1.0:
            self._scaled_count += 1

        reason = "; ".join(reasons) if reasons else ""
        _log.debug(
            "LiquidationCascade: zscore=%.2f oi_accel=%.2f cascade=%.2f → scale=%.2f",
            liq_zscore, oi_accel, cascade_score, scale,
        )
        return GateResult(allowed=True, scale=scale, reason=reason)

    @property
    def stats(self) -> dict:
        return {
            "total_checks": self._total_checks,
            "blocked": self._blocked_count,
            "scaled": self._scaled_count,
            "block_rate": self._blocked_count / max(self._total_checks, 1),
        }


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert to float, returning default for NaN/None/missing."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if f != f else f  # NaN check
    except (ValueError, TypeError):
        return default
