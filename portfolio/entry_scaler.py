"""Entry scaler for AlphaRunner — adaptive sizing, hold times, and risk scaling.

All parameters adapt to current market conditions via vol_ratio (realized_vol / vol_median).
No hardcoded thresholds — everything scales with the market.
"""
from __future__ import annotations

from typing import List


class EntryScaler:
    """Adaptive entry sizing and risk scaling.

    Core principle: vol_ratio = realized_vol / vol_median normalizes all
    parameters to the current volatility regime. When vol is 2x normal,
    hold times double, DD tolerance widens, etc.
    """

    # ── BB Entry Scale ────────────────────────────────────────

    def bb_scale(self, signal: int, closes: List[float], window: int = 12) -> float:
        """Scale entry size by Bollinger-band position.

        Long into oversold → 1.2x. Long into overbought → 0.6x.
        Mirror for shorts. Returns 1.0 if insufficient data.
        """
        if signal == 0 or len(closes) < window + 1:
            return 1.0

        recent = closes[-window:]
        ma = sum(recent) / len(recent)
        std = (sum((c - ma) ** 2 for c in recent) / len(recent)) ** 0.5
        if std <= 0:
            return 1.0

        bb_pos = (closes[-1] - ma) / std

        # Continuous scale function instead of discrete bins.
        # For long: oversold (bb_pos << 0) → boost, overbought (bb_pos >> 0) → reduce.
        # scale = 1.0 - 0.25 × tanh(effective_pos), clamped [0.75, 1.2]
        import math
        effective_pos = bb_pos if signal == 1 else -bb_pos  # mirror for short
        scale = 1.0 - 0.25 * math.tanh(effective_pos)
        return max(0.75, min(1.2, scale))
        return 1.0

    # ── Leverage Scale (DD-aware + vol-aware) ─────────────────

    def leverage_scale(self, drawdown_pct: float, vol_ratio: float = 1.0) -> float:
        """Scale leverage based on drawdown AND current volatility.

        Vol-aware: DD thresholds widen when vol is high (large swings are normal),
        tighten when vol is low (small DD = real trouble).

        Base thresholds (at vol_ratio=1.0): 10%→0.75x, 20%→0.5x, 35%→0.25x
        At vol_ratio=2.0: 20%→0.75x, 40%→0.5x, 70%→0.25x (wider)
        At vol_ratio=0.5: 5%→0.75x, 10%→0.5x, 17.5%→0.25x (tighter)

        Args:
            drawdown_pct: Current drawdown percentage (e.g. 15.0).
            vol_ratio: realized_vol / vol_median. >1 = high vol, <1 = low vol.
        """
        # Clamp vol_ratio to prevent extreme threshold shifting
        vr = max(0.5, min(2.5, vol_ratio))

        # Scale DD thresholds with vol: high vol = larger normal swings
        t1 = 10.0 * vr   # mild DD threshold
        t2 = 20.0 * vr   # moderate DD threshold
        t3 = 35.0 * vr   # severe DD threshold

        if drawdown_pct >= t3:
            return 0.25
        elif drawdown_pct >= t2:
            return 0.50
        elif drawdown_pct >= t1:
            return 0.75
        return 1.0

    # ── Adaptive Hold Times ───────────────────────────────────

    def adaptive_hold(
        self,
        base_min_hold: int,
        base_max_hold: int,
        vol_ratio: float = 1.0,
    ) -> tuple[int, int]:
        """Adapt min/max hold to current volatility.

        Low vol → shorter holds (avoid time decay in flat markets).
        High vol → longer holds (let big moves play out).

        Args:
            base_min_hold: Configured min hold (bars).
            base_max_hold: Configured max hold (bars).
            vol_ratio: realized_vol / vol_median.

        Returns:
            (adapted_min_hold, adapted_max_hold)
        """
        # Clamp vol_ratio to [0.5, 2.0]
        vr = max(0.5, min(2.0, vol_ratio))

        # Scale hold times: sqrt to dampen extreme swings
        # vol_ratio=0.5 → hold × 0.71 (shorter), vol_ratio=2.0 → hold × 1.41 (longer)
        scale = vr ** 0.5

        min_h = max(1, int(base_min_hold * scale))
        max_h = max(min_h + 1, int(base_max_hold * scale))
        return min_h, max_h

    # ── Position Cap by Signal Confidence ─────────────────────

    def confidence_cap_scale(self, z_score: float, base_dz: float) -> float:
        """Scale position cap by signal confidence (|z| / deadzone).

        z just above deadzone → 0.7x cap (weak signal, smaller position).
        z = 2× deadzone → 1.0x (normal).
        z = 3× deadzone → 1.3x (strong conviction, larger position).

        Args:
            z_score: Current z-score (absolute value used).
            base_dz: Deadzone threshold.

        Returns:
            Scale factor for position cap, in [0.7, 1.3].
        """
        if base_dz <= 0:
            return 1.0
        confidence = abs(z_score) / base_dz
        # Linear ramp: 1.0→0.7, 2.0→1.0, 3.0→1.3
        scale = 0.4 + 0.3 * confidence
        return max(0.7, min(1.3, scale))

    # ── Vol-Adaptive Deadzone (kept for backward compat) ──────

    def vol_adaptive_deadzone(
        self,
        base_deadzone: float,
        realized_vol: float,
        vol_median: float,
    ) -> float:
        """Adapt deadzone to current volatility.

        deadzone = base × (realized_vol / vol_median), clamped [0.5x, 2.0x].
        """
        if vol_median <= 0:
            return base_deadzone
        ratio = realized_vol / vol_median
        ratio = max(0.5, min(2.0, ratio))
        return base_deadzone * ratio
