# decision/regime_policy.py
"""RegimePolicy — decides whether to allow trading based on regime labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Tuple

from regime.base import RegimeLabel


@dataclass(frozen=True, slots=True)
class RegimePolicy:
    """Policy that blocks trading in unfavorable regime combinations.

    Default behavior: blocks when volatility=high AND trend=flat.
    Customizable via blocked_regimes for simple per-detector blocklist.
    """

    # Simple blocklist: detector_name -> set of blocked values
    blocked_regimes: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: {},
    )

    # Whether to apply the default high-vol + flat-trend block
    block_high_vol_flat_trend: bool = True

    def allow(self, labels: Sequence[RegimeLabel]) -> Tuple[bool, str]:
        """Check if trading is allowed given current regime labels.

        Returns (allowed, reason).
        """
        label_map = {label.name: label.value for label in labels}

        # Check simple blocklist
        for name, blocked_values in self.blocked_regimes.items():
            val = label_map.get(name)
            if val is not None and val in blocked_values:
                return False, f"regime_{name}={val}_blocked"

        # Composite detector: block on crisis
        composite_val = label_map.get("composite")
        if composite_val is not None and "crisis" in composite_val:
            return False, "composite_crisis_blocked"

        # Default combination block: high vol + flat trend
        if self.block_high_vol_flat_trend:
            vol = label_map.get("volatility")
            trend = label_map.get("trend")
            if vol == "high" and trend == "flat":
                return False, "high_vol_flat_trend"

        return True, "ok"
