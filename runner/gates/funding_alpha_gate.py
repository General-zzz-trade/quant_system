# runner/gates/funding_alpha_gate.py
"""Funding-Alpha joint signal gate for 100x leverage.

When alpha signal direction aligns with funding direction:
  - Full position (scale=1.0) — collect funding + ride alpha
When alpha opposes funding:
  - Reduced position (scale=0.3-0.5) — alpha only, pay funding cost

Funding rate is fetched from bar context or exchange ticker.
At 100x, funding = 1% of capital per 8h settlement → significant edge.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

from runner.gate_chain import GateResult

_log = logging.getLogger(__name__)


class FundingAlphaGate:
    """Gate: scale position by funding-alpha alignment.

    At 100x leverage, funding rate becomes a first-order P&L driver:
      0.01% funding × 100x = 1% of capital per 8h = 3%/day

    Logic:
      signal=+1 AND funding>0 → longs PAY funding → scale DOWN (0.3x)
      signal=+1 AND funding<0 → longs RECEIVE funding → scale UP (1.5x)
      signal=-1 AND funding>0 → shorts RECEIVE funding → scale UP (1.5x)
      signal=-1 AND funding<0 → shorts PAY funding → scale DOWN (0.3x)

    High funding magnitude (>0.05%) → stronger scaling effect.
    """

    name = "FundingAlpha"

    # Thresholds
    HIGH_FUNDING_RATE = 0.0005    # 0.05% per 8h = extreme
    NORMAL_FUNDING_RATE = 0.0001  # 0.01% per 8h = typical

    # Scaling factors
    ALIGNED_SCALE = 1.5       # funding helps us → boost
    OPPOSED_SCALE = 0.3       # funding hurts us → reduce
    NEUTRAL_SCALE = 1.0       # funding negligible
    HIGH_ALIGNED_SCALE = 2.0  # extreme funding in our favor
    HIGH_OPPOSED_SCALE = 0.0  # extreme funding against us → skip

    def __init__(
        self,
        *,
        leverage: float = 100.0,
        get_funding_rate: Optional[Callable[[str], float]] = None,
        enabled: bool = True,
    ) -> None:
        self._leverage = leverage
        self._get_funding_rate = get_funding_rate
        self._enabled = enabled
        self._funding_history: deque[tuple[float, float]] = deque(maxlen=100)
        self._last_rate: float = 0.0

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        if not self._enabled:
            return GateResult(allowed=True, scale=1.0)

        # Get signal direction from event
        signal = 0
        meta = getattr(ev, "metadata", None) or {}
        if isinstance(meta, dict):
            signal = int(meta.get("signal", 0))
        if signal == 0:
            signal = int(context.get("signal", 0))

        if signal == 0:
            return GateResult(allowed=True, scale=1.0)

        # Get funding rate
        funding = self._get_funding(context)
        self._last_rate = funding

        if abs(funding) < 1e-8:
            return GateResult(allowed=True, scale=self.NEUTRAL_SCALE)

        # Track funding
        self._funding_history.append((time.time(), funding))

        # Determine alignment
        # signal=+1 (long): positive funding means longs PAY → bad for us
        # signal=-1 (short): positive funding means shorts RECEIVE → good for us
        receives_funding = (signal == -1 and funding > 0) or (signal == 1 and funding < 0)

        abs_rate = abs(funding)
        is_high = abs_rate >= self.HIGH_FUNDING_RATE

        if receives_funding:
            scale = self.HIGH_ALIGNED_SCALE if is_high else self.ALIGNED_SCALE
            reason = f"funding_aligned rate={funding:.6f} scale={scale:.1f}"
        else:
            scale = self.HIGH_OPPOSED_SCALE if is_high else self.OPPOSED_SCALE
            reason = f"funding_opposed rate={funding:.6f} scale={scale:.1f}"

        # At high oppose, block the trade entirely
        if scale <= 0.0:
            _log.info("FundingAlpha BLOCK: %s", reason)
            return GateResult(allowed=False, reason=reason)

        _log.debug("FundingAlpha: signal=%d funding=%.6f → scale=%.2f", signal, funding, scale)
        return GateResult(allowed=True, scale=scale, reason=reason)

    def _get_funding(self, context: Dict[str, Any]) -> float:
        """Get current funding rate from context or callback."""
        # Try context first
        rate = context.get("funding_rate", None)
        if rate is not None and rate == rate:  # not NaN
            return float(rate)

        # Try callback
        if self._get_funding_rate is not None:
            symbol = context.get("symbol", "")
            try:
                return float(self._get_funding_rate(symbol))
            except Exception:
                pass

        return self._last_rate  # fallback to last known

    @property
    def last_rate(self) -> float:
        return self._last_rate

    @property
    def funding_impact_per_8h(self) -> float:
        """Estimated funding impact as % of capital per 8h settlement."""
        return abs(self._last_rate) * self._leverage * 100
