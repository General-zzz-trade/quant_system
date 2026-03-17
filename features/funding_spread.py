"""Multi-exchange funding rate spread features.

Computes cross-exchange funding rate divergence as a predictive signal.
When funding rates diverge across exchanges, it signals imbalanced
positioning that often precedes price moves.

Features:
- funding_spread: max(rates) - min(rates) across exchanges
- funding_skew: mean - median (asymmetry indicator)
- funding_zscore_spread: rolling z-score of spread (720-bar window)
"""
from __future__ import annotations
from collections import deque
from typing import Dict, Optional
import logging
import math

_log = logging.getLogger(__name__)

FUNDING_SPREAD_FEATURES = (
    "funding_spread",
    "funding_skew",
    "funding_zscore_spread",
)


class FundingSpreadComputer:
    """Compute funding rate spread features across exchanges."""

    def __init__(self, zscore_window: int = 720) -> None:
        self._spread_history: deque = deque(maxlen=zscore_window)

    def update(self, rates: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Push funding rates from multiple exchanges.

        Args:
            rates: {"binance": 0.0001, "bybit": 0.00012, "okx": 0.00008}

        Returns:
            Dict with funding_spread, funding_skew, funding_zscore_spread
        """
        valid = [v for v in rates.values() if v is not None and not math.isnan(v)]

        if len(valid) < 2:
            return {name: None for name in FUNDING_SPREAD_FEATURES}

        spread = max(valid) - min(valid)
        mean_val = sum(valid) / len(valid)
        sorted_valid = sorted(valid)
        n = len(sorted_valid)
        median_val = sorted_valid[n // 2] if n % 2 else (sorted_valid[n // 2 - 1] + sorted_valid[n // 2]) / 2
        skew = mean_val - median_val

        self._spread_history.append(spread)

        # Z-score of spread
        zscore = None
        if len(self._spread_history) >= 180:  # warmup
            vals = list(self._spread_history)
            mu = sum(vals) / len(vals)
            var = sum((x - mu) ** 2 for x in vals) / len(vals)
            std = var ** 0.5
            if std > 1e-12:
                zscore = (spread - mu) / std

        return {
            "funding_spread": spread,
            "funding_skew": skew,
            "funding_zscore_spread": zscore,
        }
