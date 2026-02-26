# decision/regime_bridge.py
"""RegimeAwareDecisionModule — wraps a DecisionModule with regime gating."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

from regime.base import RegimeDetector, RegimeLabel
from regime.volatility import VolatilityRegimeDetector
from regime.trend import TrendRegimeDetector
from decision.regime_policy import RegimePolicy

logger = logging.getLogger(__name__)


@dataclass
class _PriceBuffer:
    """Per-symbol price buffer for feature computation."""
    maxlen: int = 100
    _prices: Deque[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._prices = deque(maxlen=self.maxlen)

    def push(self, price: float) -> None:
        self._prices.append(price)

    @property
    def n(self) -> int:
        return len(self._prices)

    def rolling_vol(self, window: int = 20) -> Optional[float]:
        if self.n < window + 1:
            return None
        prices = list(self._prices)
        rets = [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(len(prices) - window, len(prices))
            if prices[i - 1] != 0
        ]
        if len(rets) < 2:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        return sqrt(max(var, 0.0))

    def ma(self, window: int) -> Optional[float]:
        if self.n < window:
            return None
        prices = list(self._prices)
        return sum(prices[-window:]) / window


@dataclass
class RegimeAwareDecisionModule:
    """DecisionModule that gates an inner module with regime detection.

    Flow:
    1. Extract prices from snapshot, update per-symbol buffers
    2. Compute rolling features (vol, MA fast/slow)
    3. Run regime detectors to get labels
    4. RegimePolicy decides allow/block
    5. If allowed, delegate to inner DecisionModule.decide()
    """

    inner: Any  # DecisionModule (has .decide(snapshot))
    detectors: Sequence[RegimeDetector] = field(default_factory=lambda: [
        VolatilityRegimeDetector(),
        TrendRegimeDetector(),
    ])
    policy: RegimePolicy = field(default_factory=RegimePolicy)

    ma_fast_window: int = 10
    ma_slow_window: int = 30
    vol_window: int = 20
    buffer_maxlen: int = 100

    _buffers: Dict[str, _PriceBuffer] = field(default_factory=dict, init=False)
    _last_labels: List[RegimeLabel] = field(default_factory=list, init=False)

    def decide(self, snapshot: Any) -> Iterable[Any]:
        ts = getattr(snapshot, "ts", None)
        markets = getattr(snapshot, "markets", {})

        # Update price buffers from snapshot
        for sym, mkt in markets.items():
            price = getattr(mkt, "close", None) or getattr(mkt, "last_price", None)
            if price is None:
                continue
            if sym not in self._buffers:
                self._buffers[sym] = _PriceBuffer(maxlen=self.buffer_maxlen)
            self._buffers[sym].push(float(price))

        # Compute features and detect regimes
        all_labels: List[RegimeLabel] = []
        for sym, buf in self._buffers.items():
            features: Dict[str, Any] = {}
            vol = buf.rolling_vol(self.vol_window)
            if vol is not None:
                features["vol"] = vol
            ma_fast = buf.ma(self.ma_fast_window)
            ma_slow = buf.ma(self.ma_slow_window)
            if ma_fast is not None:
                features["ma_fast"] = ma_fast
            if ma_slow is not None:
                features["ma_slow"] = ma_slow

            if not features or ts is None:
                continue

            for det in self.detectors:
                label = det.detect(symbol=sym, ts=ts, features=features)
                if label is not None:
                    all_labels.append(label)

        self._last_labels = all_labels

        # Policy gate
        allowed, reason = self.policy.allow(all_labels)
        if not allowed:
            logger.info(
                "Regime gate blocked: %s (labels: %s)",
                reason, [(l.name, l.value) for l in all_labels],
            )
            return ()

        # Delegate to inner module
        return self.inner.decide(snapshot)

    @property
    def current_labels(self) -> List[RegimeLabel]:
        return list(self._last_labels)
