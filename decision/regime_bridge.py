# decision/regime_bridge.py
"""RegimeAwareDecisionModule — wraps a DecisionModule with regime gating.

NOTE: This module IS used in the production path. LiveRunner, BacktestRunner, and
LivePaperRunner all wrap inner decision modules with RegimeAwareDecisionModule to
apply regime-based gating before order generation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from _quant_hotpath import RustRegimeBuffer
from regime.base import RegimeDetector, RegimeLabel
from regime.volatility import VolatilityRegimeDetector
from regime.trend import TrendRegimeDetector
from decision.market_access import get_float_attr
from decision.regime_policy import RegimePolicy

logger = logging.getLogger(__name__)


def _make_buffer(maxlen: int = 100) -> RustRegimeBuffer:
    return RustRegimeBuffer(maxlen)


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

    _buffers: Dict[str, RustRegimeBuffer] = field(default_factory=dict, init=False)
    _last_labels: List[RegimeLabel] = field(default_factory=list, init=False)

    def decide(self, snapshot: Any) -> Iterable[Any]:
        ts = getattr(snapshot, "ts", None)
        markets = getattr(snapshot, "markets", {})

        # Update price buffers from snapshot
        for sym, mkt in markets.items():
            price = get_float_attr(mkt, "close", "last_price")
            if price is None:
                continue
            if sym not in self._buffers:
                self._buffers[sym] = _make_buffer(self.buffer_maxlen)
            self._buffers[sym].push(price)

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
