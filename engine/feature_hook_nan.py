"""NaN tracking mixin methods for FeatureComputeHook."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict

_log = logging.getLogger(__name__)


class NanTrackingMixin:
    """Mixin providing NaN rate tracking for feature hooks.

    Expects the following instance attributes to be initialized by the host class:
        _nan_counts: Dict[str, Dict[str, int]]
        _nan_bar_counts: Dict[str, int]
        _nan_recent: Dict[str, Dict[str, int]]
        _nan_recent_bars: Dict[str, int]
        _NAN_WARN_WINDOW: int
        _NAN_WARN_RATE: float
    """

    _nan_counts: Dict[str, Dict[str, int]]
    _nan_bar_counts: Dict[str, int]
    _nan_recent: Dict[str, Dict[str, int]]
    _nan_recent_bars: Dict[str, int]
    _NAN_WARN_WINDOW: int
    _NAN_WARN_RATE: float

    def _track_nan(self, symbol: str, features: Dict[str, Any]) -> None:
        """Scan features for NaN/None and update counters. Lightweight dict ops only."""
        if symbol not in self._nan_counts:
            self._nan_counts[symbol] = {}
            self._nan_recent[symbol] = {}
        sym_counts = self._nan_counts[symbol]
        sym_recent = self._nan_recent[symbol]

        self._nan_bar_counts[symbol] = self._nan_bar_counts.get(symbol, 0) + 1
        recent_bars = self._nan_recent_bars.get(symbol, 0) + 1
        self._nan_recent_bars[symbol] = recent_bars

        for key, val in features.items():
            is_nan = val is None or (isinstance(val, float) and math.isnan(val))
            if is_nan:
                sym_counts[key] = sym_counts.get(key, 0) + 1
                sym_recent[key] = sym_recent.get(key, 0) + 1

        # Check recent window and warn
        if recent_bars >= self._NAN_WARN_WINDOW:
            for feat, cnt in sym_recent.items():
                rate = cnt / recent_bars
                if rate > self._NAN_WARN_RATE:
                    _log.warning(
                        "NaN rate %.1f%% for %s/%s in last %d bars",
                        rate * 100, symbol, feat, recent_bars,
                    )
            # Reset recent window
            self._nan_recent[symbol] = {}
            self._nan_recent_bars[symbol] = 0

    def nan_report(self) -> Dict[str, Dict[str, float]]:
        """Return {symbol: {feature: nan_rate}} for features with nan_rate > 1%."""
        result: Dict[str, Dict[str, float]] = {}
        for symbol, feat_counts in self._nan_counts.items():
            total = self._nan_bar_counts.get(symbol, 0)
            if total == 0:
                continue
            bad: Dict[str, float] = {}
            for feat, cnt in feat_counts.items():
                rate = cnt / total
                if rate > 0.01:
                    bad[feat] = round(rate, 4)
            if bad:
                result[symbol] = bad
        return result

    def reset_nan_stats(self) -> None:
        """Clear all NaN tracking counters."""
        self._nan_counts.clear()
        self._nan_bar_counts.clear()
        self._nan_recent.clear()
        self._nan_recent_bars.clear()
