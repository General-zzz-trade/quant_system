"""V14 BTC/ETH dominance feature computer.

Tracks BTC/ETH price ratio and computes deviation/momentum features.
Temporary Python implementation — Phase 3 migrates to Rust.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional

_log = logging.getLogger(__name__)

_DOMINANCE_FEATURES = (
    "btc_dom_ratio_dev_20",
    "btc_dom_ratio_mom_10",
    "btc_dom_return_diff_6h",
    "btc_dom_return_diff_24h",
)


class DominanceComputer:
    """Computes BTC/ETH price ratio features incrementally."""

    def __init__(self, window: int = 75) -> None:
        self._ratios: deque[float] = deque(maxlen=window)
        self._btc_returns: deque[float] = deque(maxlen=25)
        self._eth_returns: deque[float] = deque(maxlen=25)
        self._last_btc: Optional[float] = None
        self._last_eth: Optional[float] = None

    def update(self, btc_close: float, eth_close: float) -> Dict[str, Optional[float]]:
        """Push new bar prices and return 4 dominance features."""
        if eth_close <= 0 or btc_close <= 0:
            return {name: None for name in _DOMINANCE_FEATURES}

        ratio = btc_close / eth_close
        self._ratios.append(ratio)

        # Track returns for return diff features
        if self._last_btc is not None and self._last_btc > 0:
            self._btc_returns.append(btc_close / self._last_btc - 1)
        if self._last_eth is not None and self._last_eth > 0:
            self._eth_returns.append(eth_close / self._last_eth - 1)
        self._last_btc = btc_close
        self._last_eth = eth_close

        result: Dict[str, Optional[float]] = {}

        # ratio_dev_20: deviation from 20-bar MA
        if len(self._ratios) >= 20:
            recent = list(self._ratios)[-20:]
            ma20 = sum(recent) / 20
            result["btc_dom_ratio_dev_20"] = (ratio / ma20 - 1) if ma20 > 0 else None
        else:
            result["btc_dom_ratio_dev_20"] = None

        # ratio_mom_10: 10-bar momentum
        if len(self._ratios) >= 11:
            result["btc_dom_ratio_mom_10"] = ratio / self._ratios[-11] - 1
        else:
            result["btc_dom_ratio_mom_10"] = None

        # return_diff_6h and 24h
        if len(self._btc_returns) >= 6 and len(self._eth_returns) >= 6:
            btc_ret_6 = sum(list(self._btc_returns)[-6:])
            eth_ret_6 = sum(list(self._eth_returns)[-6:])
            result["btc_dom_return_diff_6h"] = btc_ret_6 - eth_ret_6
        else:
            result["btc_dom_return_diff_6h"] = None

        if len(self._btc_returns) >= 24 and len(self._eth_returns) >= 24:
            btc_ret_24 = sum(list(self._btc_returns)[-24:])
            eth_ret_24 = sum(list(self._eth_returns)[-24:])
            result["btc_dom_return_diff_24h"] = btc_ret_24 - eth_ret_24
        else:
            result["btc_dom_return_diff_24h"] = None

        return result
