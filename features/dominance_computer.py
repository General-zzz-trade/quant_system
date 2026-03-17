"""
V14 BTC/ETH dominance feature computer (Python reference implementation).

Computes 4 features from BTC and ETH close prices:
  - btc_dom_ratio_dev_20:   BTC/ETH ratio deviation from 20-bar MA
  - btc_dom_ratio_mom_10:   10-bar ratio momentum
  - btc_dom_return_diff_6h: BTC minus ETH 6-bar cumulative return difference
  - btc_dom_return_diff_24h: BTC minus ETH 24-bar cumulative return difference

This class is the parity reference for RustFeatureEngine.push_dominance().
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional


class DominanceComputer:
    """Incremental BTC/ETH dominance feature computer."""

    def __init__(self) -> None:
        # Circular buffer for BTC/ETH ratio (max 75 to cover max window 20 + 1 + margin)
        self._ratio_buf: deque[float] = deque(maxlen=75)
        # BTC and ETH return buffers (max 25 to cover 24-bar return diff + margin)
        self._btc_ret_buf: deque[float] = deque(maxlen=25)
        self._eth_ret_buf: deque[float] = deque(maxlen=25)
        self._last_btc: float = 0.0
        self._last_eth: float = 0.0

    def update(self, btc_close: float, eth_close: float) -> Dict[str, Optional[float]]:
        """Push a new bar and return the 4 dominance features.

        Returns a dict mapping feature name -> float or None (if insufficient data).
        """
        result: Dict[str, Optional[float]] = {}

        if eth_close <= 0.0 or btc_close <= 0.0:
            result["btc_dom_ratio_dev_20"] = None
            result["btc_dom_ratio_mom_10"] = None
            result["btc_dom_return_diff_6h"] = None
            result["btc_dom_return_diff_24h"] = None
            return result

        ratio = btc_close / eth_close
        self._ratio_buf.append(ratio)

        # Compute returns only when we have a previous close
        if self._last_btc > 0.0:
            self._btc_ret_buf.append(btc_close / self._last_btc - 1.0)
        if self._last_eth > 0.0:
            self._eth_ret_buf.append(eth_close / self._last_eth - 1.0)

        self._last_btc = btc_close
        self._last_eth = eth_close

        buf = list(self._ratio_buf)

        # btc_dom_ratio_dev_20: ratio / MA(20) - 1
        if len(buf) >= 20:
            ma20 = sum(buf[-20:]) / 20.0
            result["btc_dom_ratio_dev_20"] = ratio / ma20 - 1.0 if ma20 > 0.0 else None
        else:
            result["btc_dom_ratio_dev_20"] = None

        # btc_dom_ratio_mom_10: ratio / ratio[10 bars ago] - 1
        if len(buf) >= 11:
            prev = buf[-11]
            result["btc_dom_ratio_mom_10"] = ratio / prev - 1.0 if prev > 0.0 else None
        else:
            result["btc_dom_ratio_mom_10"] = None

        # btc_dom_return_diff_6h: sum(btc_ret[-6:]) - sum(eth_ret[-6:])
        btc_rets = list(self._btc_ret_buf)
        eth_rets = list(self._eth_ret_buf)
        if len(btc_rets) >= 6 and len(eth_rets) >= 6:
            btc_sum = sum(btc_rets[-6:])
            eth_sum = sum(eth_rets[-6:])
            result["btc_dom_return_diff_6h"] = btc_sum - eth_sum
        else:
            result["btc_dom_return_diff_6h"] = None

        # btc_dom_return_diff_24h: sum(btc_ret[-24:]) - sum(eth_ret[-24:])
        if len(btc_rets) >= 24 and len(eth_rets) >= 24:
            btc_sum = sum(btc_rets[-24:])
            eth_sum = sum(eth_rets[-24:])
            result["btc_dom_return_diff_24h"] = btc_sum - eth_sum
        else:
            result["btc_dom_return_diff_24h"] = None

        return result
