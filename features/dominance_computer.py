# features/dominance_computer.py
"""Standalone BTC dominance feature computer (V14).

Extracted from EnrichedFeatureComputer._dom_ratio_buf logic.

Computes BTC/ETH price ratio features that capture capital rotation
between BTC and ETH. The ratio deviation from its moving average is
the #1 feature for BTCUSDT model (h=96, Sharpe 2.03).

Features produced:
  btc_dom_ratio_dev_20  — ratio deviation from MA(20): cur/ma20 - 1
  btc_dom_ratio_dev_50  — ratio deviation from MA(50): cur/ma50 - 1
  btc_dom_ratio_ret_24  — ratio 24-bar return: cur/buf[-25] - 1
  btc_dom_ratio_ret_72  — ratio 72-bar return: cur/buf[-73] - 1
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional


class DominanceComputer:
    """Incremental BTC/ETH dominance ratio feature computer.

    Call :meth:`update` once per bar with the BTC and ETH close prices.
    Returns a dict of feature values (None if insufficient history).

    Warmup requirements:
    - btc_dom_ratio_dev_20: 21 bars
    - btc_dom_ratio_dev_50: 51 bars
    - btc_dom_ratio_ret_24: 25 bars
    - btc_dom_ratio_ret_72: 73 bars
    """

    _MAXLEN = 75  # matches enriched_computer._dom_ratio_buf maxlen

    def __init__(self) -> None:
        self._buf: deque = deque(maxlen=self._MAXLEN)

    def update(
        self,
        btc_close: float,
        eth_close: float,
    ) -> Dict[str, Optional[float]]:
        """Push one bar and return current dominance features.

        Args:
            btc_close: BTC closing price.
            eth_close: ETH closing price.

        Returns:
            Dict with keys btc_dom_ratio_dev_20, btc_dom_ratio_dev_50,
            btc_dom_ratio_ret_24, btc_dom_ratio_ret_72.
            Values are None when insufficient history.
        """
        if eth_close > 0:
            ratio = btc_close / eth_close
        else:
            ratio = None

        if ratio is not None:
            self._buf.append(ratio)

        buf = self._buf
        n = len(buf)
        feats: Dict[str, Optional[float]] = {}

        # btc_dom_ratio_dev_20: deviation from 20-bar MA
        if n >= 21:
            cur = buf[-1]
            ma20 = sum(list(buf)[-20:]) / 20
            feats["btc_dom_ratio_dev_20"] = cur / ma20 - 1 if ma20 > 0 else None
        else:
            feats["btc_dom_ratio_dev_20"] = None

        # btc_dom_ratio_dev_50: deviation from 50-bar MA
        if n >= 51:
            cur = buf[-1]
            ma50 = sum(list(buf)[-50:]) / 50
            feats["btc_dom_ratio_dev_50"] = cur / ma50 - 1 if ma50 > 0 else None
        else:
            feats["btc_dom_ratio_dev_50"] = None

        # btc_dom_ratio_ret_24: 24-bar return
        if n >= 25:
            feats["btc_dom_ratio_ret_24"] = (
                buf[-1] / buf[-25] - 1 if buf[-25] > 0 else None
            )
        else:
            feats["btc_dom_ratio_ret_24"] = None

        # btc_dom_ratio_ret_72: 72-bar return
        if n >= 73:
            feats["btc_dom_ratio_ret_72"] = (
                buf[-1] / buf[-73] - 1 if buf[-73] > 0 else None
            )
        else:
            feats["btc_dom_ratio_ret_72"] = None

        return feats
