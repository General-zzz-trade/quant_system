"""On-chain exchange inflow features.

Tracks BTC flowing into exchanges as a sell-pressure signal.
Large inflows historically precede selling activity.

Features:
- exchange_inflow_zscore: rolling z-score of hourly net inflow
- exchange_inflow_ma_ratio: current inflow / 24h MA
"""
from __future__ import annotations
from collections import deque
from typing import Dict, Optional
import math

ONCHAIN_FEATURES = ("exchange_inflow_zscore", "exchange_inflow_ma_ratio")


class OnchainFlowComputer:
    def __init__(self, zscore_window: int = 720) -> None:
        self._inflows: deque[float] = deque(maxlen=zscore_window)

    def update(self, net_inflow: Optional[float]) -> Dict[str, Optional[float]]:
        if net_inflow is None or math.isnan(net_inflow):
            return {name: None for name in ONCHAIN_FEATURES}

        self._inflows.append(net_inflow)

        zscore = None
        ma_ratio = None

        if len(self._inflows) >= 180:
            vals = list(self._inflows)
            mu = sum(vals) / len(vals)
            var = sum((x - mu) ** 2 for x in vals) / len(vals)
            std = var ** 0.5
            if std > 1e-12:
                zscore = (net_inflow - mu) / std

        if len(self._inflows) >= 24:
            recent_24 = list(self._inflows)[-24:]
            ma24 = sum(recent_24) / 24
            if abs(ma24) > 1e-12:
                ma_ratio = net_inflow / ma24

        return {"exchange_inflow_zscore": zscore, "exchange_inflow_ma_ratio": ma_ratio}
