"""Live order book feature aggregator — aggregates 100ms depth snapshots into bar-level features.

These features are only available in live mode (no historical OB data).
Collects depth snapshots during a bar interval and computes aggregated statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from execution.adapters.binance.depth_processor import OrderBookSnapshot


@dataclass
class _BarAccumulator:
    """Accumulates order book metrics within one bar."""
    imbalances: List[float] = field(default_factory=list)
    spreads_bps: List[float] = field(default_factory=list)
    depth_ratios: List[float] = field(default_factory=list)
    n_snapshots: int = 0


ORDERBOOK_FEATURE_NAMES: tuple[str, ...] = (
    "ob_imbalance_mean",
    "ob_imbalance_slope",
    "ob_spread_mean_bps",
    "ob_spread_max_bps",
    "ob_depth_ratio_mean",
    "ob_pressure_score",
)


class LiveOrderbookFeatureAggregator:
    """Aggregates 100ms depth snapshots into bar-level OB features.

    Call on_depth() for each depth snapshot during the bar.
    Call flush_bar() at bar boundary to get aggregated features and reset.
    """

    def __init__(self, depth_levels: int = 10) -> None:
        self._depth_levels = depth_levels
        self._accumulators: Dict[str, _BarAccumulator] = {}

    def on_depth(self, snapshot: OrderBookSnapshot) -> None:
        """Process one depth snapshot."""
        symbol = snapshot.symbol
        if symbol not in self._accumulators:
            self._accumulators[symbol] = _BarAccumulator()
        acc = self._accumulators[symbol]
        acc.n_snapshots += 1

        # Bid/ask imbalance (top N levels)
        n = min(self._depth_levels, len(snapshot.bids), len(snapshot.asks))
        if n > 0:
            bid_vol = sum(float(snapshot.bids[i].qty) for i in range(n))
            ask_vol = sum(float(snapshot.asks[i].qty) for i in range(n))
            total = bid_vol + ask_vol
            if total > 0:
                imb = (bid_vol - ask_vol) / total
                acc.imbalances.append(imb)

            # Depth ratio (dollar depth)
            bid_depth = sum(float(snapshot.bids[i].price * snapshot.bids[i].qty) for i in range(n))
            ask_depth = sum(float(snapshot.asks[i].price * snapshot.asks[i].qty) for i in range(n))
            if ask_depth > 0:
                acc.depth_ratios.append(bid_depth / ask_depth)

        # Spread in bps
        spread_bps = snapshot.spread_bps
        if spread_bps is not None:
            acc.spreads_bps.append(float(spread_bps))

    def flush_bar(self, symbol: str) -> Dict[str, Optional[float]]:
        """Compute bar-level features and reset accumulator."""
        feats: Dict[str, Optional[float]] = {n: None for n in ORDERBOOK_FEATURE_NAMES}

        acc = self._accumulators.pop(symbol, None)
        if acc is None or acc.n_snapshots < 2:
            return feats

        # Imbalance mean
        if acc.imbalances:
            imb_mean = sum(acc.imbalances) / len(acc.imbalances)
            feats["ob_imbalance_mean"] = imb_mean

            # Imbalance slope (linear regression y = a + bx)
            if len(acc.imbalances) >= 3:
                n = len(acc.imbalances)
                x_mean = (n - 1) / 2.0
                y_mean = imb_mean
                num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(acc.imbalances))
                denom = sum((i - x_mean) ** 2 for i in range(n))
                feats["ob_imbalance_slope"] = num / denom if denom > 0 else 0.0

        # Spread
        if acc.spreads_bps:
            feats["ob_spread_mean_bps"] = sum(acc.spreads_bps) / len(acc.spreads_bps)
            feats["ob_spread_max_bps"] = max(acc.spreads_bps)

        # Depth ratio
        if acc.depth_ratios:
            feats["ob_depth_ratio_mean"] = sum(acc.depth_ratios) / len(acc.depth_ratios)

        # Pressure score: imbalance * (1 - spread_norm)
        imb = feats.get("ob_imbalance_mean")
        spread_mean = feats.get("ob_spread_mean_bps")
        if imb is not None and spread_mean is not None:
            # Normalize spread: lower spread = higher confidence
            spread_norm = min(spread_mean / 20.0, 1.0)  # 20 bps = max normalization
            feats["ob_pressure_score"] = imb * (1.0 - spread_norm)

        return feats

    def reset(self, symbol: Optional[str] = None) -> None:
        if symbol is not None:
            self._accumulators.pop(symbol, None)
        else:
            self._accumulators.clear()
