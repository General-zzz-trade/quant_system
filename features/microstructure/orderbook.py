"""Orderbook feature extraction from L2 snapshots.

Computes microstructure features including bid-ask spread, imbalance,
depth ratios, and volume-weighted mid price.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OrderbookSnapshot:
    """L2 orderbook snapshot."""

    ts: datetime
    symbol: str
    bids: tuple[tuple[Decimal, Decimal], ...]  # (price, qty) sorted desc
    asks: tuple[tuple[Decimal, Decimal], ...]  # (price, qty) sorted asc


@dataclass(frozen=True, slots=True)
class OrderbookFeatures:
    """Features extracted from a single orderbook snapshot."""

    bid_ask_spread: Decimal
    mid_price: Decimal
    bid_ask_imbalance: float  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    depth_ratio: float  # bid_depth / ask_depth at N levels
    weighted_mid: Decimal  # Volume-weighted mid price
    trade_flow_toxicity: float  # Proportion of aggressive trades


class OrderbookFeatureExtractor:
    """Extract features from L2 orderbook snapshots."""

    def __init__(self, *, depth_levels: int = 5) -> None:
        self._depth_levels = depth_levels

    def extract(self, snapshot: OrderbookSnapshot) -> OrderbookFeatures:
        """Extract all features from a single snapshot."""
        if not snapshot.bids or not snapshot.asks:
            zero = Decimal("0")
            return OrderbookFeatures(
                bid_ask_spread=zero,
                mid_price=zero,
                bid_ask_imbalance=0.0,
                depth_ratio=0.0,
                weighted_mid=zero,
                trade_flow_toxicity=0.0,
            )

        best_bid_price, best_bid_qty = snapshot.bids[0]
        best_ask_price, best_ask_qty = snapshot.asks[0]

        bid_ask_spread = best_ask_price - best_bid_price
        mid_price = (best_ask_price + best_bid_price) / Decimal("2")

        # Bid-ask imbalance from top N levels
        n = min(self._depth_levels, len(snapshot.bids), len(snapshot.asks))
        bid_vol = sum(snapshot.bids[i][1] for i in range(min(n, len(snapshot.bids))))
        ask_vol = sum(snapshot.asks[i][1] for i in range(min(n, len(snapshot.asks))))
        total_vol = bid_vol + ask_vol

        if total_vol > Decimal("0"):
            bid_ask_imbalance = float((bid_vol - ask_vol) / total_vol)
        else:
            bid_ask_imbalance = 0.0

        # Depth ratio from top N levels
        bid_depth = sum(
            snapshot.bids[i][0] * snapshot.bids[i][1]
            for i in range(min(n, len(snapshot.bids)))
        )
        ask_depth = sum(
            snapshot.asks[i][0] * snapshot.asks[i][1]
            for i in range(min(n, len(snapshot.asks)))
        )
        if ask_depth > Decimal("0"):
            depth_ratio = float(bid_depth / ask_depth)
        else:
            depth_ratio = 0.0

        # Volume-weighted mid price
        # weighted_mid = (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)
        top_total = best_bid_qty + best_ask_qty
        if top_total > Decimal("0"):
            weighted_mid = (
                (best_bid_price * best_ask_qty + best_ask_price * best_bid_qty)
                / top_total
            )
        else:
            weighted_mid = mid_price

        # Trade flow toxicity: approximated as spread / mid
        # (real implementation would use recent trade data)
        if mid_price > Decimal("0"):
            trade_flow_toxicity = float(bid_ask_spread / mid_price)
        else:
            trade_flow_toxicity = 0.0

        return OrderbookFeatures(
            bid_ask_spread=bid_ask_spread,
            mid_price=mid_price,
            bid_ask_imbalance=bid_ask_imbalance,
            depth_ratio=depth_ratio,
            weighted_mid=weighted_mid,
            trade_flow_toxicity=trade_flow_toxicity,
        )

    def extract_series(
        self,
        snapshots: Sequence[OrderbookSnapshot],
    ) -> dict[str, list[float]]:
        """Extract features from a sequence of snapshots.

        Returns a dict mapping feature names to lists of float values.
        """
        result: dict[str, list[float]] = {
            "bid_ask_spread": [],
            "mid_price": [],
            "bid_ask_imbalance": [],
            "depth_ratio": [],
            "weighted_mid": [],
            "trade_flow_toxicity": [],
        }

        for snap in snapshots:
            feats = self.extract(snap)
            result["bid_ask_spread"].append(float(feats.bid_ask_spread))
            result["mid_price"].append(float(feats.mid_price))
            result["bid_ask_imbalance"].append(feats.bid_ask_imbalance)
            result["depth_ratio"].append(feats.depth_ratio)
            result["weighted_mid"].append(float(feats.weighted_mid))
            result["trade_flow_toxicity"].append(feats.trade_flow_toxicity)

        return result
