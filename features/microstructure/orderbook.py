"""Orderbook feature extraction from L2 snapshots.

Delegates computation to rust_extract_orderbook_features.
Keeps Python dataclasses for API compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Sequence

from _quant_hotpath import rust_extract_orderbook_features


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
    bid_ask_imbalance: float
    depth_ratio: float
    weighted_mid: Decimal
    trade_flow_toxicity: float


class OrderbookFeatureExtractor:
    """Extract features from L2 orderbook snapshots — delegates to Rust."""

    def __init__(self, *, depth_levels: int = 5) -> None:
        self._depth_levels = depth_levels

    def extract(self, snapshot: OrderbookSnapshot) -> OrderbookFeatures:
        """Extract all features from a single snapshot."""
        if not snapshot.bids or not snapshot.asks:
            zero = Decimal("0")
            return OrderbookFeatures(
                bid_ask_spread=zero, mid_price=zero,
                bid_ask_imbalance=0.0, depth_ratio=0.0,
                weighted_mid=zero, trade_flow_toxicity=0.0,
            )

        bids_list = [[float(p), float(q)] for p, q in snapshot.bids]
        asks_list = [[float(p), float(q)] for p, q in snapshot.asks]
        r = rust_extract_orderbook_features(bids_list, asks_list, self._depth_levels)

        return OrderbookFeatures(
            bid_ask_spread=Decimal(str(r["bid_ask_spread"])),
            mid_price=Decimal(str(r["mid_price"])),
            bid_ask_imbalance=r["bid_ask_imbalance"],
            depth_ratio=r["depth_ratio"],
            weighted_mid=Decimal(str(r["weighted_mid"])),
            trade_flow_toxicity=r["trade_flow_toxicity"],
        )

    def extract_series(
        self,
        snapshots: Sequence[OrderbookSnapshot],
    ) -> dict[str, list[float]]:
        """Extract features from a sequence of snapshots."""
        result: dict[str, list[float]] = {
            "bid_ask_spread": [], "mid_price": [],
            "bid_ask_imbalance": [], "depth_ratio": [],
            "weighted_mid": [], "trade_flow_toxicity": [],
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
