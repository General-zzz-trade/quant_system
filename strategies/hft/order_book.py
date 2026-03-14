"""Order book analysis — imbalance signals and microstructure features.

Processes order book snapshots to generate trading signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from execution.adapters.binance.depth_processor import OrderBookSnapshot


@dataclass(frozen=True, slots=True)
class OrderBookSignal:
    """Signal derived from order book analysis."""
    imbalance: float       # -1 (sell pressure) to +1 (buy pressure)
    depth_ratio: float     # bid depth / ask depth
    spread_bps: float
    weighted_mid: Decimal
    signal: str            # "buy_pressure", "sell_pressure", "neutral"


def compute_imbalance(snapshot: OrderBookSnapshot, levels: int = 5) -> float:
    """Compute order book imbalance from top N levels.

    Imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    Range: [-1, 1]. Positive = buy pressure.
    """
    bid_qty = sum(
        level.qty for level in snapshot.bids[:levels]
    )
    ask_qty = sum(
        level.qty for level in snapshot.asks[:levels]
    )
    total = bid_qty + ask_qty
    if total == 0:
        return 0.0
    return float((bid_qty - ask_qty) / total)


def compute_weighted_mid(snapshot: OrderBookSnapshot) -> Optional[Decimal]:
    """Volume-weighted mid price from best bid/ask."""
    if not snapshot.bids or not snapshot.asks:
        return None

    bb = snapshot.bids[0]
    ba = snapshot.asks[0]
    total_qty = bb.qty + ba.qty
    if total_qty == 0:
        return snapshot.mid_price

    # Weight by opposite side (more ask qty → mid closer to bid)
    return (bb.price * ba.qty + ba.price * bb.qty) / total_qty


def analyze_book(
    snapshot: OrderBookSnapshot,
    *,
    levels: int = 5,
    imbalance_threshold: float = 0.3,
) -> OrderBookSignal:
    """Full order book analysis producing a trading signal."""
    imbalance = compute_imbalance(snapshot, levels=levels)

    bid_depth = sum(level.qty for level in snapshot.bids[:levels])
    ask_depth = sum(level.qty for level in snapshot.asks[:levels])
    depth_ratio = float(bid_depth / ask_depth) if ask_depth > 0 else float("inf")

    spread_bps = float(snapshot.spread_bps or Decimal("0"))
    weighted_mid = compute_weighted_mid(snapshot) or snapshot.mid_price or Decimal("0")

    if imbalance > imbalance_threshold:
        signal = "buy_pressure"
    elif imbalance < -imbalance_threshold:
        signal = "sell_pressure"
    else:
        signal = "neutral"

    return OrderBookSignal(
        imbalance=imbalance,
        depth_ratio=depth_ratio,
        spread_bps=spread_bps,
        weighted_mid=weighted_mid,
        signal=signal,
    )
