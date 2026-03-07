"""Tick-level event types for HFT engine.

These events bypass the bar engine entirely and flow through the TickEngine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from execution.adapters.binance.depth_processor import OrderBookSnapshot


@dataclass(frozen=True, slots=True)
class TradeTickEvent:
    """A single aggTrade from the exchange."""

    symbol: str
    price: Decimal
    qty: Decimal
    side: str  # "buy" or "sell"
    trade_id: int
    ts_ms: int
    received_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class DepthUpdateEvent:
    """Wraps an OrderBookSnapshot for tick-level consumption."""

    snapshot: OrderBookSnapshot
    received_at: float = field(default_factory=time.monotonic)
