# context/market/market_snapshot.py
"""Market snapshot utilities — aggregation and queries."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Optional, Sequence

from context.market.market_state import MarketSnapshot


@dataclass(frozen=True, slots=True)
class MarketSummary:
    """多品种市场摘要。"""
    symbol_count: int
    symbols: tuple[str, ...]
    latest_ts: int


def aggregate_market_snapshots(
    snapshots: Sequence[MarketSnapshot],
) -> MarketSummary:
    """汇总多个品种的市场快照。"""
    symbols = tuple(s.symbol for s in snapshots)
    latest_ts = max((s.ts for s in snapshots), default=0)
    return MarketSummary(
        symbol_count=len(snapshots),
        symbols=symbols,
        latest_ts=latest_ts,
    )


def filter_stale_snapshots(
    snapshots: Sequence[MarketSnapshot],
    *,
    max_age_ms: int,
    current_ts: int,
) -> Sequence[MarketSnapshot]:
    """过滤掉过期的快照。"""
    return [s for s in snapshots if (current_ts - s.ts) <= max_age_ms]
