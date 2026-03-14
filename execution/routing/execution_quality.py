"""Execution quality analysis — measures slippage, fill rates, and latency.

Tracks order execution quality metrics for performance monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Single order execution record for quality analysis."""
    order_id: str
    symbol: str
    side: str
    intended_qty: Decimal
    filled_qty: Decimal
    intended_price: Decimal  # mid price at decision time
    avg_fill_price: Decimal
    venue_id: str
    submit_ts: float
    fill_ts: float


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Aggregated execution quality metrics."""
    total_orders: int
    fill_rate: float  # filled_qty / intended_qty
    avg_slippage_bps: float  # avg slippage in basis points
    median_slippage_bps: float
    worst_slippage_bps: float
    avg_latency_ms: float
    total_slippage_cost: Decimal  # total $ cost of slippage


def _slippage_bps(
    side: str, intended_price: Decimal, fill_price: Decimal,
) -> float:
    """Calculate slippage in basis points (positive = unfavorable)."""
    if intended_price <= 0:
        return 0.0
    diff = fill_price - intended_price
    if side.lower() == "sell":
        diff = -diff  # For sells, lower fill = worse
    return float(diff / intended_price * 10000)


class ExecutionQualityTracker:
    """Tracks and analyzes execution quality over time."""

    def __init__(self, *, max_history: int = 10000) -> None:
        self._records: List[ExecutionRecord] = []
        self._max_history = max_history
        self._by_venue: Dict[str, List[ExecutionRecord]] = {}
        self._by_symbol: Dict[str, List[ExecutionRecord]] = {}

    def record(self, rec: ExecutionRecord) -> None:
        """Add an execution record."""
        self._records.append(rec)
        self._by_venue.setdefault(rec.venue_id, []).append(rec)
        self._by_symbol.setdefault(rec.symbol, []).append(rec)

        # Trim oldest if over limit
        if len(self._records) > self._max_history:
            removed = self._records.pop(0)
            venue_list = self._by_venue.get(removed.venue_id)
            if venue_list and venue_list[0] is removed:
                venue_list.pop(0)
            sym_list = self._by_symbol.get(removed.symbol)
            if sym_list and sym_list[0] is removed:
                sym_list.pop(0)

    @property
    def record_count(self) -> int:
        return len(self._records)

    def compute_metrics(
        self,
        *,
        venue_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Optional[QualityMetrics]:
        """Compute quality metrics, optionally filtered by venue or symbol."""
        if venue_id:
            records = self._by_venue.get(venue_id, [])
        elif symbol:
            records = self._by_symbol.get(symbol, [])
        else:
            records = self._records

        if not records:
            return None

        slippages: List[float] = []
        latencies: List[float] = []
        total_intended = Decimal("0")
        total_filled = Decimal("0")
        total_slippage_cost = Decimal("0")

        for rec in records:
            slip = _slippage_bps(rec.side, rec.intended_price, rec.avg_fill_price)
            slippages.append(slip)
            latencies.append((rec.fill_ts - rec.submit_ts) * 1000)
            total_intended += rec.intended_qty
            total_filled += rec.filled_qty

            price_diff = rec.avg_fill_price - rec.intended_price
            if rec.side.lower() == "sell":
                price_diff = -price_diff
            total_slippage_cost += price_diff * rec.filled_qty

        sorted_slippages = sorted(slippages)
        n = len(sorted_slippages)
        median_slip = sorted_slippages[n // 2] if n else 0.0

        fill_rate = float(total_filled / total_intended) if total_intended > 0 else 0.0

        return QualityMetrics(
            total_orders=len(records),
            fill_rate=fill_rate,
            avg_slippage_bps=sum(slippages) / n if n else 0.0,
            median_slippage_bps=median_slip,
            worst_slippage_bps=max(slippages) if slippages else 0.0,
            avg_latency_ms=sum(latencies) / n if n else 0.0,
            total_slippage_cost=total_slippage_cost,
        )

    def should_reduce_size(self, symbol: str, threshold_bps: float = 30.0) -> float:
        """Return position scale factor (0.0-1.0) based on recent slippage.

        If recent average slippage exceeds 2x threshold -> 0.0 (halt).
        If recent average slippage exceeds threshold -> 0.5 (reduce).
        Otherwise -> 1.0 (no reduction).
        Requires at least 5 records for the symbol.
        """
        records = self._by_symbol.get(symbol, [])
        if len(records) < 5:
            return 1.0
        recent = records[-20:]
        avg_slip = sum(
            _slippage_bps(r.side, r.intended_price, r.avg_fill_price)
            for r in recent
        ) / len(recent)
        if avg_slip > threshold_bps * 2:
            return 0.0
        if avg_slip > threshold_bps:
            return 0.5
        return 1.0

    def venue_comparison(self) -> Dict[str, QualityMetrics]:
        """Compare execution quality across venues."""
        result: Dict[str, QualityMetrics] = {}
        for venue_id in self._by_venue:
            metrics = self.compute_metrics(venue_id=venue_id)
            if metrics:
                result[venue_id] = metrics
        return result
