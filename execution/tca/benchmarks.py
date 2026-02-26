"""TCA benchmark calculations — arrival price, VWAP, TWAP, implementation shortfall."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TCABenchmark:
    """A single execution benchmark value."""

    name: str
    value: Decimal


class BenchmarkCalculator:
    """Calculate execution price benchmarks for TCA."""

    def arrival_price(
        self,
        order_ts: datetime,
        ticks: Sequence[Any],
    ) -> TCABenchmark:
        """Price at order arrival time.

        Finds the tick closest to (but not before) order_ts.
        Falls back to the first tick if all are after order_ts.
        """
        if not ticks:
            return TCABenchmark(name="arrival", value=Decimal("0"))

        best: Any = ticks[0]
        best_delta: float = float("inf")

        for tick in ticks:
            ts = getattr(tick, "ts", None)
            if ts is None:
                continue
            delta = abs((ts - order_ts).total_seconds())
            if delta < best_delta:
                best_delta = delta
                best = tick

        return TCABenchmark(
            name="arrival",
            value=getattr(best, "price", Decimal("0")),
        )

    def vwap_benchmark(
        self,
        ticks: Sequence[Any],
        start: datetime,
        end: datetime,
    ) -> TCABenchmark:
        """Volume-weighted average price over period."""
        total_notional = Decimal("0")
        total_qty = Decimal("0")

        for tick in ticks:
            ts = getattr(tick, "ts", None)
            if ts is None:
                continue
            if ts < start or ts > end:
                continue
            price = getattr(tick, "price", Decimal("0"))
            qty = getattr(tick, "qty", Decimal("0"))
            total_notional += price * qty
            total_qty += qty

        if total_qty > Decimal("0"):
            vwap = total_notional / total_qty
        else:
            vwap = Decimal("0")

        return TCABenchmark(name="vwap", value=vwap)

    def twap_benchmark(
        self,
        ticks: Sequence[Any],
        start: datetime,
        end: datetime,
    ) -> TCABenchmark:
        """Time-weighted average price over period."""
        prices: list[Decimal] = []

        for tick in ticks:
            ts = getattr(tick, "ts", None)
            if ts is None:
                continue
            if ts < start or ts > end:
                continue
            prices.append(getattr(tick, "price", Decimal("0")))

        if prices:
            twap = sum(prices) / len(prices)
        else:
            twap = Decimal("0")

        return TCABenchmark(name="twap", value=twap)

    def implementation_shortfall(
        self,
        decision_price: Decimal,
        avg_fill_price: Decimal,
        side: str,
    ) -> TCABenchmark:
        """Implementation shortfall = (fill_price - decision_price) * direction.

        Positive value = unfavorable slippage.
        For buys: fill > decision = bad.
        For sells: fill < decision = bad.
        """
        direction = Decimal("1") if side == "buy" else Decimal("-1")
        shortfall = (avg_fill_price - decision_price) * direction

        return TCABenchmark(name="implementation_shortfall", value=shortfall)
