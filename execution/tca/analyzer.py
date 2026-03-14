"""TCA analyzer — post-trade transaction cost analysis."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Sequence

from execution.tca.benchmarks import BenchmarkCalculator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FillRecord:
    """Record of a single order fill."""

    ts: datetime
    symbol: str
    side: str
    qty: Decimal
    price: Decimal
    fee: Decimal = Decimal("0")
    algo: str = ""
    order_id: str = ""


@dataclass(frozen=True, slots=True)
class TCAResult:
    """Result of transaction cost analysis for one order."""

    order_id: str
    symbol: str
    side: str
    total_qty: Decimal
    avg_fill_price: Decimal
    benchmarks: dict[str, Decimal]  # benchmark_name -> value
    slippage_bps: dict[str, float]  # benchmark_name -> slippage in bps
    total_fee: Decimal
    algo: str
    fill_count: int
    duration_sec: float


class TCAAnalyzer:
    """Post-trade Transaction Cost Analysis.

    Computes execution quality metrics by comparing fill prices
    against standard benchmarks (arrival, VWAP, TWAP, IS).
    """

    def __init__(
        self,
        benchmark_calc: Optional[BenchmarkCalculator] = None,
    ) -> None:
        self._benchmark = benchmark_calc or BenchmarkCalculator()

    def analyze(
        self,
        fills: Sequence[FillRecord],
        ticks: Optional[Sequence[Any]] = None,
        decision_price: Optional[Decimal] = None,
    ) -> TCAResult:
        """Analyze execution quality for a set of fills.

        Steps:
        1. Compute VWAP of fills (average fill price)
        2. Compute benchmarks (arrival, VWAP, TWAP, IS)
        3. Compute slippage vs each benchmark in bps
        """
        if not fills:
            return TCAResult(
                order_id="",
                symbol="",
                side="",
                total_qty=Decimal("0"),
                avg_fill_price=Decimal("0"),
                benchmarks={},
                slippage_bps={},
                total_fee=Decimal("0"),
                algo="",
                fill_count=0,
                duration_sec=0.0,
            )

        first = fills[0]
        order_id = first.order_id
        symbol = first.symbol
        side = first.side
        algo = first.algo

        # Compute fill VWAP
        total_notional = sum(f.qty * f.price for f in fills)
        total_qty = sum(f.qty for f in fills)
        avg_fill_price = total_notional / total_qty if total_qty else Decimal("0")
        total_fee = sum(f.fee for f in fills)

        # Duration
        timestamps = [f.ts for f in fills]
        earliest = min(timestamps)
        latest = max(timestamps)
        duration_sec = (latest - earliest).total_seconds()

        # Benchmarks
        benchmarks: dict[str, Decimal] = {}
        slippage_bps: dict[str, float] = {}

        if ticks:
            arrival = self._benchmark.arrival_price(earliest, ticks)
            benchmarks[arrival.name] = arrival.value
            slippage_bps[arrival.name] = self._slippage_bps(
                avg_fill_price, arrival.value, side,
            )

            vwap = self._benchmark.vwap_benchmark(ticks, earliest, latest)
            benchmarks[vwap.name] = vwap.value
            slippage_bps[vwap.name] = self._slippage_bps(
                avg_fill_price, vwap.value, side,
            )

            twap = self._benchmark.twap_benchmark(ticks, earliest, latest)
            benchmarks[twap.name] = twap.value
            slippage_bps[twap.name] = self._slippage_bps(
                avg_fill_price, twap.value, side,
            )

        if decision_price is not None:
            is_bench = self._benchmark.implementation_shortfall(
                decision_price, avg_fill_price, side,
            )
            benchmarks[is_bench.name] = is_bench.value
            # IS slippage is the raw shortfall in bps
            if decision_price > Decimal("0"):
                slippage_bps["implementation_shortfall"] = float(
                    is_bench.value / decision_price * Decimal("10000"),
                )

        return TCAResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_qty=total_qty,
            avg_fill_price=avg_fill_price,
            benchmarks=benchmarks,
            slippage_bps=slippage_bps,
            total_fee=total_fee,
            algo=algo,
            fill_count=len(fills),
            duration_sec=duration_sec,
        )

    def _slippage_bps(
        self,
        fill_price: Decimal,
        benchmark: Decimal,
        side: str,
    ) -> float:
        """Slippage in basis points. Positive = unfavorable."""
        if benchmark == Decimal("0"):
            return 0.0
        direction = Decimal("1") if side == "buy" else Decimal("-1")
        slip = (fill_price - benchmark) * direction / benchmark * Decimal("10000")
        return float(slip)
