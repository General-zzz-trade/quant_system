"""TCA reporting — aggregate analysis results into summary reports."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from execution.tca.analyzer import TCAResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TCAReportEntry:
    """Summary row in a TCA report."""

    period: str  # "2024-01-15" or "2024-W03"
    symbol: str
    algo: str
    n_orders: int
    avg_slippage_bps: float
    total_fees: Decimal
    total_volume: Decimal


class TCAReporter:
    """Aggregate TCA results into summary reports."""

    def __init__(self) -> None:
        self._results: list[TCAResult] = []

    def add_result(self, result: TCAResult) -> None:
        self._results.append(result)

    def summary_by_algo(self) -> list[TCAReportEntry]:
        """Group results by algo, compute averages."""
        return self._group_by(key_fn=lambda r: ("all", "all", r.algo))

    def summary_by_symbol(self) -> list[TCAReportEntry]:
        """Group results by symbol."""
        return self._group_by(key_fn=lambda r: ("all", r.symbol, "all"))

    def summary_by_period(self, period: str = "daily") -> list[TCAReportEntry]:
        """Group results by time period."""

        def key_fn(r: TCAResult) -> tuple[str, str, str]:
            # We don't have fill timestamps on TCAResult directly,
            # so use order_id prefix as a fallback grouping key
            return (r.order_id[:10] if len(r.order_id) >= 10 else "unknown", r.symbol, r.algo)

        return self._group_by(key_fn=key_fn)

    def _group_by(
        self,
        key_fn: Any,
    ) -> list[TCAReportEntry]:
        """Group results using a key function that returns (period, symbol, algo)."""
        groups: dict[tuple[str, str, str], list[TCAResult]] = defaultdict(list)

        for r in self._results:
            key = key_fn(r)
            groups[key].append(r)

        entries: list[TCAReportEntry] = []
        for (period, symbol, algo), results in sorted(groups.items()):
            n = len(results)
            total_fees = sum(r.total_fee for r in results)
            total_volume = sum(r.total_qty for r in results)

            # Average slippage: use first available benchmark
            slippage_values: list[float] = []
            for r in results:
                if r.slippage_bps:
                    # Use the first benchmark's slippage
                    slippage_values.append(next(iter(r.slippage_bps.values())))

            avg_slippage = (
                sum(slippage_values) / len(slippage_values)
                if slippage_values
                else 0.0
            )

            entries.append(TCAReportEntry(
                period=period,
                symbol=symbol,
                algo=algo,
                n_orders=n,
                avg_slippage_bps=avg_slippage,
                total_fees=total_fees,
                total_volume=total_volume,
            ))

        return entries
