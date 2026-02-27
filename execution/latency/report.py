"""Latency reporting — compute P50/P95/P99 statistics from tracker records."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from execution.latency.tracker import LatencyTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Percentile statistics for a single latency metric."""

    metric: str  # "signal_to_fill", "submit_to_ack", etc.
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    count: int


class LatencyReporter:
    """Compute latency statistics from tracker records."""

    def __init__(self, tracker: LatencyTracker) -> None:
        self._tracker = tracker

    def compute_stats(self) -> list[LatencyStats]:
        """Compute P50/P95/P99 for all latency metrics."""
        records = self._tracker.all_records()

        metrics = {
            "signal_to_fill": [],
            "submit_to_ack": [],
            "signal_to_decision": [],
            "decision_to_submit": [],
            "ack_to_fill": [],
        }

        for rec in records:
            for name, values in metrics.items():
                attr = f"{name}_ms"
                val: Optional[float] = getattr(rec, attr, None)
                if val is not None:
                    values.append(val)

        results: list[LatencyStats] = []
        for name, values in metrics.items():
            if not values:
                continue
            results.append(LatencyStats(
                metric=name,
                p50_ms=_percentile(values, 50),
                p95_ms=_percentile(values, 95),
                p99_ms=_percentile(values, 99),
                mean_ms=sum(values) / len(values),
                count=len(values),
            ))

        return results


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile using nearest-rank method."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = math.ceil(pct / 100.0 * len(sorted_data)) - 1
    k = max(0, min(k, len(sorted_data) - 1))
    return sorted_data[k]
