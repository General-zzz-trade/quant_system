"""execution.latency — Order latency tracking and reporting (Domain 4: ops).

Records timestamps at each pipeline stage (signal -> decision -> submit -> ack -> fill)
and computes P50/P95/P99 statistics for performance monitoring.
"""
from execution.latency.tracker import LatencyRecord, LatencyTracker
from execution.latency.report import LatencyReporter, LatencyStats

__all__ = [
    "LatencyRecord",
    "LatencyTracker",
    "LatencyReporter",
    "LatencyStats",
]
