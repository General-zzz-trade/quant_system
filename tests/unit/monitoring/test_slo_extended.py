"""Extended tests for SLOTracker — latency, fill rate, violations, error budget."""
from __future__ import annotations

import pytest

from monitoring.slo import SLOConfig, SLOTracker


class TestSLOLatencyTracking:
    def test_p99_with_single_spike(self):
        tracker = SLOTracker(SLOConfig(pipeline_latency_p99_sec=1.0))
        for _ in range(99):
            tracker.record_latency(0.05)
        tracker.record_latency(3.0)
        snap = tracker.snapshot()
        assert snap.pipeline_latency_p99_ms >= 3000.0

    def test_p99_all_low_no_violation(self):
        tracker = SLOTracker(SLOConfig(pipeline_latency_p99_sec=1.0))
        for _ in range(200):
            tracker.record_latency(0.01)
        snap = tracker.snapshot()
        assert snap.pipeline_latency_p99_ms < 1000.0
        assert not any("latency" in v for v in snap.violations)

    def test_empty_latencies_returns_zero(self):
        tracker = SLOTracker()
        snap = tracker.snapshot()
        assert snap.pipeline_latency_p99_ms == 0.0


class TestSLOFillRate:
    def test_fill_rate_perfect(self):
        tracker = SLOTracker()
        for _ in range(20):
            tracker.record_order(filled=True)
        snap = tracker.snapshot()
        assert snap.fill_rate == pytest.approx(1.0)
        assert snap.total_orders == 20
        assert snap.filled_orders == 20

    def test_fill_rate_zero(self):
        tracker = SLOTracker()
        for _ in range(5):
            tracker.record_order(filled=False)
        snap = tracker.snapshot()
        assert snap.fill_rate == pytest.approx(0.0)

    def test_fill_rate_violation(self):
        tracker = SLOTracker(SLOConfig(fill_rate_target=0.90))
        for _ in range(7):
            tracker.record_order(filled=True)
        for _ in range(3):
            tracker.record_order(filled=False)
        snap = tracker.snapshot()
        assert snap.fill_rate == pytest.approx(0.7, abs=0.01)
        assert any("fill_rate" in v for v in snap.violations)

    def test_no_orders_gives_fill_rate_one(self):
        tracker = SLOTracker()
        snap = tracker.snapshot()
        assert snap.fill_rate == 1.0


class TestSLOViolationDetection:
    def test_availability_violation(self):
        tracker = SLOTracker(SLOConfig(availability_target=0.99))
        for _ in range(90):
            tracker.record_uptime_check(True)
        for _ in range(10):
            tracker.record_uptime_check(False)
        snap = tracker.snapshot()
        assert any("availability" in v for v in snap.violations)

    def test_multiple_violations_reported(self):
        tracker = SLOTracker(SLOConfig(
            pipeline_latency_p99_sec=0.01,
            fill_rate_target=0.99,
        ))
        for _ in range(100):
            tracker.record_latency(1.0)
        tracker.record_order(filled=False)
        snap = tracker.snapshot()
        assert len(snap.violations) >= 2


class TestSLOErrorBudget:
    def test_full_budget_when_fully_up(self):
        tracker = SLOTracker(SLOConfig(availability_target=0.99, window_sec=100))
        for _ in range(100):
            tracker.record_uptime_check(True)
        snap = tracker.snapshot()
        assert snap.error_budget_remaining_pct == pytest.approx(100.0)

    def test_budget_depletes_with_downtime(self):
        tracker = SLOTracker(SLOConfig(availability_target=0.99, window_sec=100))
        for _ in range(95):
            tracker.record_uptime_check(True)
        for _ in range(5):
            tracker.record_uptime_check(False)
        snap = tracker.snapshot()
        # 5% downtime vs 1% budget -> budget should be depleted
        assert snap.error_budget_remaining_pct == 0.0

    def test_budget_100_with_no_checks(self):
        tracker = SLOTracker()
        snap = tracker.snapshot()
        assert snap.error_budget_remaining_pct == 100.0
