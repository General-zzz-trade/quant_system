"""Tests for SLO/SLI tracker."""
from __future__ import annotations

import pytest
from monitoring.slo import SLOConfig, SLOTracker


class TestSLOTracker:
    def test_record_latency_p99(self):
        tracker = SLOTracker(SLOConfig(pipeline_latency_p99_sec=1.0))
        for i in range(100):
            tracker.record_latency(0.1)
        tracker.record_latency(2.0)  # spike
        snap = tracker.snapshot()
        assert snap.pipeline_latency_p99_ms > 0

    def test_record_uptime(self):
        tracker = SLOTracker()
        for _ in range(90):
            tracker.record_uptime_check(True)
        for _ in range(10):
            tracker.record_uptime_check(False)
        snap = tracker.snapshot()
        assert snap.uptime_fraction == pytest.approx(0.9, abs=0.01)

    def test_record_fill_rate(self):
        tracker = SLOTracker()
        for _ in range(8):
            tracker.record_order(filled=True)
        for _ in range(2):
            tracker.record_order(filled=False)
        snap = tracker.snapshot()
        assert snap.fill_rate == pytest.approx(0.8, abs=0.01)
        assert snap.total_orders == 10
        assert snap.filled_orders == 8

    def test_violation_detected(self):
        tracker = SLOTracker(SLOConfig(
            pipeline_latency_p99_sec=0.1,
            availability_target=0.999,
            fill_rate_target=0.95,
        ))
        for _ in range(100):
            tracker.record_latency(0.5)  # 500ms > 100ms target
        snap = tracker.snapshot()
        assert len(snap.violations) > 0
        assert any("latency" in v for v in snap.violations)

    def test_no_violation_when_healthy(self):
        tracker = SLOTracker()
        for _ in range(100):
            tracker.record_latency(0.01)
            tracker.record_uptime_check(True)
            tracker.record_order(filled=True)
        tracker.record_market_data()
        snap = tracker.snapshot()
        assert len(snap.violations) == 0

    def test_error_budget(self):
        tracker = SLOTracker(SLOConfig(availability_target=0.99, window_sec=100))
        for _ in range(100):
            tracker.record_uptime_check(True)
        snap = tracker.snapshot()
        assert snap.error_budget_remaining_pct > 0
