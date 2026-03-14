"""Tests for latency tracker and reporter."""
from __future__ import annotations

import time

import pytest

from execution.latency.tracker import LatencyRecord, LatencyTracker
from execution.latency.report import LatencyReporter


# ── LatencyRecord Tests ─────────────────────────────────────


class TestLatencyRecord:
    def test_signal_to_fill_ms(self) -> None:
        rec = LatencyRecord(order_id="o1", signal_ts=1000.0, fill_ts=1000.05)
        assert rec.signal_to_fill_ms == pytest.approx(50.0)

    def test_submit_to_ack_ms(self) -> None:
        rec = LatencyRecord(order_id="o1", submit_ts=1000.0, ack_ts=1000.01)
        assert rec.submit_to_ack_ms == pytest.approx(10.0)

    def test_none_when_missing(self) -> None:
        rec = LatencyRecord(order_id="o1", signal_ts=1000.0)
        assert rec.signal_to_fill_ms is None
        assert rec.submit_to_ack_ms is None
        assert rec.signal_to_decision_ms is None


# ── LatencyTracker Tests ────────────────────────────────────


class TestLatencyTracker:
    def test_record_all_stages(self) -> None:
        tracker = LatencyTracker()
        tracker.record_signal("o1")
        tracker.record_decision("o1")
        tracker.record_submit("o1")
        tracker.record_ack("o1")
        tracker.record_fill("o1")
        rec = tracker.get("o1")
        assert rec is not None
        assert rec.signal_ts is not None
        assert rec.decision_ts is not None
        assert rec.submit_ts is not None
        assert rec.ack_ts is not None
        assert rec.fill_ts is not None

    def test_get_nonexistent(self) -> None:
        tracker = LatencyTracker()
        assert tracker.get("missing") is None

    def test_max_records_eviction(self) -> None:
        tracker = LatencyTracker(max_records=3)
        tracker.record_signal("o1")
        tracker.record_signal("o2")
        tracker.record_signal("o3")
        tracker.record_signal("o4")
        assert tracker.get("o1") is None  # Evicted
        assert tracker.get("o4") is not None

    def test_all_records(self) -> None:
        tracker = LatencyTracker()
        tracker.record_signal("a")
        tracker.record_signal("b")
        records = tracker.all_records()
        assert len(records) == 2

    def test_signal_to_fill_timing(self) -> None:
        tracker = LatencyTracker()
        tracker.record_signal("o1")
        time.sleep(0.01)
        tracker.record_fill("o1")
        rec = tracker.get("o1")
        assert rec is not None
        assert rec.signal_to_fill_ms is not None
        assert rec.signal_to_fill_ms >= 5.0  # At least ~5ms


# ── LatencyReporter Tests ──────────────────────────────────


class TestLatencyReporter:
    def test_compute_stats_empty(self) -> None:
        tracker = LatencyTracker()
        reporter = LatencyReporter(tracker)
        stats = reporter.compute_stats()
        assert stats == []

    def test_compute_stats_percentiles(self) -> None:
        tracker = LatencyTracker()
        # Inject records with known timings
        for i in range(100):
            oid = f"o{i}"
            tracker.record_signal(oid)
        # Small sleep to get measurable deltas
        time.sleep(0.01)
        for i in range(100):
            oid = f"o{i}"
            tracker.record_fill(oid)

        reporter = LatencyReporter(tracker)
        stats = reporter.compute_stats()
        s2f = next((s for s in stats if s.metric == "signal_to_fill"), None)
        assert s2f is not None
        assert s2f.count == 100
        assert s2f.p50_ms > 0
        assert s2f.p95_ms >= s2f.p50_ms
        assert s2f.p99_ms >= s2f.p95_ms
        assert s2f.mean_ms > 0

    def test_stats_only_for_available_metrics(self) -> None:
        tracker = LatencyTracker()
        tracker.record_submit("o1")
        time.sleep(0.005)
        tracker.record_ack("o1")
        reporter = LatencyReporter(tracker)
        stats = reporter.compute_stats()
        metrics = [s.metric for s in stats]
        assert "submit_to_ack" in metrics
        assert "signal_to_fill" not in metrics
