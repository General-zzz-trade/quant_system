"""Tests for alpha/monitoring/drift.py — ModelDriftMonitor."""
from __future__ import annotations

from typing import List, Tuple

import pytest

from alpha.monitoring.drift import DriftAlert, ModelDriftMonitor


class TestModelDriftMonitor:
    def test_no_alert_with_stable_features(self):
        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 5.0)},
            window_size=10,
            z_threshold=2.0,
        )
        # Feed values close to baseline mean
        for _ in range(10):
            monitor.on_features({"f1": 100.5})
        alerts = monitor.check()
        assert alerts == []

    def test_alert_on_significant_shift(self):
        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 5.0)},
            window_size=10,
            z_threshold=2.0,
        )
        # Feed values far from baseline mean (z > 2)
        for _ in range(10):
            monitor.on_features({"f1": 120.0})  # z = |120-100|/5 = 4.0
        alerts = monitor.check()
        assert len(alerts) == 1
        assert alerts[0].feature == "f1"
        assert alerts[0].z_score > 2.0

    def test_multiple_features_tracked_independently(self):
        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 5.0), "f2": (50.0, 2.0)},
            window_size=10,
            z_threshold=2.0,
        )
        for _ in range(10):
            monitor.on_features({"f1": 100.0, "f2": 60.0})  # f2 drifted, z = |60-50|/2 = 5.0
        alerts = monitor.check()
        # Only f2 should alert
        names = [a.feature for a in alerts]
        assert "f2" in names
        assert "f1" not in names

    def test_window_size_respected(self):
        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 5.0)},
            window_size=20,
            z_threshold=2.0,
        )
        # Only 5 observations (< window_size // 2 = 10), should not alert
        for _ in range(5):
            monitor.on_features({"f1": 200.0})
        alerts = monitor.check()
        assert alerts == []

    def test_window_half_enough_to_check(self):
        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 5.0)},
            window_size=20,
            z_threshold=2.0,
        )
        # 10 observations (= window_size // 2), should check
        for _ in range(10):
            monitor.on_features({"f1": 200.0})
        alerts = monitor.check()
        assert len(alerts) == 1

    def test_alert_count_property(self):
        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 5.0)},
            window_size=10,
            z_threshold=2.0,
        )
        assert monitor.alert_count == 0
        for _ in range(10):
            monitor.on_features({"f1": 150.0})
        monitor.check()
        assert monitor.alert_count == 1
        # Check again — same window, same alert
        monitor.check()
        assert monitor.alert_count == 2

    def test_alert_fn_called(self):
        fired: List[Tuple[str, str]] = []

        def _alert(category: str, msg: str) -> None:
            fired.append((category, msg))

        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 5.0)},
            window_size=10,
            z_threshold=2.0,
            alert_fn=_alert,
        )
        for _ in range(10):
            monitor.on_features({"f1": 150.0})
        monitor.check()
        assert len(fired) == 1
        assert fired[0][0] == "model_drift"
        assert "f1" in fired[0][1]

    def test_unknown_features_ignored(self):
        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 5.0)},
            window_size=10,
        )
        for _ in range(10):
            monitor.on_features({"unknown_feat": 999.0})
        alerts = monitor.check()
        assert alerts == []

    def test_none_value_skipped(self):
        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 5.0)},
            window_size=10,
        )
        for _ in range(10):
            monitor.on_features({"f1": None})
        # Window should be empty
        assert "f1" not in monitor._windows

    def test_zero_std_skipped(self):
        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 0.0)},  # std=0
            window_size=10,
            z_threshold=2.0,
        )
        for _ in range(10):
            monitor.on_features({"f1": 200.0})
        alerts = monitor.check()
        assert alerts == []

    def test_alert_detail_fields(self):
        monitor = ModelDriftMonitor(
            baseline_stats={"f1": (100.0, 5.0)},
            window_size=10,
            z_threshold=2.0,
        )
        for _ in range(10):
            monitor.on_features({"f1": 120.0})
        alerts = monitor.check()
        a = alerts[0]
        assert a.baseline_mean == 100.0
        assert a.baseline_std == 5.0
        assert abs(a.current_mean - 120.0) < 0.01
        assert abs(a.z_score - 4.0) < 0.01
