"""Tests for data quality alerting."""
from __future__ import annotations

import math
import pytest
from monitoring.data_quality_alerts import DataQualityConfig, DataQualityMonitor


class TestDataQualityMonitor:
    def test_nan_detection(self):
        mon = DataQualityMonitor(DataQualityConfig(nan_fraction_threshold=0.1, min_observations=10))
        for i in range(10):
            features = {"price": 100.0 + i, "volume": float("nan")}
            issues = mon.on_features(features, "BTC")
        # volume is always NaN → 100% NaN rate
        assert any("NaN rate" in issue for issue in issues)

    def test_no_issues_for_clean_data(self):
        mon = DataQualityMonitor(DataQualityConfig(min_observations=5))
        for i in range(10):
            issues = mon.on_features({"price": 100.0 + i * 0.1}, "BTC")
        assert issues == []

    def test_distribution_shift(self):
        mon = DataQualityMonitor(DataQualityConfig(
            mean_shift_z_threshold=2.0,
            window_size=20,
            min_observations=5,
        ))
        # Build baseline
        for i in range(25):
            mon.on_features({"price": 100.0 + (i % 5) * 0.1}, "BTC")
        # Shift mean dramatically
        issues: list[str] = []
        for i in range(25):
            issues = mon.on_features({"price": 200.0 + (i % 5) * 0.1}, "BTC")
        assert any("distribution shift" in issue for issue in issues)

    def test_constant_feature_detected(self):
        mon = DataQualityMonitor(DataQualityConfig(
            window_size=15,
            min_observations=15,
        ))
        # Build baseline (varied)
        for i in range(20):
            mon.on_features({"feat": float(i)}, "BTC")
        # Now constant
        issues: list[str] = []
        for i in range(20):
            issues = mon.on_features({"feat": 42.0}, "BTC")
        assert any("zero variance" in issue for issue in issues)

    def test_get_stats(self):
        mon = DataQualityMonitor(DataQualityConfig(window_size=10, min_observations=5))
        for i in range(15):
            mon.on_features({"price": 100.0 + i}, "BTC")
        stats = mon.get_stats()
        assert "price" in stats
        assert stats["price"]["count"] > 0
        assert stats["price"]["nan_rate"] == 0.0
