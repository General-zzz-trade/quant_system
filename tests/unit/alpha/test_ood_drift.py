"""Tests for OODDetector and ConceptDriftAdapter."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from alpha.monitoring.ood_detector import OODDetector
from alpha.monitoring.drift_adapter import ConceptDriftAdapter


# ── OODDetector ───────────────────────────────────────────────


class TestOODDetector:
    def _normal_data(self, n: int = 200, seed: int = 42) -> list[dict[str, float]]:
        rng = random.Random(seed)
        return [{"f1": rng.gauss(0, 1), "f2": rng.gauss(100, 10)} for _ in range(n)]

    def test_fit_and_score_in_distribution(self):
        det = OODDetector(z_threshold=3.0)
        data = self._normal_data()
        det.fit(data)

        # A point near the mean should not be OOD
        result = det.score({"f1": 0.1, "f2": 101.0})
        assert not result.is_ood
        assert result.score < 3.0
        assert result.method == "mahalanobis_diag"

    def test_score_ood_point(self):
        det = OODDetector(z_threshold=3.0)
        det.fit(self._normal_data())

        # A point 10 sigma away should be OOD
        result = det.score({"f1": 15.0, "f2": 300.0})
        assert result.is_ood
        assert result.score > 3.0

    def test_score_partial_features(self):
        det = OODDetector(z_threshold=3.0)
        det.fit(self._normal_data())

        # Only one feature present — should still score
        result = det.score({"f1": 0.0})
        assert not result.is_ood

    def test_score_no_overlap_is_ood(self):
        det = OODDetector(z_threshold=3.0)
        det.fit(self._normal_data())

        # No matching features → maximally OOD
        result = det.score({"unknown_feat": 42.0})
        assert result.is_ood
        assert result.score == float("inf")

    def test_fit_empty_raises(self):
        det = OODDetector()
        with pytest.raises(ValueError, match="empty"):
            det.fit([])

    def test_fit_no_numeric_raises(self):
        det = OODDetector()
        with pytest.raises(ValueError, match="No valid numeric"):
            det.fit([{"a": float("nan")}])

    def test_fit_single_obs_raises(self):
        det = OODDetector()
        with pytest.raises(ValueError, match="at least 2"):
            det.fit([{"a": 1.0}])

    def test_score_before_fit_raises(self):
        det = OODDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.score({"f1": 1.0})

    def test_save_load_roundtrip(self, tmp_path: Path):
        det = OODDetector(z_threshold=2.5)
        det.fit(self._normal_data())

        path = tmp_path / "ood_params.json"
        det.save(path)

        det2 = OODDetector()
        det2.load(path)

        # Same results after reload
        point = {"f1": 0.5, "f2": 105.0}
        r1 = det.score(point)
        r2 = det2.score(point)
        assert r1.is_ood == r2.is_ood
        assert abs(r1.score - r2.score) < 1e-9
        assert r1.threshold == r2.threshold

    def test_save_unfitted_raises(self, tmp_path: Path):
        det = OODDetector()
        with pytest.raises(RuntimeError, match="unfitted"):
            det.save(tmp_path / "bad.json")

    def test_fitted_property(self):
        det = OODDetector()
        assert not det.fitted
        det.fit(self._normal_data(n=10))
        assert det.fitted

    def test_nan_in_features_ignored(self):
        det = OODDetector(z_threshold=3.0)
        data = [{"f1": 1.0, "f2": float("nan")} for _ in range(10)]
        det.fit(data)
        # f2 should be excluded since all values are NaN
        assert "f2" not in det._mean
        result = det.score({"f1": 1.0})
        assert not result.is_ood


# ── ConceptDriftAdapter ──────────────────────────────────────


class TestConceptDriftAdapter:
    def test_no_drift_with_good_predictions(self):
        adapter = ConceptDriftAdapter(window=50, baseline_window=100)

        # Feed mix of correct long/short predictions with genuine variance
        rng = random.Random(42)
        for _ in range(200):
            ret = rng.gauss(0.0, 0.02)  # centered at 0, so roughly 50/50 long/short
            side = "long" if ret > 0 else "short"
            adapter.on_prediction(side, ret)

        state = adapter.check()
        assert not state.is_drifting
        assert state.severity == "none"
        assert state.recommendation == "continue"

    def test_insufficient_data_returns_no_drift(self):
        adapter = ConceptDriftAdapter(window=100)
        adapter.on_prediction("long", 0.01)
        state = adapter.check()
        assert not state.is_drifting
        assert state.recommendation == "continue"
        assert state.metrics == {}

    def test_drift_detected_with_bad_predictions(self):
        adapter = ConceptDriftAdapter(
            window=50,
            baseline_window=100,
            sharpe_floor=0.0,
            ic_floor=0.02,
            hit_rate_floor=0.48,
        )

        # Good baseline phase
        rng = random.Random(42)
        for _ in range(100):
            adapter.on_prediction("long", abs(rng.gauss(0.01, 0.005)))

        # Bad recent phase — all predictions wrong
        for _ in range(60):
            adapter.on_prediction("long", -abs(rng.gauss(0.01, 0.005)))

        state = adapter.check()
        assert state.is_drifting
        assert state.recommendation in ("pause", "retrain")

    def test_recommendation_escalation(self):
        """Test that severity escalates as more metrics degrade."""
        # Only hit rate degraded (sharpe and IC may still be ok)
        adapter = ConceptDriftAdapter(
            window=50,
            baseline_window=100,
            sharpe_floor=-999.0,  # effectively disabled
            ic_floor=-999.0,      # effectively disabled
            hit_rate_floor=0.99,   # very high bar, will be breached
        )

        rng = random.Random(42)
        for _ in range(200):
            ret = rng.gauss(0.01, 0.02)
            adapter.on_prediction("long", ret)

        state = adapter.check()
        # Only one metric below floor → warning
        assert state.severity == "warning"
        assert state.recommendation == "reduce_size"

    def test_all_metrics_degraded_recommends_retrain(self):
        adapter = ConceptDriftAdapter(
            window=50,
            baseline_window=100,
            sharpe_floor=999.0,   # impossible to meet
            ic_floor=999.0,       # impossible to meet
            hit_rate_floor=0.99,  # impossible to meet
        )

        rng = random.Random(42)
        for _ in range(200):
            adapter.on_prediction("long", rng.gauss(0, 0.01))

        state = adapter.check()
        assert state.is_drifting
        assert state.severity == "critical"
        assert state.recommendation == "retrain"

    def test_two_metrics_degraded_recommends_pause(self):
        adapter = ConceptDriftAdapter(
            window=50,
            baseline_window=100,
            sharpe_floor=999.0,   # impossible
            ic_floor=999.0,       # impossible
            hit_rate_floor=0.0,   # easy to meet
        )

        rng = random.Random(42)
        for _ in range(200):
            ret = abs(rng.gauss(0.01, 0.005))
            adapter.on_prediction("long", ret)

        state = adapter.check()
        assert state.severity == "critical"
        assert state.recommendation == "pause"

    def test_reset_baseline(self):
        adapter = ConceptDriftAdapter(window=50, baseline_window=100)

        rng = random.Random(42)
        for _ in range(150):
            adapter.on_prediction("long", abs(rng.gauss(0.01, 0.005)))

        assert adapter._baseline_frozen
        adapter.reset_baseline()
        assert not adapter._baseline_frozen
        assert len(adapter._baseline_hits) == 0

    def test_metrics_present_in_state(self):
        adapter = ConceptDriftAdapter(window=50, baseline_window=100)

        rng = random.Random(42)
        for _ in range(200):
            adapter.on_prediction("long", abs(rng.gauss(0.01, 0.005)))

        state = adapter.check()
        assert "rolling_hit_rate" in state.metrics
        assert "rolling_ic" in state.metrics
        assert "rolling_sharpe" in state.metrics

    def test_short_predictions_counted_correctly(self):
        adapter = ConceptDriftAdapter(window=50, baseline_window=100, ic_floor=-1.0)

        rng = random.Random(42)
        for _ in range(200):
            ret = -abs(rng.gauss(0.01, 0.005))  # negative returns
            adapter.on_prediction("short", ret)  # short + negative = hit

        state = adapter.check()
        assert state.metrics["rolling_hit_rate"] > 0.9
        assert not state.is_drifting
