# tests/unit/decision/test_ensemble_combiner.py
"""Tests for Direction 13: EnsembleCombiner multi-timeframe signal fusion."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from decision.ensemble_combiner import EnsembleCombiner


class FakeBridge:
    """Fake inference bridge that returns a fixed score via enrich()."""

    def __init__(self, score: float, score_key: str = "ml_score"):
        self._score = score
        self._score_key = score_key

    def enrich(
        self, symbol: str, ts: Optional[datetime], features: Dict[str, Any]
    ) -> Dict[str, Any]:
        features[self._score_key] = self._score
        return features

    def checkpoint(self) -> dict:
        return {"score": self._score}

    def restore(self, data: dict) -> None:
        self._score = data.get("score", self._score)


@pytest.fixture
def ts():
    return datetime(2026, 3, 13, tzinfo=timezone.utc)


class TestEnsembleCombinerBasic:
    """Basic combining and weighting tests."""

    def test_equal_weight_average_same_direction(self, ts):
        """Two bridges both positive -> weighted average."""
        b1 = FakeBridge(0.8)
        b2 = FakeBridge(0.4)
        combiner = EnsembleCombiner(bridges=[("h1", b1), ("m15", b2)])

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        assert features["ml_score"] == pytest.approx(0.6)
        assert features["ml_score_h1"] == pytest.approx(0.8)
        assert features["ml_score_m15"] == pytest.approx(0.4)

    def test_equal_weight_average_both_negative(self, ts):
        """Two bridges both negative -> weighted average (not conflict)."""
        b1 = FakeBridge(-0.6)
        b2 = FakeBridge(-0.2)
        combiner = EnsembleCombiner(bridges=[("h1", b1), ("m15", b2)])

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        assert features["ml_score"] == pytest.approx(-0.4)

    def test_custom_weights(self, ts):
        """Custom weights: heavier on first bridge."""
        b1 = FakeBridge(1.0)
        b2 = FakeBridge(0.0)
        combiner = EnsembleCombiner(
            bridges=[("h1", b1), ("m15", b2)],
            weights=[3.0, 1.0],
        )

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        # (1.0*3 + 0.0*1) / (3+1) = 0.75
        assert features["ml_score"] == pytest.approx(0.75)


class TestEnsembleCombinerConflict:
    """Conflict detection tests."""

    def test_conflict_flat_policy(self, ts):
        """One positive, one negative -> flat (0.0) with 'flat' policy."""
        b1 = FakeBridge(0.5)
        b2 = FakeBridge(-0.3)
        combiner = EnsembleCombiner(
            bridges=[("h1", b1), ("m15", b2)],
            conflict_policy="flat",
        )

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        assert features["ml_score"] == 0.0
        # Per-bridge scores are still stored for attribution
        assert features["ml_score_h1"] == pytest.approx(0.5)
        assert features["ml_score_m15"] == pytest.approx(-0.3)

    def test_conflict_average_policy(self, ts):
        """One positive, one negative -> average with 'average' policy."""
        b1 = FakeBridge(0.5)
        b2 = FakeBridge(-0.3)
        combiner = EnsembleCombiner(
            bridges=[("h1", b1), ("m15", b2)],
            conflict_policy="average",
        )

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        assert features["ml_score"] == pytest.approx(0.1)

    def test_one_zero_one_positive_no_conflict(self, ts):
        """One zero, one positive -> no conflict (zero has no sign)."""
        b1 = FakeBridge(0.0)
        b2 = FakeBridge(0.5)
        combiner = EnsembleCombiner(
            bridges=[("h1", b1), ("m15", b2)],
            conflict_policy="flat",
        )

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        # Only one sign (positive) -> no conflict -> average
        assert features["ml_score"] == pytest.approx(0.25)


class TestEnsembleCombinerWeightOverrides:
    """Dynamic weight override tests."""

    def test_update_weight_override(self, ts):
        b1 = FakeBridge(1.0)
        b2 = FakeBridge(1.0)
        combiner = EnsembleCombiner(bridges=[("h1", b1), ("m15", b2)])

        # Override m15 weight to 0 -> only h1 contributes
        combiner.update_weight("m15", 0.0)

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        # m15 has weight 0 -> filtered out; only h1 (w=1.0) contributes
        assert features["ml_score"] == pytest.approx(1.0)

    def test_get_weights_with_override(self):
        b1 = FakeBridge(0.0)
        b2 = FakeBridge(0.0)
        combiner = EnsembleCombiner(
            bridges=[("h1", b1), ("m15", b2)],
            weights=[1.0, 1.0],
        )
        combiner.update_weight("m15", 0.5)

        weights = combiner.get_weights()
        assert weights["h1"] == 1.0
        assert weights["m15"] == 0.5


class TestEnsembleCombinerCheckpoint:
    """Checkpoint and restore tests."""

    def test_checkpoint_restore(self, ts):
        b1 = FakeBridge(0.5)
        b2 = FakeBridge(0.3)
        combiner = EnsembleCombiner(bridges=[("h1", b1), ("m15", b2)])
        combiner.update_weight("m15", 0.7)

        state = combiner.checkpoint()
        assert "bridge_states" in state
        assert "weight_overrides" in state
        assert state["weight_overrides"]["m15"] == 0.7

        # Restore into a fresh combiner
        b1_new = FakeBridge(0.0)
        b2_new = FakeBridge(0.0)
        combiner2 = EnsembleCombiner(bridges=[("h1", b1_new), ("m15", b2_new)])
        combiner2.restore(state)

        assert combiner2._weight_overrides["m15"] == 0.7


class TestEnsembleCombinerEdgeCases:
    """Edge cases."""

    def test_single_bridge(self, ts):
        """Single bridge -> passthrough."""
        b1 = FakeBridge(0.42)
        combiner = EnsembleCombiner(bridges=[("only", b1)])

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        assert features["ml_score"] == pytest.approx(0.42)

    def test_weight_length_mismatch_raises(self):
        b1 = FakeBridge(0.0)
        with pytest.raises(ValueError, match="weights length"):
            EnsembleCombiner(bridges=[("a", b1)], weights=[1.0, 2.0])

    def test_all_zero_weights(self, ts):
        b1 = FakeBridge(0.5)
        b2 = FakeBridge(0.3)
        combiner = EnsembleCombiner(
            bridges=[("h1", b1), ("m15", b2)],
            weights=[0.0, 0.0],
        )

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        assert features["ml_score"] == 0.0

    def test_no_bridges(self, ts):
        combiner = EnsembleCombiner(bridges=[])

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        assert features["ml_score"] == 0.0

    def test_three_bridges_partial_conflict(self, ts):
        """Three bridges: 2 positive, 1 negative -> conflict -> flat."""
        b1 = FakeBridge(0.5)
        b2 = FakeBridge(0.3)
        b3 = FakeBridge(-0.1)
        combiner = EnsembleCombiner(
            bridges=[("a", b1), ("b", b2), ("c", b3)],
            conflict_policy="flat",
        )

        features: Dict[str, Any] = {}
        combiner.enrich("BTCUSDT", ts, features)

        assert features["ml_score"] == 0.0
