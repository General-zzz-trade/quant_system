"""Tests for statistical signals and ML signals."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace


from strategy.signals.statistical.cointegration import CointegrationSignal
from strategy.signals.statistical.zscore import ZScoreSignal
from strategy.signals.ml.features_contract import FeaturesContract
from strategy.signals.ml.model_runner import ModelRunnerSignal


# ── CointegrationSignal ──────────────────────────────────────────────

class TestCointegrationSignal:
    def _snap(self, spread_z):
        return SimpleNamespace(features={"spread_zscore": spread_z})

    def test_sell_above_threshold(self):
        sig = CointegrationSignal(threshold=Decimal("2.0"))
        r = sig.compute(self._snap(3.0), "PAIR")
        assert r.side == "sell"
        assert r.score < Decimal("0")

    def test_buy_below_threshold(self):
        sig = CointegrationSignal(threshold=Decimal("2.0"))
        r = sig.compute(self._snap(-3.0), "PAIR")
        assert r.side == "buy"
        assert r.score > Decimal("0")

    def test_flat_within_threshold(self):
        sig = CointegrationSignal(threshold=Decimal("2.0"))
        r = sig.compute(self._snap(1.0), "PAIR")
        assert r.side == "flat"
        assert r.score == Decimal("0")

    def test_no_features(self):
        snap = SimpleNamespace()
        sig = CointegrationSignal()
        r = sig.compute(snap, "PAIR")
        assert r.side == "flat"
        assert r.confidence == Decimal("0")

    def test_missing_spread_key(self):
        snap = SimpleNamespace(features={"other": 5})
        sig = CointegrationSignal()
        r = sig.compute(snap, "PAIR")
        assert r.side == "flat"

    def test_at_threshold_boundary(self):
        sig = CointegrationSignal(threshold=Decimal("2.0"))
        r = sig.compute(self._snap(2.0), "PAIR")
        assert r.side == "sell"

    def test_meta_includes_z_and_threshold(self):
        sig = CointegrationSignal(threshold=Decimal("2.0"))
        r = sig.compute(self._snap(3.0), "PAIR")
        assert "spread_z" in r.meta
        assert "threshold" in r.meta


# ── ZScoreSignal ─────────────────────────────────────────────────────

class TestZScoreSignal:
    def _snap(self, z):
        return SimpleNamespace(features={"zscore": z})

    def test_sell_above_threshold(self):
        sig = ZScoreSignal(threshold=Decimal("1.0"))
        r = sig.compute(self._snap(2.0), "X")
        assert r.side == "sell"
        assert r.score < Decimal("0")

    def test_buy_below_threshold(self):
        sig = ZScoreSignal(threshold=Decimal("1.0"))
        r = sig.compute(self._snap(-2.0), "X")
        assert r.side == "buy"
        assert r.score > Decimal("0")

    def test_flat_within(self):
        sig = ZScoreSignal(threshold=Decimal("1.0"))
        r = sig.compute(self._snap(0.5), "X")
        assert r.side == "flat"

    def test_no_features(self):
        snap = SimpleNamespace()
        sig = ZScoreSignal()
        r = sig.compute(snap, "X")
        assert r.side == "flat"

    def test_invalid_z_value(self):
        snap = SimpleNamespace(features={"zscore": "not_a_number_xyz"})
        sig = ZScoreSignal()
        r = sig.compute(snap, "X")
        assert r.side == "flat"


# ── FeaturesContract ─────────────────────────────────────────────────

class TestFeaturesContract:
    def test_validate_all_present(self):
        c = FeaturesContract(required=("a", "b"))
        ok, missing = c.validate({"a": 1, "b": 2, "c": 3})
        assert ok is True
        assert missing == []

    def test_validate_missing_keys(self):
        c = FeaturesContract(required=("a", "b", "c"))
        ok, missing = c.validate({"a": 1})
        assert ok is False
        assert set(missing) == {"b", "c"}

    def test_validate_empty_required(self):
        c = FeaturesContract(required=())
        ok, missing = c.validate({})
        assert ok is True

    def test_validate_empty_features(self):
        c = FeaturesContract(required=("x",))
        ok, missing = c.validate({})
        assert ok is False
        assert missing == ["x"]


# ── ModelRunnerSignal ────────────────────────────────────────────────

class TestModelRunnerSignal:
    def _snap(self, **feats):
        return SimpleNamespace(features=feats)

    def test_buy_positive_score(self):
        sig = ModelRunnerSignal()
        r = sig.compute(self._snap(ml_score=0.8), "X")
        assert r.side == "buy"
        assert r.score == Decimal("0.8")

    def test_sell_negative_score(self):
        sig = ModelRunnerSignal()
        r = sig.compute(self._snap(ml_score=-0.5), "X")
        assert r.side == "sell"
        assert r.score == Decimal("-0.5")

    def test_flat_zero_score(self):
        sig = ModelRunnerSignal()
        r = sig.compute(self._snap(ml_score=0), "X")
        assert r.side == "flat"

    def test_contract_failure(self):
        contract = FeaturesContract(required=("feat_a", "feat_b"))
        sig = ModelRunnerSignal(contract=contract)
        r = sig.compute(self._snap(ml_score=1.0), "X")
        assert r.side == "flat"
        assert "missing" in r.meta

    def test_contract_passes(self):
        contract = FeaturesContract(required=("ml_score",))
        sig = ModelRunnerSignal(contract=contract)
        r = sig.compute(self._snap(ml_score=0.5), "X")
        assert r.side == "buy"

    def test_no_features(self):
        sig = ModelRunnerSignal()
        snap = SimpleNamespace()
        r = sig.compute(snap, "X")
        assert r.side == "flat"

    def test_no_contract(self):
        sig = ModelRunnerSignal(contract=None)
        r = sig.compute(self._snap(ml_score=0.3), "X")
        assert r.side == "buy"
