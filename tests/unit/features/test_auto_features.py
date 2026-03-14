"""Tests for auto feature generation and selection."""
from __future__ import annotations


import pytest

from features.auto.generator import FeatureGenerator
from features.auto.selector import FeatureSelector


class TestFeatureGenerator:
    def test_register_operator(self):
        gen = FeatureGenerator()
        gen.register_operator("sma", lambda bars, w: [0.0], category="technical")
        assert gen.operator_count == 1
        assert len(gen.explicit_candidates) == 1
        assert gen.explicit_candidates[0].name == "sma"

    def test_generate_candidates_default_windows(self):
        def sma(bars, window):
            return [0.0] * len(bars)

        gen = FeatureGenerator()
        gen.register_operator("sma", sma)
        candidates = gen.generate_candidates()

        auto_candidates = [c for c in candidates if c.category == "auto"]
        assert len(auto_candidates) == 4  # 4 default windows
        names = {c.name for c in auto_candidates}
        assert "sma_5" in names
        assert "sma_10" in names
        assert "sma_20" in names
        assert "sma_50" in names

    def test_generate_candidates_custom_windows(self):
        gen = FeatureGenerator()
        gen.register_operator("rsi", lambda bars, w: [0.0])
        candidates = gen.generate_candidates(windows=(7, 14, 21))

        auto_names = {c.name for c in candidates if c.category == "auto"}
        assert auto_names == {"rsi_7", "rsi_14", "rsi_21"}

    def test_generated_compute_fn_captures_window(self):
        results = {}

        def mock_op(bars, window):
            results[window] = len(bars)
            return [float(window)]

        gen = FeatureGenerator()
        gen.register_operator("test_op", mock_op)
        candidates = gen.generate_candidates(windows=(5, 10))

        auto_candidates = [c for c in candidates if c.category == "auto"]
        for c in auto_candidates:
            c.compute_fn([1, 2, 3])

        assert 5 in results
        assert 10 in results

    def test_multiple_operators(self):
        gen = FeatureGenerator()
        gen.register_operator("sma", lambda b, w: [0.0])
        gen.register_operator("ema", lambda b, w: [0.0])
        candidates = gen.generate_candidates(windows=(10, 20))

        auto_names = {c.name for c in candidates if c.category == "auto"}
        assert len(auto_names) == 4
        assert "sma_10" in auto_names
        assert "ema_20" in auto_names

    def test_empty_generator(self):
        gen = FeatureGenerator()
        candidates = gen.generate_candidates()
        assert candidates == []


class TestFeatureSelector:
    def test_correlation_select_perfect_positive(self):
        features = {"feat_a": [1.0, 2.0, 3.0, 4.0, 5.0]}
        target = [2.0, 4.0, 6.0, 8.0, 10.0]

        selector = FeatureSelector(method="correlation", top_k=5)
        scores = selector.select(features, target)

        assert len(scores) == 1
        assert scores[0].name == "feat_a"
        assert scores[0].score == pytest.approx(1.0, abs=1e-6)

    def test_correlation_select_negative(self):
        features = {"neg": [5.0, 4.0, 3.0, 2.0, 1.0]}
        target = [1.0, 2.0, 3.0, 4.0, 5.0]

        selector = FeatureSelector(method="correlation", top_k=5)
        scores = selector.select(features, target)

        assert scores[0].score == pytest.approx(1.0, abs=1e-6)

    def test_correlation_select_ranking(self):
        features = {
            "strong": [1.0, 2.0, 3.0, 4.0, 5.0],
            "weak": [1.0, 1.1, 0.9, 1.0, 1.05],
            "noise": [3.0, 1.0, 4.0, 1.0, 5.0],
        }
        target = [1.0, 2.0, 3.0, 4.0, 5.0]

        selector = FeatureSelector(method="correlation", top_k=2)
        scores = selector.select(features, target)

        assert len(scores) == 2
        assert scores[0].name == "strong"

    def test_mutual_info_select(self):
        features = {
            "informative": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            "constant": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
        target = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

        selector = FeatureSelector(method="mutual_info", top_k=2)
        scores = selector.select(features, target)

        assert len(scores) == 2
        assert scores[0].name == "informative"
        assert scores[0].score >= scores[1].score

    def test_unknown_method_raises(self):
        selector = FeatureSelector(method="unknown")
        with pytest.raises(ValueError, match="Unknown method"):
            selector.select({"a": [1.0]}, [1.0])

    def test_length_mismatch_skipped(self):
        features = {"bad": [1.0, 2.0]}
        target = [1.0, 2.0, 3.0]

        selector = FeatureSelector(method="correlation", top_k=5)
        scores = selector.select(features, target)

        assert len(scores) == 0

    def test_top_k_limit(self):
        features = {f"f{i}": [float(i)] * 10 for i in range(20)}
        target = list(range(10))

        selector = FeatureSelector(method="correlation", top_k=3)
        scores = selector.select(features, [float(x) for x in target])

        assert len(scores) <= 3
