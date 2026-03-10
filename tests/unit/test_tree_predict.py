"""Tests for RustTreePredictor — LightGBM/XGBoost native Rust inference."""
import json
import math
import tempfile
from pathlib import Path

import pytest

_quant_hotpath = pytest.importorskip("_quant_hotpath")
from _quant_hotpath import RustTreePredictor


# ── Helpers ──

def _make_lgbm_json(trees, features=None, num_features=3, is_classifier=False):
    if features is None:
        features = [f"f{i}" for i in range(num_features)]
    return json.dumps({
        "format": "lgbm",
        "features": features,
        "num_features": num_features,
        "num_trees": len(trees),
        "is_classifier": is_classifier,
        "base_score": 0.0,
        "trees": trees,
    })


def _make_xgb_json(trees, features=None, num_features=3, base_score=0.5):
    if features is None:
        features = [f"f{i}" for i in range(num_features)]
    return json.dumps({
        "format": "xgb",
        "features": features,
        "num_features": num_features,
        "num_trees": len(trees),
        "is_classifier": False,
        "base_score": base_score,
        "trees": trees,
    })


def _single_leaf(value, shrinkage=1.0):
    return {"shrinkage": shrinkage, "nodes": [{"type": "leaf", "value": value}]}


def _simple_split(feature, threshold, left_val, right_val, shrinkage=1.0,
                  default_left=True, nan_as_zero=False):
    return {
        "shrinkage": shrinkage,
        "nodes": [
            {"type": "split", "feature": feature, "threshold": threshold,
             "default_left": default_left, "nan_as_zero": nan_as_zero,
             "left": 1, "right": 2},
            {"type": "leaf", "value": left_val},
            {"type": "leaf", "value": right_val},
        ],
    }


# ── Loading ──

class TestLoading:
    def test_from_json_lgbm(self):
        model_json = _make_lgbm_json([_single_leaf(0.5)])
        m = RustTreePredictor.from_json(model_json)
        assert m.num_trees() == 1
        assert m.feature_names() == ["f0", "f1", "f2"]
        assert not m.is_classifier()

    def test_from_json_xgb(self):
        model_json = _make_xgb_json([_single_leaf(0.1)], base_score=0.5)
        m = RustTreePredictor.from_json(model_json)
        assert m.num_trees() == 1
        assert "xgb" in m.info()

    def test_load_from_file(self, tmp_path):
        p = tmp_path / "model.json"
        p.write_text(_make_lgbm_json([_single_leaf(1.0)]))
        m = RustTreePredictor.load(str(p))
        assert m.num_trees() == 1

    def test_load_missing_file(self):
        with pytest.raises(OSError):
            RustTreePredictor.load("/nonexistent/path.json")

    def test_load_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        with pytest.raises(ValueError):
            RustTreePredictor.load(str(p))


# ── Single-leaf prediction ──

class TestLeafPrediction:
    def test_lgbm_single_leaf(self):
        m = RustTreePredictor.from_json(_make_lgbm_json([_single_leaf(0.42)]))
        assert m.predict_array([1.0, 2.0, 3.0]) == pytest.approx(0.42)

    def test_lgbm_multiple_leaves_sum(self):
        trees = [_single_leaf(0.1), _single_leaf(0.2), _single_leaf(0.3)]
        m = RustTreePredictor.from_json(_make_lgbm_json(trees))
        assert m.predict_array([0.0, 0.0, 0.0]) == pytest.approx(0.6)

    def test_lgbm_shrinkage(self):
        m = RustTreePredictor.from_json(
            _make_lgbm_json([_single_leaf(1.0, shrinkage=0.1)])
        )
        assert m.predict_array([0.0, 0.0, 0.0]) == pytest.approx(0.1)

    def test_xgb_base_score(self):
        m = RustTreePredictor.from_json(
            _make_xgb_json([_single_leaf(0.0)], base_score=0.5)
        )
        assert m.predict_array([0.0, 0.0, 0.0]) == pytest.approx(0.5)

    def test_xgb_base_score_plus_tree(self):
        m = RustTreePredictor.from_json(
            _make_xgb_json([_single_leaf(0.3)], base_score=0.5)
        )
        assert m.predict_array([0.0, 0.0, 0.0]) == pytest.approx(0.8)


# ── Split prediction ──

class TestSplitPrediction:
    def test_go_left(self):
        tree = _simple_split(feature=0, threshold=5.0, left_val=1.0, right_val=-1.0)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        assert m.predict_array([3.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_go_right(self):
        tree = _simple_split(feature=0, threshold=5.0, left_val=1.0, right_val=-1.0)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        assert m.predict_array([7.0, 0.0, 0.0]) == pytest.approx(-1.0)

    def test_threshold_equal_goes_left(self):
        tree = _simple_split(feature=0, threshold=5.0, left_val=1.0, right_val=-1.0)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        assert m.predict_array([5.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_split_on_different_feature(self):
        tree = _simple_split(feature=2, threshold=10.0, left_val=0.5, right_val=-0.5)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        assert m.predict_array([0.0, 0.0, 15.0]) == pytest.approx(-0.5)


# ── NaN handling ──

class TestNaNHandling:
    def test_nan_default_left(self):
        tree = _simple_split(feature=0, threshold=5.0, left_val=1.0, right_val=-1.0,
                             default_left=True, nan_as_zero=False)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        assert m.predict_array([float("nan"), 0.0, 0.0]) == pytest.approx(1.0)

    def test_nan_default_right(self):
        tree = _simple_split(feature=0, threshold=5.0, left_val=1.0, right_val=-1.0,
                             default_left=False, nan_as_zero=False)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        assert m.predict_array([float("nan"), 0.0, 0.0]) == pytest.approx(-1.0)

    def test_nan_as_zero_goes_left(self):
        """nan_as_zero: NaN replaced with 0.0, then 0.0 <= 5.0 → left."""
        tree = _simple_split(feature=0, threshold=5.0, left_val=1.0, right_val=-1.0,
                             default_left=False, nan_as_zero=True)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        assert m.predict_array([float("nan"), 0.0, 0.0]) == pytest.approx(1.0)

    def test_nan_as_zero_negative_threshold(self):
        """nan_as_zero: NaN → 0.0, 0.0 > -1.0 → right."""
        tree = _simple_split(feature=0, threshold=-1.0, left_val=1.0, right_val=-1.0,
                             nan_as_zero=True)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        assert m.predict_array([float("nan"), 0.0, 0.0]) == pytest.approx(-1.0)

    def test_all_nan_features(self):
        tree = _simple_split(feature=0, threshold=5.0, left_val=0.5, right_val=-0.5,
                             default_left=True, nan_as_zero=False)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        result = m.predict_array([float("nan"), float("nan"), float("nan")])
        assert math.isfinite(result)


# ── predict_dict ──

class TestPredictDict:
    def test_dict_matches_array(self):
        tree = _simple_split(feature=1, threshold=3.0, left_val=0.7, right_val=-0.3)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        arr_result = m.predict_array([10.0, 2.0, 5.0])
        dict_result = m.predict_dict({"f0": 10.0, "f1": 2.0, "f2": 5.0})
        assert dict_result == pytest.approx(arr_result)

    def test_dict_missing_key_treated_as_nan(self):
        tree = _simple_split(feature=0, threshold=5.0, left_val=1.0, right_val=-1.0,
                             default_left=True, nan_as_zero=False)
        m = RustTreePredictor.from_json(_make_lgbm_json([tree]))
        result = m.predict_dict({"f1": 0.0, "f2": 0.0})
        assert result == pytest.approx(1.0)


# ── Classifier ──

class TestClassifier:
    def test_classifier_sigmoid_center(self):
        """Classifier: sum=0 → sigmoid(0)=0.5 → 0.5-0.5=0.0."""
        m = RustTreePredictor.from_json(
            _make_lgbm_json([_single_leaf(0.0)], is_classifier=True)
        )
        assert m.predict_array([0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-10)

    def test_classifier_positive(self):
        """Large positive sum → sigmoid→1.0 → 1.0-0.5=0.5."""
        m = RustTreePredictor.from_json(
            _make_lgbm_json([_single_leaf(100.0)], is_classifier=True)
        )
        assert m.predict_array([0.0, 0.0, 0.0]) == pytest.approx(0.5, abs=1e-6)

    def test_classifier_negative(self):
        """Large negative sum → sigmoid→0.0 → 0.0-0.5=-0.5."""
        m = RustTreePredictor.from_json(
            _make_lgbm_json([_single_leaf(-100.0)], is_classifier=True)
        )
        assert m.predict_array([0.0, 0.0, 0.0]) == pytest.approx(-0.5, abs=1e-6)


# ── Array validation ──

class TestValidation:
    def test_wrong_feature_count(self):
        m = RustTreePredictor.from_json(_make_lgbm_json([_single_leaf(0.1)]))
        with pytest.raises(ValueError, match="Expected 3"):
            m.predict_array([1.0, 2.0])

    def test_info_string(self):
        m = RustTreePredictor.from_json(_make_lgbm_json([_single_leaf(0.1)]))
        info = m.info()
        assert "lgbm" in info
        assert "trees=1" in info
        assert "features=3" in info
