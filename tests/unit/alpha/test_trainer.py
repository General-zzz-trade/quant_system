"""Tests for alpha/training/trainer.py — ModelTrainer walk-forward."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

import numpy as np

from alpha.training.trainer import FoldResult, ModelTrainer


# ── Helpers ─────────────────────────────────────────────────


@dataclass
class MockTrainableModel:
    name: str = "mock_model"
    _fit_calls: int = 0
    _saved_paths: List[str] = field(default_factory=list)
    _last_train_size: int = 0
    _return_metrics: Dict[str, float] = field(default_factory=lambda: {"val_mse": 0.01, "direction_acc": 0.6})

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Dict[str, float]:
        self._fit_calls += 1
        self._last_train_size = len(X)
        return dict(self._return_metrics)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("mock")
        self._saved_paths.append(str(path))


# ── Tests ───────────────────────────────────────────────────


class TestModelTrainer:
    def _gen_data(self, n: int = 600):
        X = np.random.randn(n, 3)
        y = np.random.randn(n)
        return X, y

    def test_walk_forward_train_basic(self):
        model = MockTrainableModel()
        trainer = ModelTrainer(model=model)
        X, y = self._gen_data(600)
        results = trainer.walk_forward_train(X, y, n_splits=5)
        assert len(results) == 5
        assert all(isinstance(r, FoldResult) for r in results)
        assert model._fit_calls == 5

    def test_fold_result_fields(self):
        model = MockTrainableModel()
        trainer = ModelTrainer(model=model)
        X, y = self._gen_data(300)
        results = trainer.walk_forward_train(X, y, n_splits=3)
        r = results[0]
        assert r.fold == 0
        assert r.train_size > 0
        assert r.val_size > 0
        assert "val_mse" in r.metrics

    def test_expanding_window(self):
        model = MockTrainableModel()
        trainer = ModelTrainer(model=model)
        X, y = self._gen_data(600)
        results = trainer.walk_forward_train(X, y, n_splits=3, expanding=True)
        # Expanding: each fold trains on more data
        train_sizes = [r.train_size for r in results]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_fixed_window(self):
        model = MockTrainableModel()
        trainer = ModelTrainer(model=model)
        X, y = self._gen_data(800)
        results = trainer.walk_forward_train(X, y, n_splits=4, expanding=False)
        # Fixed: each fold trains on the same chunk_size
        train_sizes = [r.train_size for r in results]
        assert all(s == train_sizes[0] for s in train_sizes)

    def test_correct_number_of_folds(self):
        model = MockTrainableModel()
        trainer = ModelTrainer(model=model)
        X, y = self._gen_data(1000)
        for n_splits in [2, 3, 5, 7]:
            model._fit_calls = 0
            results = trainer.walk_forward_train(X, y, n_splits=n_splits)
            assert len(results) == n_splits

    def test_output_dir_saves_model(self, tmp_path: Path):
        model = MockTrainableModel()
        trainer = ModelTrainer(model=model, out_dir=tmp_path)
        X, y = self._gen_data(300)
        trainer.walk_forward_train(X, y, n_splits=2)
        assert len(model._saved_paths) == 1
        assert "mock_model_final.pkl" in model._saved_paths[0]

    def test_no_output_dir_no_save(self):
        model = MockTrainableModel()
        trainer = ModelTrainer(model=model)
        X, y = self._gen_data(300)
        trainer.walk_forward_train(X, y, n_splits=2)
        assert model._saved_paths == []

    def test_auto_reduce_splits_for_small_data(self):
        model = MockTrainableModel()
        trainer = ModelTrainer(model=model)
        X, y = self._gen_data(200)
        # Request 10 splits with min_train_size=100 — chunk=200/11~18 < 100
        # Should auto-reduce n_splits
        results = trainer.walk_forward_train(X, y, n_splits=10, min_train_size=100)
        assert len(results) >= 1
        assert len(results) < 10

    def test_fit_kwargs_passed(self):
        calls: list[dict] = []

        @dataclass
        class KwargsModel:
            name: str = "kw_model"

            def fit(self, X: Any, y: Any, **kwargs: Any) -> Dict[str, float]:
                calls.append(kwargs)
                return {"mse": 0.01}

            def save(self, path: str | Path) -> None:
                pass

        model = KwargsModel()
        trainer = ModelTrainer(model=model)
        X, y = self._gen_data(300)
        trainer.walk_forward_train(X, y, n_splits=2, fit_kwargs={"learning_rate": 0.05})
        assert all(c.get("learning_rate") == 0.05 for c in calls)

    def test_metrics_stored_in_results(self):
        model = MockTrainableModel(
            _return_metrics={"val_mse": 0.02, "direction_acc": 0.65}
        )
        trainer = ModelTrainer(model=model)
        X, y = self._gen_data(300)
        results = trainer.walk_forward_train(X, y, n_splits=2)
        for r in results:
            assert r.metrics["val_mse"] == 0.02
            assert r.metrics["direction_acc"] == 0.65
