"""Tests for model registry and artifact store."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from research.model_registry.artifact import ArtifactMeta, ArtifactStore
from research.model_registry.registry import ModelRegistry, ModelVersion


class TestModelRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        return ModelRegistry(db_path=tmp_path / "test_registry.db")

    def test_register_creates_version(self, registry):
        mv = registry.register(
            name="alpha_v1",
            params={"window": 20, "threshold": 0.5},
            features=["sma_20", "rsi_14"],
            metrics={"sharpe": 1.5, "sortino": 2.0},
        )
        assert mv.name == "alpha_v1"
        assert mv.version == 1
        assert mv.params == {"window": 20, "threshold": 0.5}
        assert mv.features == ("sma_20", "rsi_14")
        assert mv.metrics == {"sharpe": 1.5, "sortino": 2.0}
        assert not mv.is_production

    def test_auto_increment_version(self, registry):
        mv1 = registry.register(name="model_a", params={}, features=[], metrics={"sharpe": 1.0})
        mv2 = registry.register(name="model_a", params={}, features=[], metrics={"sharpe": 1.2})
        mv3 = registry.register(name="model_a", params={}, features=[], metrics={"sharpe": 0.9})

        assert mv1.version == 1
        assert mv2.version == 2
        assert mv3.version == 3

    def test_get_by_id(self, registry):
        mv = registry.register(name="test", params={"x": 1}, features=["f1"], metrics={"sharpe": 1.0})
        retrieved = registry.get(mv.model_id)

        assert retrieved is not None
        assert retrieved.model_id == mv.model_id
        assert retrieved.params == {"x": 1}
        assert retrieved.features == ("f1",)

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get("nonexistent-id") is None

    def test_list_versions(self, registry):
        registry.register(name="model_b", params={"v": 1}, features=[], metrics={"sharpe": 1.0})
        registry.register(name="model_b", params={"v": 2}, features=[], metrics={"sharpe": 1.5})
        registry.register(name="other_model", params={}, features=[], metrics={"sharpe": 0.5})

        versions = registry.list_versions("model_b")
        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[1].version == 2

    def test_promote_model(self, registry):
        mv1 = registry.register(name="prod_model", params={}, features=[], metrics={"sharpe": 1.0})
        registry.promote(mv1.model_id)

        prod = registry.get_production("prod_model")
        assert prod is not None
        assert prod.model_id == mv1.model_id
        assert prod.is_production

    def test_promote_demotes_previous(self, registry):
        mv1 = registry.register(name="model_c", params={}, features=[], metrics={"sharpe": 1.0})
        mv2 = registry.register(name="model_c", params={}, features=[], metrics={"sharpe": 1.5})

        registry.promote(mv1.model_id)
        assert registry.get_production("model_c").model_id == mv1.model_id

        registry.promote(mv2.model_id)
        prod = registry.get_production("model_c")
        assert prod.model_id == mv2.model_id

        old = registry.get(mv1.model_id)
        assert not old.is_production

    def test_promote_previous_version_behaves_like_rollback(self, registry):
        mv1 = registry.register(name="model_r", params={"v": 1}, features=[], metrics={"sharpe": 1.0})
        mv2 = registry.register(name="model_r", params={"v": 2}, features=[], metrics={"sharpe": 1.5})

        registry.promote(mv2.model_id)
        assert registry.get_production("model_r").model_id == mv2.model_id

        # Current rollback mechanism is promotion of a previous stable version.
        registry.promote(mv1.model_id)
        prod = registry.get_production("model_r")

        assert prod is not None
        assert prod.model_id == mv1.model_id
        assert prod.version == 1
        assert not registry.get(mv2.model_id).is_production

    def test_promote_nonexistent_raises(self, registry):
        with pytest.raises(ValueError, match="not found"):
            registry.promote("nonexistent")

    def test_get_production_no_model(self, registry):
        assert registry.get_production("no_such_model") is None

    def test_compare_two_versions(self, registry):
        mv1 = registry.register(
            name="cmp",
            params={"window": 10},
            features=["sma_10", "rsi_14"],
            metrics={"sharpe": 1.0, "max_dd": 0.15},
        )
        mv2 = registry.register(
            name="cmp",
            params={"window": 20},
            features=["sma_20", "rsi_14"],
            metrics={"sharpe": 1.5, "max_dd": 0.10},
        )
        comparison = registry.compare(mv1.model_id, mv2.model_id)

        assert comparison["metrics"]["sharpe"]["diff"] == pytest.approx(0.5)
        assert comparison["metrics"]["max_dd"]["diff"] == pytest.approx(-0.05)
        assert "window" in comparison["param_diff"]
        assert "sma_10" in comparison["features_a_only"]
        assert "sma_20" in comparison["features_b_only"]
        assert "rsi_14" in comparison["features_shared"]

    def test_compare_nonexistent_raises(self, registry):
        mv = registry.register(name="x", params={}, features=[], metrics={"sharpe": 1.0})
        with pytest.raises(ValueError, match="not found"):
            registry.compare(mv.model_id, "nonexistent")

    def test_register_with_tags(self, registry):
        mv = registry.register(
            name="tagged",
            params={},
            features=[],
            metrics={"sharpe": 1.0},
            tags=("experimental", "v2"),
        )
        retrieved = registry.get(mv.model_id)
        assert retrieved.tags == ("experimental", "v2")


class TestArtifactStore:
    @pytest.fixture
    def store(self, tmp_path):
        return ArtifactStore(root=tmp_path / "artifacts")

    def test_save_and_load(self, store):
        data = b"model weights binary data"
        meta = store.save("model-1", "weights", data)

        assert meta.model_id == "model-1"
        assert meta.artifact_type == "weights"
        assert meta.size_bytes == len(data)

        loaded = store.load("model-1", "weights")
        assert loaded == data

    def test_save_weights_writes_signature_when_key_present(self, store, monkeypatch):
        monkeypatch.setenv("QUANT_MODEL_SIGN_KEY", "test-key")
        data = b"model weights binary data"

        store.save("model-1", "weights", data)

        sig = store.load("model-1", "weights.sig")
        assert sig is not None
        assert sig.decode("utf-8")

    def test_load_nonexistent_returns_none(self, store):
        assert store.load("no-model", "weights") is None

    def test_list_artifacts(self, store):
        store.save("model-2", "weights", b"w")
        store.save("model-2", "config", b"c")
        store.save("model-2", "report", b"r")

        artifacts = store.list_artifacts("model-2")
        assert len(artifacts) == 3
        types = {a.artifact_type for a in artifacts}
        assert types == {"weights", "config", "report"}

    def test_list_artifacts_empty(self, store):
        assert store.list_artifacts("nonexistent") == []

    def test_delete_artifact(self, store):
        store.save("model-3", "weights", b"data")
        assert store.delete("model-3", "weights")
        assert store.load("model-3", "weights") is None
        assert not store.delete("model-3", "weights")
