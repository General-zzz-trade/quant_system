"""Tests for ProductionModelLoader."""
from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from alpha.model_loader import ProductionModelLoader


# ── Stub registry and artifact store ─────────────────────────

@dataclass
class FakeModelVersion:
    model_id: str
    name: str
    version: int
    params: dict = None
    features: tuple = ()
    metrics: dict = None
    created_at: datetime = None
    is_production: bool = True
    tags: tuple = ()

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.metrics is None:
            self.metrics = {}
        if self.created_at is None:
            self.created_at = datetime(2024, 1, 1)


class FakeRegistry:
    def __init__(self, versions: dict[str, FakeModelVersion] | None = None):
        self._versions = versions or {}

    def get_production(self, name: str) -> Optional[FakeModelVersion]:
        return self._versions.get(name)


class FakeArtifactStore:
    def __init__(self, artifacts: dict[tuple[str, str], bytes] | None = None):
        self._artifacts = artifacts or {}

    def load(self, model_id: str, artifact_type: str) -> Optional[bytes]:
        return self._artifacts.get((model_id, artifact_type))


# ── Tests ────────────────────────────────────────────────────

class TestProductionModelLoader:
    def test_no_production_model_returns_empty(self):
        registry = FakeRegistry()
        store = FakeArtifactStore()
        loader = ProductionModelLoader(registry, store)
        models = loader.load_production_models(["alpha_btc"])
        assert models == []

    def test_no_weights_artifact_returns_empty(self):
        version = FakeModelVersion(model_id="m1", name="alpha_btc", version=1, tags=("lgbm",))
        registry = FakeRegistry({"alpha_btc": version})
        store = FakeArtifactStore()  # no artifacts
        loader = ProductionModelLoader(registry, store)
        models = loader.load_production_models(["alpha_btc"])
        assert models == []

    def test_detect_type_from_tags_lgbm(self):
        loader = ProductionModelLoader(FakeRegistry(), FakeArtifactStore())
        version = FakeModelVersion(model_id="m1", name="x", version=1, tags=("lgbm",))
        assert loader._detect_type(version) == "lgbm"

    def test_detect_type_from_tags_xgb(self):
        loader = ProductionModelLoader(FakeRegistry(), FakeArtifactStore())
        version = FakeModelVersion(model_id="m1", name="x", version=1, tags=("xgb",))
        assert loader._detect_type(version) == "xgb"

    def test_detect_type_from_name(self):
        loader = ProductionModelLoader(FakeRegistry(), FakeArtifactStore())
        version = FakeModelVersion(model_id="m1", name="lgbm_alpha_btc", version=1)
        assert loader._detect_type(version) == "lgbm"

    def test_detect_type_default_lgbm(self):
        loader = ProductionModelLoader(FakeRegistry(), FakeArtifactStore())
        version = FakeModelVersion(model_id="m1", name="generic", version=1)
        assert loader._detect_type(version) == "lgbm"

    def test_load_lgbm_model(self):
        """End-to-end: register → save weights → load via ProductionModelLoader."""
        version = FakeModelVersion(
            model_id="m-lgbm-1", name="alpha_btc", version=1, tags=("lgbm",),
            features=("sma_20", "rsi_14"),
        )

        # Create a fake pickle that LGBMAlphaModel.load() expects
        model_data = {"model": None, "features": ("sma_20", "rsi_14"), "is_classifier": False}
        weights = pickle.dumps(model_data)

        registry = FakeRegistry({"alpha_btc": version})
        store = FakeArtifactStore({
            ("m-lgbm-1", "weights"): weights,
            ("m-lgbm-1", "weights.sig"): b"dummy",
        })
        loader = ProductionModelLoader(registry, store)

        # Patch verify_file to skip signature check in tests
        with patch("infra.model_signing.verify_file", return_value=None):
            models = loader.load_production_models(["alpha_btc"])

        assert len(models) == 1
        assert models[0].name == "alpha_btc"
        assert tuple(models[0].feature_names) == ("sma_20", "rsi_14")

    def test_load_xgb_model(self):
        version = FakeModelVersion(
            model_id="m-xgb-1", name="xgb_alpha_eth", version=2, tags=("xgb",),
            features=("ema_50",),
        )
        model_data = {"model": None, "features": ("ema_50",)}
        weights = pickle.dumps(model_data)

        registry = FakeRegistry({"xgb_alpha_eth": version})
        store = FakeArtifactStore({
            ("m-xgb-1", "weights"): weights,
            ("m-xgb-1", "weights.sig"): b"dummy",
        })
        loader = ProductionModelLoader(registry, store)

        with patch("infra.model_signing.verify_file", return_value=None):
            models = loader.load_production_models(["xgb_alpha_eth"])
        assert len(models) == 1
        assert models[0].name == "xgb_alpha_eth"

    def test_reload_if_changed_returns_none_when_same(self):
        version = FakeModelVersion(model_id="m1", name="alpha", version=1, tags=("lgbm",))
        model_data = {"model": None, "features": (), "is_classifier": False}
        weights = pickle.dumps(model_data)

        registry = FakeRegistry({"alpha": version})
        store = FakeArtifactStore({
            ("m1", "weights"): weights,
            ("m1", "weights.sig"): b"dummy",
        })
        loader = ProductionModelLoader(registry, store)

        with patch("infra.model_signing.verify_file", return_value=None):
            loader.load_production_models(["alpha"])
            result = loader.reload_if_changed(["alpha"])

        assert result is None  # no change

    def test_reload_if_changed_detects_change(self):
        v1 = FakeModelVersion(model_id="m1", name="alpha", version=1, tags=("lgbm",))
        model_data = {"model": None, "features": (), "is_classifier": False}
        weights = pickle.dumps(model_data)

        registry = FakeRegistry({"alpha": v1})
        store = FakeArtifactStore({
            ("m1", "weights"): weights,
            ("m1", "weights.sig"): b"dummy",
            ("m2", "weights"): weights,
            ("m2", "weights.sig"): b"dummy",
        })
        loader = ProductionModelLoader(registry, store)

        with patch("infra.model_signing.verify_file", return_value=None):
            loader.load_production_models(["alpha"])

        # Simulate promotion to new version
        v2 = FakeModelVersion(model_id="m2", name="alpha", version=2, tags=("lgbm",))
        registry._versions["alpha"] = v2

        with patch("infra.model_signing.verify_file", return_value=None):
            result = loader.reload_if_changed(["alpha"])

        assert result is not None
        assert len(result) == 1

    def test_reload_if_changed_detects_rollback_to_previous_model(self):
        v1 = FakeModelVersion(model_id="m1", name="alpha", version=1, tags=("lgbm",))
        v2 = FakeModelVersion(model_id="m2", name="alpha", version=2, tags=("lgbm",))
        model_data = {"model": None, "features": (), "is_classifier": False}
        weights = pickle.dumps(model_data)

        registry = FakeRegistry({"alpha": v2})
        store = FakeArtifactStore({
            ("m1", "weights"): weights,
            ("m1", "weights.sig"): b"dummy",
            ("m2", "weights"): weights,
            ("m2", "weights.sig"): b"dummy",
        })
        loader = ProductionModelLoader(registry, store)

        with patch("infra.model_signing.verify_file", return_value=None):
            loader.load_production_models(["alpha"])

        # Simulate rollback by promoting the previous model_id again.
        registry._versions["alpha"] = v1

        with patch("infra.model_signing.verify_file", return_value=None):
            result = loader.reload_if_changed(["alpha"])

        assert result is not None
        assert len(result) == 1
        assert loader._loaded_ids["alpha"] == "m1"

    def test_load_multiple_models(self):
        v1 = FakeModelVersion(model_id="m1", name="btc", version=1, tags=("lgbm",))
        v2 = FakeModelVersion(model_id="m2", name="eth", version=1, tags=("xgb",))
        model_data = {"model": None, "features": (), "is_classifier": False}
        weights = pickle.dumps(model_data)

        registry = FakeRegistry({"btc": v1, "eth": v2})
        store = FakeArtifactStore({
            ("m1", "weights"): weights,
            ("m1", "weights.sig"): b"dummy",
            ("m2", "weights"): weights,
            ("m2", "weights.sig"): b"dummy",
        })
        loader = ProductionModelLoader(registry, store)

        with patch("infra.model_signing.verify_file", return_value=None):
            models = loader.load_production_models(["btc", "eth"])

        assert len(models) == 2

    def test_corrupt_weights_returns_empty(self):
        version = FakeModelVersion(model_id="m1", name="alpha", version=1, tags=("lgbm",))
        registry = FakeRegistry({"alpha": version})
        store = FakeArtifactStore({("m1", "weights"): b"corrupt data"})
        loader = ProductionModelLoader(registry, store)

        models = loader.load_production_models(["alpha"])
        assert models == []

    def test_feature_schema_mismatch_returns_empty(self):
        version = FakeModelVersion(
            model_id="m1",
            name="alpha",
            version=1,
            tags=("lgbm",),
            features=("expected_a", "expected_b"),
        )
        model_data = {
            "model": None,
            "features": ("actual_a", "actual_b"),
            "is_classifier": False,
        }
        weights = pickle.dumps(model_data)

        registry = FakeRegistry({"alpha": version})
        store = FakeArtifactStore({
            ("m1", "weights"): weights,
            ("m1", "weights.sig"): b"dummy",
        })
        loader = ProductionModelLoader(registry, store)

        with patch("infra.model_signing.verify_file", return_value=None):
            models = loader.load_production_models(["alpha"])

        assert models == []
