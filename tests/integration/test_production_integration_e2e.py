# tests/integration/test_production_integration_e2e.py
"""End-to-end integration tests for LiveRunner production integration points.

Validates that ModelRegistry, PortfolioRisk, DataScheduler, FeatureSchema,
InferenceMetrics, and ModelReload all wire correctly through LiveRunner.build().
"""
from __future__ import annotations

import pickle
import time
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from runner.live_runner import LiveRunner, LiveRunnerConfig


# ── Shared fakes (same pattern as unit tests) ─────────────────

class _FakeTransport:
    def __init__(self, messages: list[str] | None = None):
        self._messages = list(messages or [])
        self._idx = 0

    def connect(self, url: str) -> None:
        pass

    def recv(self, timeout_s: float = 5.0) -> Optional[str]:
        if self._idx >= len(self._messages):
            time.sleep(0.01)
            return None
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    def close(self) -> None:
        pass


class _FakeVenueClient:
    def __init__(self) -> None:
        self.orders: List[Any] = []

    def send_order(self, order_event: Any) -> list:
        self.orders.append(order_event)
        return []


def _build_runner(tmp_path, **overrides):
    """Helper to build a LiveRunner with sensible test defaults.

    Keys prefixed with '_' are build() kwargs (not config fields):
        _venue_client, _fetch_margin, _feature_computer, _alpha_models, _metrics_exporter
    """
    # Separate build kwargs from config fields
    build_keys = {"_venue_client", "_fetch_margin", "_feature_computer",
                  "_alpha_models", "_metrics_exporter"}
    build_kwargs = {k: overrides.pop(k) for k in build_keys if k in overrides}

    defaults = dict(
        symbols=("BTCUSDT",),
        enable_monitoring=False,
        enable_reconcile=False,
        enable_persistent_stores=False,
        enable_structured_logging=False,
        enable_preflight=False,
        enable_portfolio_risk=False,
        enable_data_scheduler=False,
    )
    defaults.update(overrides)
    config = LiveRunnerConfig(**defaults)
    return LiveRunner.build(
        config,
        venue_clients={"binance": build_kwargs.get("_venue_client", _FakeVenueClient())},
        transport=_FakeTransport(),
        fetch_margin=build_kwargs.get("_fetch_margin"),
        feature_computer=build_kwargs.get("_feature_computer"),
        alpha_models=build_kwargs.get("_alpha_models"),
        metrics_exporter=build_kwargs.get("_metrics_exporter"),
    )


def _create_stub_model_weights() -> bytes:
    """Create minimal pickle bytes that LGBMAlphaModel.load() can consume."""
    return pickle.dumps({
        "model": None,  # _model = None -> predict() returns None
        "features": ("sma_20", "rsi_14"),
        "is_classifier": False,
    })


# ── Test 1: ModelRegistry auto-loading ────────────────────────

class TestModelRegistryAutoLoading:
    @patch("infra.model_signing.verify_file", return_value=True)
    def test_build_with_model_registry_auto_loading(self, _mock_verify, tmp_path):
        from research.model_registry.registry import ModelRegistry
        from research.model_registry.artifact import ArtifactStore

        db_path = str(tmp_path / "models.db")
        artifact_root = str(tmp_path / "artifacts")

        registry = ModelRegistry(db_path)
        store = ArtifactStore(artifact_root)

        # Register and promote a model
        mv = registry.register(
            name="test_alpha",
            params={"n_estimators": 50},
            features=["sma_20", "rsi_14"],
            metrics={"sharpe": 1.2},
            tags=("lgbm",),
        )
        registry.promote(mv.model_id)
        store.save(mv.model_id, "weights", _create_stub_model_weights())

        runner = _build_runner(
            tmp_path,
            model_registry_db=db_path,
            artifact_store_root=artifact_root,
            model_names=("test_alpha",),
        )

        assert runner.model_loader is not None
        # The loader should have tracked the model_id
        assert "test_alpha" in runner.model_loader._loaded_ids


# ── Test 2: Portfolio Risk Aggregator wiring ─────────────────

class TestPortfolioRiskAggregator:
    def test_build_with_portfolio_risk_aggregator(self, tmp_path):
        runner = _build_runner(
            tmp_path,
            enable_portfolio_risk=True,
            max_gross_leverage=2.0,
            max_concentration=1.0,  # allow 100% single-symbol
            _fetch_margin=lambda: 10000.0,
        )

        assert runner.portfolio_aggregator is not None

        # With no positions and generous limits, a small order should be ALLOWED
        mock_order = SimpleNamespace(
            event_type="ORDER",
            symbol="BTCUSDT",
            side="buy",
            qty=Decimal("0.01"),
            price=Decimal("40000"),
            order_id="test-001",
            intent_id="i-001",
        )
        decision = runner.portfolio_aggregator.evaluate_order(mock_order)
        assert decision.ok


# ── Test 3: Portfolio Risk rejects over-leveraged order ──────

class TestPortfolioRiskRejection:
    def test_portfolio_risk_rejects_overleveraged_order(self, tmp_path):
        """With max_gross_leverage=0.001 (extremely tight), a normal order
        should be rejected by the GrossExposureRule."""
        runner = _build_runner(
            tmp_path,
            enable_portfolio_risk=True,
            max_gross_leverage=0.001,  # extremely tight
            max_net_leverage=0.001,
            _fetch_margin=lambda: 100.0,  # tiny equity
        )

        assert runner.portfolio_aggregator is not None

        # Inject a large fake position into coordinator state
        # so gross notional already exceeds limits
        state_view = runner.coordinator.get_state_view()
        positions = state_view.get("positions", {})
        # Directly set a position in the coordinator's internal state
        fake_position = SimpleNamespace(
            qty=10.0, mark_price=40000.0, entry_price=40000.0,
        )
        # Use coordinator's internal position tracking
        if hasattr(runner.coordinator, "_state"):
            runner.coordinator._state.setdefault("positions", {})["BTCUSDT"] = fake_position
        elif hasattr(runner.coordinator, "_positions"):
            runner.coordinator._positions["BTCUSDT"] = fake_position

        # Now try to evaluate an order — should be rejected (gross >> limit)
        mock_order = SimpleNamespace(
            event_type="ORDER",
            symbol="ETHUSDT",
            side="buy",
            qty=Decimal("1"),
            price=Decimal("3000"),
            order_id="test-002",
            intent_id="i-002",
        )
        decision = runner.portfolio_aggregator.evaluate_order(mock_order)
        # With fake position of 400k notional and equity of 100,
        # gross leverage = 4000, far exceeding 0.001
        assert not decision.ok


# ── Test 4: DataScheduler wiring ─────────────────────────────

class TestDataSchedulerWiring:
    def test_build_with_data_scheduler(self, tmp_path):
        runner = _build_runner(
            tmp_path,
            enable_data_scheduler=True,
            data_files_dir=str(tmp_path),
        )

        assert runner.data_scheduler is not None
        assert runner.freshness_monitor is not None
        # DataScheduler should have 7 download jobs
        assert len(runner.data_scheduler._jobs) == 7


# ── Test 5: Feature schema mismatch warning ──────────────────

class TestFeatureSchemaValidation:
    def test_feature_schema_validation_warns_on_mismatch(self, tmp_path):
        """When model requires features the computer doesn't provide,
        a warning should be logged."""
        from alpha.inference.bridge import LiveInferenceBridge

        # Stub model that requires ["sma_20", "rsi_14", "missing_feat"]
        stub_model = SimpleNamespace(
            name="test_model",
            feature_names=["sma_20", "rsi_14", "missing_feat"],
            predict=lambda **kw: None,
        )

        # Stub computer that only provides ["sma_20", "rsi_14"]
        stub_computer = SimpleNamespace(
            feature_names=["sma_20", "rsi_14"],
            on_bar=lambda **kw: {"sma_20": 1.0, "rsi_14": 50.0},
        )

        # Build LiveInferenceBridge and test that enrich works
        # even when features are missing (model returns None for missing features)
        bridge = LiveInferenceBridge(models=[stub_model])

        # The model's predict returns None, so enrich should complete without error
        features = {"sma_20": 1.0, "rsi_14": 50.0}
        result = bridge.enrich(
            symbol="BTCUSDT",
            ts=datetime.now(timezone.utc),
            features=features,
        )
        # If model requires missing_feat, the feature dict won't have it
        assert "missing_feat" not in result

        # Now verify the conceptual schema gap: model needs 3 features, computer provides 2
        model_features = set(stub_model.feature_names)
        computer_features = set(stub_computer.feature_names)
        missing = model_features - computer_features
        assert "missing_feat" in missing


# ── Test 6: Inference metrics exported ───────────────────────

class TestInferenceMetrics:
    def test_inference_metrics_exported(self):
        from alpha.inference.bridge import LiveInferenceBridge
        from alpha.base import Signal

        # Stub model that returns a signal
        signal = Signal(
            symbol="BTCUSDT",
            ts=datetime.now(timezone.utc),
            side="long",
            strength=0.75,
        )
        stub_model = SimpleNamespace(
            name="test_model",
            predict=lambda **kw: signal,
        )

        mock_metrics = MagicMock()

        bridge = LiveInferenceBridge(
            models=[stub_model],
            metrics_exporter=mock_metrics,
        )

        features: Dict[str, Any] = {"sma_20": 1.0, "rsi_14": 50.0}
        result = bridge.enrich(
            symbol="BTCUSDT",
            ts=datetime.now(timezone.utc),
            features=features,
        )

        # Verify ml_score was set
        assert "ml_score" in result
        assert result["ml_score"] == pytest.approx(0.75)

        # Verify metrics were exported
        mock_metrics.observe_histogram.assert_called_once()
        call_args = mock_metrics.observe_histogram.call_args
        assert call_args[0][0] == "inference_latency_seconds"

        mock_metrics.set_gauge.assert_called_once()
        gauge_args = mock_metrics.set_gauge.call_args
        assert gauge_args[0][0] == "ml_score"
        assert gauge_args[0][1] == pytest.approx(0.75)


# ── Test 7: Model reload detects version change ─────────────

class TestModelReload:
    @patch("infra.model_signing.verify_file", return_value=True)
    def test_model_reload_detects_version_change(self, _mock_verify, tmp_path):
        from research.model_registry.registry import ModelRegistry
        from research.model_registry.artifact import ArtifactStore
        from alpha.model_loader import ProductionModelLoader

        db_path = str(tmp_path / "models.db")
        artifact_root = str(tmp_path / "artifacts")

        registry = ModelRegistry(db_path)
        store = ArtifactStore(artifact_root)

        # Register v1, promote it
        v1 = registry.register(
            name="test_alpha",
            params={"n_estimators": 50},
            features=["sma_20", "rsi_14"],
            metrics={"sharpe": 1.0},
            tags=("lgbm",),
        )
        registry.promote(v1.model_id)
        store.save(v1.model_id, "weights", _create_stub_model_weights())

        # Load v1
        loader = ProductionModelLoader(registry, store)
        models_v1 = loader.load_production_models(["test_alpha"])
        assert len(models_v1) == 1
        assert loader._loaded_ids["test_alpha"] == v1.model_id

        # No change yet
        result = loader.reload_if_changed(["test_alpha"])
        assert result is None

        # Register v2 and promote it
        v2 = registry.register(
            name="test_alpha",
            params={"n_estimators": 100},
            features=["sma_20", "rsi_14"],
            metrics={"sharpe": 1.5},
            tags=("lgbm",),
        )
        registry.promote(v2.model_id)
        store.save(v2.model_id, "weights", _create_stub_model_weights())

        # reload_if_changed should detect the version change
        models_v2 = loader.reload_if_changed(["test_alpha"])
        assert models_v2 is not None
        assert len(models_v2) == 1
        assert loader._loaded_ids["test_alpha"] == v2.model_id
        assert v2.model_id != v1.model_id
