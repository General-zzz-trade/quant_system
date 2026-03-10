"""Tests for RustTickProcessor — full hot-path integration."""
import pytest
import json
from pathlib import Path

_quant_hotpath = pytest.importorskip("_quant_hotpath")
RustTickProcessor = _quant_hotpath.RustTickProcessor
RustTickResult = _quant_hotpath.RustTickResult


def _find_model_json(model_dir: Path) -> list:
    """Find JSON model files in a model directory."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return []
    with config_path.open() as f:
        mcfg = json.load(f)
    paths = []
    for fname in mcfg.get("models", []):
        json_name = fname.replace(".pkl", ".json")
        json_path = model_dir / json_name
        if json_path.exists():
            paths.append(str(json_path))
    return paths


@pytest.fixture
def btc_model_paths():
    """Get BTC model JSON paths if available."""
    model_dir = Path("models_v8/BTCUSDT_gate_v2")
    paths = _find_model_json(model_dir)
    if not paths:
        pytest.skip("No BTC model JSON files found")
    return paths


class TestRustTickProcessorCreate:
    def test_create_basic(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        assert tp is not None
        assert tp.event_index == 0

    def test_create_with_config(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
            zscore_window=720, zscore_warmup=168,
        )
        tp.configure_symbol("BTCUSDT", min_hold=48, deadzone=1.0)
        assert tp is not None


class TestRustTickProcessorTick:
    def test_process_tick_returns_result(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        tp.configure_symbol("BTCUSDT", min_hold=0, deadzone=0.5)

        result = tp.process_tick("BTCUSDT", 50000.0, 100.0, 50100.0, 49900.0, 50000.0, 100)
        assert isinstance(result, RustTickResult)
        assert result.advanced is True
        assert result.event_index == 1

    def test_process_tick_increments_index(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        tp.configure_symbol("BTCUSDT", min_hold=0, deadzone=0.5)

        for i in range(10):
            result = tp.process_tick("BTCUSDT", 50000.0 + i, 100.0, 50100.0, 49900.0, 50000.0, 100 + i)
            assert result.event_index == i + 1

    def test_process_tick_state_export(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        tp.configure_symbol("BTCUSDT", min_hold=0, deadzone=0.5)

        result = tp.process_tick("BTCUSDT", 50000.0, 100.0, 50100.0, 49900.0, 50000.0, 100)

        # Verify state exports are present
        assert result.markets is not None
        assert result.positions is not None
        assert result.account is not None
        assert result.portfolio is not None
        assert result.risk is not None

    def test_process_tick_features(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        tp.configure_symbol("BTCUSDT", min_hold=0, deadzone=0.5)

        result = tp.process_tick("BTCUSDT", 50000.0, 100.0, 50100.0, 49900.0, 50000.0, 100)
        features = result.get_features()

        assert isinstance(features, dict)
        # After first bar, some features should be present
        assert len(features) > 0

    def test_process_tick_ml_scores(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        tp.configure_symbol("BTCUSDT", min_hold=0, deadzone=0.5)

        result = tp.process_tick("BTCUSDT", 50000.0, 100.0, 50100.0, 49900.0, 50000.0, 100)
        assert isinstance(result.ml_score, float)
        assert isinstance(result.raw_score, float)
        assert isinstance(result.ml_short_score, float)

    def test_process_tick_with_ts(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        tp.configure_symbol("BTCUSDT", min_hold=0, deadzone=0.5)

        result = tp.process_tick(
            "BTCUSDT", 50000.0, 100.0, 50100.0, 49900.0, 50000.0, 100,
            ts="2024-01-01T00:00:00",
        )
        assert result.last_ts == "2024-01-01T00:00:00"


class TestRustTickProcessorExternalData:
    def test_push_external_data(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        tp.configure_symbol("BTCUSDT", min_hold=0, deadzone=0.5)
        tp.push_external_data("BTCUSDT", hour=12, dow=3, funding_rate=0.001)

        result = tp.process_tick("BTCUSDT", 50000.0, 100.0, 50100.0, 49900.0, 50000.0, 100)
        assert result.advanced is True


class TestRustTickProcessorStateAccess:
    def test_export_state(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        state = tp.export_state()
        assert "markets" in state
        assert "positions" in state
        assert "account" in state
        assert "event_index" in state

    def test_checkpoint_restore(self, btc_model_paths):
        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        tp.configure_symbol("BTCUSDT", min_hold=48, deadzone=1.0)

        # Process some bars to build signal state
        for i in range(5):
            tp.process_tick("BTCUSDT", 50000.0 + i * 10, 100.0, 50100.0, 49900.0, 50000.0, 100 + i)

        # Checkpoint
        ckpt = tp.checkpoint()
        assert isinstance(ckpt, dict)

        # Create new processor and restore
        tp2 = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        tp2.configure_symbol("BTCUSDT", min_hold=48, deadzone=1.0)
        tp2.restore(ckpt)


class TestRustTickProcessorCoordinatorIntegration:
    def test_coordinator_with_tick_processor(self, btc_model_paths):
        """Test that coordinator accepts tick_processor in config."""
        from engine.coordinator import CoordinatorConfig, EngineCoordinator

        tp = RustTickProcessor.create(
            ["BTCUSDT"], "USDT", 10000.0, btc_model_paths,
        )
        tp.configure_symbol("BTCUSDT", min_hold=0, deadzone=0.5)

        cfg = CoordinatorConfig(
            symbol_default="BTCUSDT",
            symbols=("BTCUSDT",),
            currency="USDT",
            starting_balance=10000.0,
            tick_processor={"BTCUSDT": tp},
        )
        coord = EngineCoordinator(cfg=cfg)
        assert coord is not None
