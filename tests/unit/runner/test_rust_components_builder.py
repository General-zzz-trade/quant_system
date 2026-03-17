# tests/unit/runner/test_rust_components_builder.py
"""Test Rust components builder."""
import pytest


class TestRustComponentsBuilder:
    def test_disabled_returns_all_none(self):
        from types import SimpleNamespace
        from runner.builders.rust_components_builder import build_rust_components

        config = SimpleNamespace(enable_rust_components=False)
        result = build_rust_components(config, symbols=("ETHUSDT",))
        assert result.feature_engine is None
        assert result.inference_bridges is None
        assert result.risk_evaluator is None
        assert result.kill_switch_rust is None
        assert result.order_state_machine is None
        assert result.circuit_breaker is None
        assert result.state_store is None

    def test_disabled_with_multiple_symbols(self):
        from types import SimpleNamespace
        from runner.builders.rust_components_builder import build_rust_components

        config = SimpleNamespace(enable_rust_components=False)
        result = build_rust_components(config, symbols=("ETHUSDT", "BTCUSDT", "SUIUSDT"))
        assert result.feature_engine is None
        assert result.inference_bridges is None

    def test_enabled_returns_components(self):
        pytest.importorskip("_quant_hotpath")
        from types import SimpleNamespace
        from runner.builders.rust_components_builder import build_rust_components

        config = SimpleNamespace(
            enable_rust_components=True,
            zscore_window=720,
            zscore_warmup=180,
        )
        result = build_rust_components(config, symbols=("ETHUSDT", "BTCUSDT"))
        assert result.feature_engine is not None
        assert result.risk_evaluator is not None
        assert result.kill_switch_rust is not None
        assert result.order_state_machine is not None
        assert result.circuit_breaker is not None
        assert result.state_store is not None
        assert result.inference_bridges is not None
        assert "ETHUSDT" in result.inference_bridges
        assert "BTCUSDT" in result.inference_bridges

    def test_components_are_callable(self):
        pytest.importorskip("_quant_hotpath")
        from types import SimpleNamespace
        from runner.builders.rust_components_builder import build_rust_components

        config = SimpleNamespace(
            enable_rust_components=True,
            zscore_window=720,
            zscore_warmup=180,
        )
        result = build_rust_components(config, symbols=("ETHUSDT",))
        # Verify components have expected methods
        assert hasattr(result.feature_engine, "push_bar")
        assert hasattr(result.circuit_breaker, "allow_request")
        assert hasattr(result.state_store, "process_event")

    def test_inference_bridge_count_matches_symbols(self):
        pytest.importorskip("_quant_hotpath")
        from types import SimpleNamespace
        from runner.builders.rust_components_builder import build_rust_components

        symbols = ("ETHUSDT", "BTCUSDT", "SUIUSDT")
        config = SimpleNamespace(
            enable_rust_components=True,
            zscore_window=720,
            zscore_warmup=180,
        )
        result = build_rust_components(config, symbols=symbols)
        assert len(result.inference_bridges) == len(symbols)
        for sym in symbols:
            assert sym in result.inference_bridges

    def test_default_zscore_params_used_when_absent(self):
        pytest.importorskip("_quant_hotpath")
        from types import SimpleNamespace
        from runner.builders.rust_components_builder import build_rust_components

        # config has no zscore_window / zscore_warmup — should fall back to defaults
        config = SimpleNamespace(enable_rust_components=True)
        result = build_rust_components(config, symbols=("ETHUSDT",))
        assert result.inference_bridges is not None
        assert "ETHUSDT" in result.inference_bridges

    def test_returns_named_tuple(self):
        from types import SimpleNamespace
        from runner.builders.rust_components_builder import (
            RustComponents,
            build_rust_components,
        )

        config = SimpleNamespace(enable_rust_components=False)
        result = build_rust_components(config, symbols=())
        assert isinstance(result, RustComponents)
