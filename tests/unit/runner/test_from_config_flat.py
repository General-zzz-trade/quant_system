"""Tests for LiveRunner.from_config() with flat YAML format."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from runner.live_runner import LiveRunner, LiveRunnerConfig


@pytest.fixture
def flat_yaml(tmp_path):
    """Create a flat-format production YAML config."""
    cfg = tmp_path / "production.yaml"
    cfg.write_text("""\
symbols:
  - BTCUSDT
  - ETHUSDT
currency: USDT
venue: binance
testnet: true
shadow_mode: false
enable_regime_gate: true
enable_monitoring: true
health_port: 9090
deadzone: 0.5
min_hold_bars:
  BTCUSDT: 12
  ETHUSDT: 12
dd_warning_pct: 10.0
dd_reduce_pct: 15.0
dd_kill_pct: 20.0
""")
    return cfg


@pytest.fixture
def nested_yaml(tmp_path):
    """Create a nested-format legacy YAML config."""
    cfg = tmp_path / "legacy.yaml"
    cfg.write_text("""\
trading:
  symbol: BTCUSDT
  exchange: binance
  testnet: false
strategy:
  name: gate_v2
risk:
  max_leverage: 3.0
logging:
  level: INFO
  structured: true
""")
    return cfg


@pytest.fixture
def mock_venue_clients():
    return {"binance": MagicMock()}


def test_flat_config_parses_symbols(flat_yaml, mock_venue_clients):
    """Flat YAML correctly populates symbols tuple."""
    with patch.object(LiveRunner, "build") as mock_build, \
         patch("infra.config.loader.load_config_secure") as mock_load, \
         patch("infra.config.loader.resolve_credentials"), \
         patch("runner.model_discovery.discover_active_models", return_value={}), \
         patch("runner.model_discovery.build_feature_computer", return_value=MagicMock()):

        import yaml
        mock_load.return_value = yaml.safe_load(flat_yaml.read_text())
        mock_build.return_value = MagicMock()

        LiveRunner.from_config(
            flat_yaml,
            venue_clients=mock_venue_clients,
        )

        call_args = mock_build.call_args
        config = call_args[0][0]
        assert isinstance(config, LiveRunnerConfig)
        assert config.symbols == ("BTCUSDT", "ETHUSDT")
        assert config.venue == "binance"
        assert config.testnet is True
        assert config.health_port == 9090
        assert config.deadzone == 0.5


def test_flat_config_shadow_mode_override(flat_yaml, mock_venue_clients):
    """CLI shadow_mode overrides YAML value."""
    with patch.object(LiveRunner, "build") as mock_build, \
         patch("infra.config.loader.load_config_secure") as mock_load, \
         patch("infra.config.loader.resolve_credentials"), \
         patch("runner.model_discovery.discover_active_models", return_value={}), \
         patch("runner.model_discovery.build_feature_computer", return_value=MagicMock()):

        import yaml
        mock_load.return_value = yaml.safe_load(flat_yaml.read_text())
        mock_build.return_value = MagicMock()

        LiveRunner.from_config(
            flat_yaml,
            venue_clients=mock_venue_clients,
            shadow_mode=True,
        )

        config = mock_build.call_args[0][0]
        assert config.shadow_mode is True


def test_flat_config_auto_loads_models(flat_yaml, mock_venue_clients):
    """from_config() auto-discovers and loads models when not provided."""
    mock_bridge = MagicMock()

    with patch.object(LiveRunner, "build") as mock_build, \
         patch("infra.config.loader.load_config_secure") as mock_load, \
         patch("infra.config.loader.resolve_credentials"), \
         patch("runner.model_discovery.discover_active_models") as mock_discover, \
         patch("runner.model_discovery.load_symbol_models") as mock_load_models, \
         patch("runner.model_discovery.build_inference_bridge") as mock_build_bridge, \
         patch("runner.model_discovery.build_feature_computer") as mock_build_fc:

        import yaml
        mock_load.return_value = yaml.safe_load(flat_yaml.read_text())
        mock_build.return_value = MagicMock()

        mock_discover.return_value = {
            "BTCUSDT": {"dir": Path("models_v8/BTCUSDT_gate_v2"), "config": {"symbol": "BTCUSDT"}},
        }
        mock_load_models.return_value = ([MagicMock()], [1.0])
        mock_build_bridge.return_value = mock_bridge
        mock_build_fc.return_value = MagicMock()

        LiveRunner.from_config(
            flat_yaml,
            venue_clients=mock_venue_clients,
        )

        # build() should be called with feature_computer and inference_bridges
        build_kwargs = mock_build.call_args[1]
        assert build_kwargs["feature_computer"] is not None
        assert build_kwargs["inference_bridges"] is not None
        assert "BTCUSDT" in build_kwargs["inference_bridges"]


def test_flat_config_skips_model_loading_when_provided(flat_yaml, mock_venue_clients):
    """from_config() skips auto-loading when feature_computer + bridges are provided."""
    mock_fc = MagicMock()
    mock_bridges = {"BTCUSDT": MagicMock()}

    with patch.object(LiveRunner, "build") as mock_build, \
         patch("infra.config.loader.load_config_secure") as mock_load, \
         patch("infra.config.loader.resolve_credentials"), \
         patch("runner.model_discovery.discover_active_models") as mock_discover:

        import yaml
        mock_load.return_value = yaml.safe_load(flat_yaml.read_text())
        mock_build.return_value = MagicMock()

        LiveRunner.from_config(
            flat_yaml,
            venue_clients=mock_venue_clients,
            feature_computer=mock_fc,
            inference_bridges=mock_bridges,
        )

        # Should NOT call discover_active_models
        mock_discover.assert_not_called()

        build_kwargs = mock_build.call_args[1]
        assert build_kwargs["feature_computer"] is mock_fc
        assert build_kwargs["inference_bridges"] is mock_bridges


def test_nested_config_still_works(nested_yaml, mock_venue_clients):
    """Legacy nested YAML format still works."""
    with patch.object(LiveRunner, "build") as mock_build, \
         patch("infra.config.loader.load_config_secure") as mock_load, \
         patch("infra.config.loader.resolve_credentials"), \
         patch("runner.model_discovery.discover_active_models", return_value={}), \
         patch("runner.model_discovery.build_feature_computer", return_value=MagicMock()):

        import yaml
        mock_load.return_value = yaml.safe_load(nested_yaml.read_text())
        mock_build.return_value = MagicMock()

        LiveRunner.from_config(
            nested_yaml,
            venue_clients=mock_venue_clients,
        )

        config = mock_build.call_args[0][0]
        assert config.symbols == ("BTCUSDT",)
        assert config.venue == "binance"
