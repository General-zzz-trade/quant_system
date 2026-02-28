"""Tests for LiveRunner.from_config() config-driven construction."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


VALID_CONFIG = {
    "trading": {"symbol": "ETHUSDT", "exchange": "binance", "mode": "live"},
    "strategy": {"name": "momentum", "ma_window": 20, "order_qty": "0.1"},
    "risk": {
        "max_position_notional": 50000.0,
        "max_leverage": 3.0,
        "max_drawdown_pct": 15.0,
        "max_orders_per_minute": 30,
    },
    "execution": {"fee_bps": 4.0, "slippage_bps": 2.0},
    "credentials": {
        "api_key_env": "BINANCE_API_KEY",
        "api_secret_env": "BINANCE_API_SECRET",
    },
    "logging": {"level": "DEBUG", "structured": True, "file": "logs/test.log"},
    "monitoring": {"health_check_interval": 5.0, "metrics_port": 9090},
}


def _write_yaml(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "test_config.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


@patch("runner.live_runner.LiveRunner.build")
def test_from_config_valid(mock_build, tmp_path, monkeypatch):
    """from_config() with a valid YAML config maps fields and delegates to build()."""
    monkeypatch.setenv("BINANCE_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
    mock_build.return_value = MagicMock()
    config_path = _write_yaml(tmp_path, VALID_CONFIG)

    from runner.live_runner import LiveRunner

    venue_clients = {"binance": MagicMock()}
    result = LiveRunner.from_config(config_path, venue_clients=venue_clients)

    mock_build.assert_called_once()
    call_args = mock_build.call_args
    runner_cfg = call_args[0][0]

    assert runner_cfg.symbols == ("ETHUSDT",)
    assert runner_cfg.venue == "binance"
    assert runner_cfg.log_level == "DEBUG"
    assert runner_cfg.log_file == "logs/test.log"
    assert runner_cfg.enable_structured_logging is True
    assert runner_cfg.health_stale_data_sec == 5.0
    assert runner_cfg.margin_warning_ratio == 3.0
    assert runner_cfg.margin_critical_ratio == pytest.approx(0.15)
    assert result is mock_build.return_value


@patch("runner.live_runner.LiveRunner.build")
def test_from_config_missing_required_fields(mock_build, tmp_path):
    """from_config() raises ValueError when required schema keys are missing."""
    bad_config = {"trading": {"symbol": "BTCUSDT"}}  # missing exchange, strategy.name
    config_path = _write_yaml(tmp_path, bad_config)

    from runner.live_runner import LiveRunner

    with pytest.raises(ValueError, match="Config validation failed"):
        LiveRunner.from_config(config_path, venue_clients={"binance": MagicMock()})

    mock_build.assert_not_called()


@patch("runner.live_runner.LiveRunner.build")
def test_from_config_validates_schema_types(mock_build, tmp_path):
    """from_config() rejects configs where field types don't match schema."""
    bad_types = dict(VALID_CONFIG)
    bad_types = {**VALID_CONFIG, "risk": {**VALID_CONFIG["risk"], "max_leverage": "not_a_float"}}
    config_path = _write_yaml(tmp_path, bad_types)

    from runner.live_runner import LiveRunner

    with pytest.raises(ValueError, match="Config validation failed"):
        LiveRunner.from_config(config_path, venue_clients={"binance": MagicMock()})

    mock_build.assert_not_called()


@patch("runner.live_runner.LiveRunner.build")
def test_from_config_file_not_found(mock_build, tmp_path):
    """from_config() raises FileNotFoundError for nonexistent path."""
    from runner.live_runner import LiveRunner

    with pytest.raises(FileNotFoundError):
        LiveRunner.from_config(
            tmp_path / "does_not_exist.yaml",
            venue_clients={"binance": MagicMock()},
        )


@patch("runner.live_runner.LiveRunner.build")
def test_from_config_optional_fields_use_defaults(mock_build, tmp_path):
    """from_config() uses LiveRunnerConfig defaults when optional sections are absent."""
    mock_build.return_value = MagicMock()
    minimal = {
        "trading": {"symbol": "BTCUSDT", "exchange": "binance"},
        "strategy": {"name": "basic"},
    }
    config_path = _write_yaml(tmp_path, minimal)

    from runner.live_runner import LiveRunner

    LiveRunner.from_config(config_path, venue_clients={"binance": MagicMock()})

    runner_cfg = mock_build.call_args[0][0]
    assert runner_cfg.symbols == ("BTCUSDT",)
    assert runner_cfg.venue == "binance"
    assert runner_cfg.log_level == "INFO"
    assert runner_cfg.enable_structured_logging is True
    assert runner_cfg.health_stale_data_sec == 120.0
