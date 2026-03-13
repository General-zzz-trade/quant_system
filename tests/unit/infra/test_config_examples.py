from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from infra.config.loader import load_config_secure
from infra.config.schema import validate_trading_config
from runner.live_runner import LiveRunner
from runner.testnet_validation import _ensure_testnet, _monitoring_runtime_kwargs


def _rewrite_example_as_json(raw: dict[str, Any], example_name: str, tmp_path: Path) -> Path:
    raw.setdefault("monitoring", {})
    raw["monitoring"]["health_port"] = 0
    raw["monitoring"].pop("metrics_port", None)
    out = tmp_path / f"{example_name}.json"
    out.write_text(json.dumps(raw))
    return out


@pytest.mark.parametrize(
    "example_name, expected_interval",
    [
        ("live.yaml", 10.0),
        ("paper_trading.yaml", 5.0),
    ],
)
def test_live_runner_examples_validate_and_map_runtime_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    example_name: str,
    expected_interval: float,
) -> None:
    example_path = Path("infra/config/examples") / example_name
    raw = load_config_secure(example_path)
    assert validate_trading_config(raw) == []
    assert "metrics_port" not in raw.get("monitoring", {})
    assert raw["monitoring"]["health_port"] == 9090

    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")

    captured: dict[str, object] = {}

    def _fake_build(cls, config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return type("_RunnerStub", (), {"_config": config})()

    monkeypatch.setattr(LiveRunner, "build", classmethod(_fake_build))
    config_path = _rewrite_example_as_json(dict(raw), example_path.stem, tmp_path)
    runner = LiveRunner.from_config(
        config_path,
        venue_clients={"binance": object()},
        transport=object(),
    )

    assert captured["config"] is runner._config
    assert runner._config.health_port == 0
    assert runner._config.health_stale_data_sec == expected_interval


def test_default_testnet_example_loads_securely_and_is_marked_testnet() -> None:
    example_path = Path("infra/config/examples/testnet_multi_gate_v2.yaml")
    raw = load_config_secure(example_path)
    _ensure_testnet(raw)
    monitoring_kwargs = _monitoring_runtime_kwargs(raw)
    assert raw["trading"]["testnet"] is True
    assert tuple(raw["trading"]["symbols"]) == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    assert monitoring_kwargs["health_stale_data_sec"] == 30.0
    assert monitoring_kwargs["health_port"] == 9090
    assert monitoring_kwargs["health_host"] == "127.0.0.1"
