from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from runner.live_runner import LiveRunner


class _FakeTransport:
    def connect(self, url: str) -> None:
        pass

    def recv(self, timeout_s: float = 5.0) -> Optional[str]:
        time.sleep(0.01)
        return None

    def close(self) -> None:
        pass


class _FakeVenueClient:
    def send_order(self, order_event: Any) -> list:
        return []


def _write_config(tmp_path: Path, monitoring: dict[str, Any]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "trading": {
                    "symbol": "BTCUSDT",
                    "exchange": "binance",
                    "mode": "paper",
                },
                "strategy": {
                    "name": "test_strategy",
                },
                "credentials": {
                    "api_key_env": "TEST_API_KEY",
                    "api_secret_env": "TEST_API_SECRET",
                },
                "monitoring": monitoring,
            }
        )
    )
    return path


def test_from_config_reads_health_endpoint_settings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    monkeypatch.setenv("TEST_API_SECRET", "test-secret")

    config_path = _write_config(
        tmp_path,
        {
            "health_check_interval": 12.5,
            "health_port": 0,
            "health_host": "127.0.0.1",
        },
    )

    runner = LiveRunner.from_config(
        config_path,
        venue_clients={"binance": _FakeVenueClient()},
        transport=_FakeTransport(),
    )
    try:
        assert runner._config.health_stale_data_sec == 12.5
        assert runner._config.health_port == 0
        assert runner._config.health_host == "127.0.0.1"
        assert runner.health_server is not None
    finally:
        runner.stop()
