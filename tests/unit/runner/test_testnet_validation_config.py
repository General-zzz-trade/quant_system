from __future__ import annotations

from runner.testnet_validation import _monitoring_runtime_kwargs


def test_monitoring_runtime_kwargs_maps_health_server_fields() -> None:
    kwargs = _monitoring_runtime_kwargs(
        {
            "monitoring": {
                "health_check_interval": "15.5",
                "health_port": "18080",
                "health_host": "0.0.0.0",
                "health_auth_token_env": "HEALTH_API_TOKEN",
            }
        }
    )

    assert kwargs == {
        "health_stale_data_sec": 15.5,
        "health_port": 18080,
        "health_host": "0.0.0.0",
        "health_auth_token_env": "HEALTH_API_TOKEN",
    }


def test_monitoring_runtime_kwargs_ignores_missing_or_invalid_sections() -> None:
    assert _monitoring_runtime_kwargs({}) == {}
    assert _monitoring_runtime_kwargs({"monitoring": None}) == {}
