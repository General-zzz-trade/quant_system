"""Tests for infra.config — config loading and validation."""
from __future__ import annotations

from infra.config.loader import validate_config


class TestValidateConfig:
    def test_valid_config(self) -> None:
        config = {"exchange": {"name": "binance"}, "risk": {"max_dd": 0.1}}
        errors = validate_config(
            config,
            required_keys=["exchange.name", "risk.max_dd"],
        )
        assert errors == []

    def test_missing_key(self) -> None:
        config = {"exchange": {"name": "binance"}}
        errors = validate_config(
            config,
            required_keys=["exchange.name", "risk.max_dd"],
        )
        assert len(errors) == 1
        assert "risk.max_dd" in errors[0]

    def test_type_check(self) -> None:
        config = {"exchange": {"name": 123}}
        errors = validate_config(
            config,
            required_keys=["exchange.name"],
            type_checks={"exchange.name": str},
        )
        assert len(errors) == 1
        assert "type" in errors[0].lower() or "str" in errors[0]

    def test_empty_config_all_keys_missing(self) -> None:
        errors = validate_config({}, required_keys=["a", "b", "c"])
        assert len(errors) == 3
