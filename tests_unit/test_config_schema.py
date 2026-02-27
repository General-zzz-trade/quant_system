"""Tests for infra.config.schema."""
from __future__ import annotations

from infra.config.schema import validate_trading_config, get_schema_docs


class TestValidateTradingConfig:

    def test_valid_config(self) -> None:
        config = {
            "trading": {"symbol": "BTCUSDT", "exchange": "binance"},
            "strategy": {"name": "ma_cross"},
        }
        errors = validate_trading_config(config)
        assert errors == []

    def test_missing_required_keys(self) -> None:
        errors = validate_trading_config({})
        assert len(errors) > 0
        assert any("trading.symbol" in e for e in errors)

    def test_schema_docs_generated(self) -> None:
        docs = get_schema_docs()
        assert "trading.symbol" in docs
        assert "required" in docs
