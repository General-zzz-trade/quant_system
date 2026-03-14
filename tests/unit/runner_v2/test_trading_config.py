"""Tests for simplified TradingConfig."""
import dataclasses

import pytest

from runner.trading_config import TradingConfig


class TestTradingConfigDefaults:
    def test_default_symbols(self):
        cfg = TradingConfig(symbols=("BTCUSDT",))
        assert cfg.symbols == ("BTCUSDT",)
        assert cfg.testnet is True
        assert cfg.shadow_mode is True
        assert cfg.venue == "binance"

    def test_field_count_under_55(self):
        fields = dataclasses.fields(TradingConfig)
        assert len(fields) <= 55, f"TradingConfig has {len(fields)} fields, max 55"


class TestTradingConfigFactories:
    def test_paper(self):
        cfg = TradingConfig.paper(symbols=["BTCUSDT"])
        assert cfg.testnet is True
        assert cfg.shadow_mode is True
        assert cfg.enable_reconcile is False

    def test_testnet_full(self):
        cfg = TradingConfig.testnet_full(symbols=["BTCUSDT"])
        assert cfg.testnet is True
        assert cfg.shadow_mode is False
        assert cfg.enable_reconcile is True

    def test_prod(self):
        cfg = TradingConfig.prod(symbols=["BTCUSDT"])
        assert cfg.testnet is False
        assert cfg.use_ws_orders is True
