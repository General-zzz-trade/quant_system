"""Tests for strategy/config.py — SYMBOL_CONFIG, notional limits, leverage ladder."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_config():
    """Import strategy.config fresh (module-level constants depend on env)."""
    import strategy.config as cfg
    return cfg


# ---------------------------------------------------------------------------
# SYMBOL_CONFIG structure
# ---------------------------------------------------------------------------

class TestSymbolConfig:
    """Validate that every entry in SYMBOL_CONFIG has the required keys."""

    def test_symbol_config_nonempty(self):
        cfg = _import_config()
        assert len(cfg.SYMBOL_CONFIG) >= 2, "Need at least BTC + ETH"

    @pytest.mark.parametrize("key", ["BTCUSDT", "ETHUSDT"])
    def test_primary_symbols_present(self, key):
        cfg = _import_config()
        assert key in cfg.SYMBOL_CONFIG

    @pytest.mark.parametrize("required_key", ["size", "model_dir", "step"])
    def test_all_entries_have_required_keys(self, required_key):
        cfg = _import_config()
        for sym, entry in cfg.SYMBOL_CONFIG.items():
            assert required_key in entry, f"{sym} missing '{required_key}'"

    def test_primary_symbols_have_max_qty(self):
        cfg = _import_config()
        for sym in ["BTCUSDT", "ETHUSDT"]:
            assert "max_qty" in cfg.SYMBOL_CONFIG[sym], f"{sym} missing 'max_qty'"

    def test_size_positive(self):
        cfg = _import_config()
        for sym, entry in cfg.SYMBOL_CONFIG.items():
            assert entry["size"] > 0, f"{sym} size must be positive"

    def test_step_positive(self):
        cfg = _import_config()
        for sym, entry in cfg.SYMBOL_CONFIG.items():
            assert entry["step"] > 0, f"{sym} step must be positive"

    def test_max_qty_positive_where_present(self):
        cfg = _import_config()
        for sym, entry in cfg.SYMBOL_CONFIG.items():
            if "max_qty" in entry:
                assert entry["max_qty"] > 0, f"{sym} max_qty must be positive"

    def test_model_dir_non_empty_string(self):
        cfg = _import_config()
        for sym, entry in cfg.SYMBOL_CONFIG.items():
            assert isinstance(entry["model_dir"], str) and entry["model_dir"]

    def test_4h_entries_have_interval_240(self):
        cfg = _import_config()
        for sym, entry in cfg.SYMBOL_CONFIG.items():
            if "_4h" in sym:
                assert entry.get("interval") == "240", f"{sym} should have interval=240"

    def test_15m_entries_have_interval_15(self):
        cfg = _import_config()
        for sym, entry in cfg.SYMBOL_CONFIG.items():
            if "_15m" in sym:
                assert entry.get("interval") == "15", f"{sym} should have interval=15"


# ---------------------------------------------------------------------------
# get_max_order_notional
# ---------------------------------------------------------------------------

class TestGetMaxOrderNotional:
    def test_zero_equity_returns_floor(self):
        cfg = _import_config()
        assert cfg.get_max_order_notional(0) == cfg.MAX_ORDER_NOTIONAL_FLOOR

    def test_small_equity_returns_floor(self):
        cfg = _import_config()
        # 10 * 2.5 = 25 < floor(100)
        assert cfg.get_max_order_notional(10) == cfg.MAX_ORDER_NOTIONAL_FLOOR

    def test_normal_equity(self):
        cfg = _import_config()
        # 10000 * 2.5 = 25000
        result = cfg.get_max_order_notional(10_000)
        assert result == 25_000

    def test_large_equity_capped_at_ceiling(self):
        cfg = _import_config()
        # 1_000_000 * 2.5 = 2_500_000 → ceiling 100K
        result = cfg.get_max_order_notional(1_000_000)
        assert result == cfg.MAX_ORDER_NOTIONAL_CEILING

    def test_ceiling_boundary(self):
        cfg = _import_config()
        # 40_000 * 2.5 = 100_000 = ceiling exactly
        result = cfg.get_max_order_notional(40_000)
        assert result == cfg.MAX_ORDER_NOTIONAL_CEILING

    def test_negative_equity_returns_floor(self):
        cfg = _import_config()
        # negative * 2.5 < 0, but max(floor, ...) = floor
        assert cfg.get_max_order_notional(-500) == cfg.MAX_ORDER_NOTIONAL_FLOOR

    def test_equity_100(self):
        cfg = _import_config()
        # 100 * 2.5 = 250
        result = cfg.get_max_order_notional(100)
        assert result == 250.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_order_notional_pct(self):
        cfg = _import_config()
        assert cfg.MAX_ORDER_NOTIONAL_PCT == 2.50

    def test_ceiling_value(self):
        cfg = _import_config()
        assert cfg.MAX_ORDER_NOTIONAL_CEILING == 100_000.0

    def test_floor_value(self):
        cfg = _import_config()
        assert cfg.MAX_ORDER_NOTIONAL_FLOOR == 100.0


# ---------------------------------------------------------------------------
# LEVERAGE_LADDER
# ---------------------------------------------------------------------------

class TestLeverageLadder:
    def test_leverage_ladder_nonempty(self):
        cfg = _import_config()
        assert len(cfg.LEVERAGE_LADDER) >= 1

    def test_leverage_ladder_first_bracket_starts_at_zero(self):
        cfg = _import_config()
        assert cfg.LEVERAGE_LADDER[0][0] == 0


# ---------------------------------------------------------------------------
# _IS_LIVE detection
# ---------------------------------------------------------------------------

class TestIsLive:
    def test_demo_url_not_live(self):
        with patch.dict(os.environ, {"BYBIT_BASE_URL": "https://api-demo.bybit.com"}):
            # Need to reimport to pick up new env
            import importlib
            import strategy.config as cfg
            importlib.reload(cfg)
            assert cfg._IS_LIVE is False

    def test_live_url_is_live(self):
        with patch.dict(os.environ, {"BYBIT_BASE_URL": "https://api.bybit.com"}):
            import importlib
            import strategy.config as cfg
            importlib.reload(cfg)
            assert cfg._IS_LIVE is True

    def test_empty_url_not_live(self):
        with patch.dict(os.environ, {"BYBIT_BASE_URL": ""}, clear=False):
            import importlib
            import strategy.config as cfg
            importlib.reload(cfg)
            assert cfg._IS_LIVE is False

    def test_missing_url_not_live(self):
        env = os.environ.copy()
        env.pop("BYBIT_BASE_URL", None)
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import strategy.config as cfg
            importlib.reload(cfg)
            assert cfg._IS_LIVE is False
