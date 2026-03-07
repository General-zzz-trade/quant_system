"""Regression: verify Strategy F configs match expected feature lists and constraints."""
from __future__ import annotations

from alpha.strategy_config import SYMBOL_CONFIG, get_config


def test_btc_strategy_f_features():
    """BTC Strategy F must have 10 fixed features and model_dir set."""
    cfg = get_config("BTCUSDT")
    assert len(cfg.fixed_features) == 10
    assert "basis" in cfg.fixed_features
    assert "atr_norm_14" in cfg.fixed_features
    assert cfg.model_dir == "models_v8/BTCUSDT_gate_v2"
    assert cfg.monthly_gate_window == 480
    assert cfg.deadzone == 0.5
    assert cfg.min_hold == 24
    assert cfg.long_only is True


def test_eth_strategy_f_features():
    """ETH Strategy F must have 13 fixed features with BTC-lead."""
    cfg = get_config("ETHUSDT")
    assert len(cfg.fixed_features) == 13
    assert "btc_ret_24" in cfg.fixed_features
    assert "btc_rsi_14" in cfg.fixed_features
    assert "btc_mean_reversion_20" in cfg.fixed_features
    assert cfg.model_dir == "models_v8/ETHUSDT_gate_v2"


def test_sol_strategy_btc_lead():
    """SOL Strategy must include BTC-lead features."""
    cfg = get_config("SOLUSDT")
    assert "btc_ret_24" in cfg.fixed_features
    assert "btc_rsi_14" in cfg.fixed_features
    assert "btc_mean_reversion_20" in cfg.fixed_features
    assert cfg.model_dir == "models_v8/SOLUSDT_gate_v3"
    assert cfg.deadzone == 1.0
    assert cfg.min_hold == 48
    assert cfg.n_flexible == 5


def test_all_configs_have_required_fields():
    """All symbol configs must have non-empty feature lists and valid constraints."""
    for sym, cfg in SYMBOL_CONFIG.items():
        assert len(cfg.fixed_features) >= 10, f"{sym}: too few fixed features"
        assert len(cfg.candidate_pool) >= 6, f"{sym}: too few candidates"
        assert cfg.n_flexible > 0, f"{sym}: n_flexible must be > 0"
        assert cfg.deadzone > 0, f"{sym}: deadzone must be > 0"
        assert cfg.min_hold > 0, f"{sym}: min_hold must be > 0"
        assert cfg.model_dir, f"{sym}: model_dir must be set"
