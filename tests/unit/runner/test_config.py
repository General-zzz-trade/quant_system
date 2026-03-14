# tests/unit/runner/test_config.py
"""Tests for LiveRunnerConfig factory classmethods and field grouping."""
from __future__ import annotations

from runner.config import LiveRunnerConfig


class TestProfileLite:
    def test_disables_experimental(self):
        cfg = LiveRunnerConfig.lite(symbols=["BTCUSDT"])
        assert cfg.testnet is True
        assert cfg.enable_monitoring is False
        assert cfg.adaptive_btc_enabled is False

    def test_disables_all_optional_subsystems(self):
        cfg = LiveRunnerConfig.lite(symbols=["BTCUSDT"])
        assert cfg.enable_persistent_stores is False
        assert cfg.enable_reconcile is False
        assert cfg.enable_alpha_health is False
        assert cfg.enable_regime_sizing is False
        assert cfg.use_ws_orders is False
        assert cfg.enable_multi_tf_ensemble is False
        assert cfg.enable_burnin_gate is False
        assert cfg.enable_decision_recording is False
        assert cfg.enable_structured_logging is False

    def test_symbols_converted_to_tuple(self):
        cfg = LiveRunnerConfig.lite(symbols=["BTCUSDT", "ETHUSDT"])
        assert cfg.symbols == ("BTCUSDT", "ETHUSDT")
        assert isinstance(cfg.symbols, tuple)


class TestProfilePaper:
    def test_enables_monitoring_and_shadow(self):
        cfg = LiveRunnerConfig.paper(symbols=["BTCUSDT"])
        assert cfg.testnet is True
        assert cfg.enable_monitoring is True
        assert cfg.shadow_mode is True
        assert cfg.enable_alpha_health is True

    def test_disables_reconcile(self):
        cfg = LiveRunnerConfig.paper(symbols=["BTCUSDT"])
        assert cfg.enable_reconcile is False


class TestProfileTestnetFull:
    def test_full_stack_on_testnet(self):
        cfg = LiveRunnerConfig.testnet_full(symbols=["BTCUSDT"])
        assert cfg.testnet is True
        assert cfg.enable_monitoring is True
        assert cfg.enable_persistent_stores is True
        assert cfg.enable_reconcile is True
        assert cfg.enable_alpha_health is True
        assert cfg.use_ws_orders is True


class TestProfileProd:
    def test_enables_safety(self):
        cfg = LiveRunnerConfig.prod(symbols=["BTCUSDT"])
        assert cfg.testnet is False
        assert cfg.enable_monitoring is True
        assert cfg.enable_persistent_stores is True
        assert cfg.enable_reconcile is True
        assert cfg.enable_alpha_health is True
        assert cfg.enable_structured_logging is True
        assert cfg.use_ws_orders is True


class TestProfileOverrides:
    def test_override_works(self):
        cfg = LiveRunnerConfig.lite(symbols=["BTCUSDT"], enable_monitoring=True)
        assert cfg.enable_monitoring is True  # override works

    def test_override_testnet_flag(self):
        cfg = LiveRunnerConfig.prod(symbols=["ETHUSDT"], testnet=True)
        assert cfg.testnet is True

    def test_override_symbols_still_tuple(self):
        cfg = LiveRunnerConfig.paper(symbols=["BTCUSDT"])
        assert isinstance(cfg.symbols, tuple)


class TestDefaultConstructorUnchanged:
    def test_default_instance(self):
        """Ensure the default constructor still works with original defaults."""
        cfg = LiveRunnerConfig()
        assert cfg.symbols == ("BTCUSDT",)
        assert cfg.testnet is False
        assert cfg.enable_monitoring is True
        assert cfg.enable_persistent_stores is True
        assert cfg.venue == "binance"
