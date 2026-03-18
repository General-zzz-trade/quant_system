"""Test run_bybit_alpha.py LiveRunner/AlphaRunner switching."""


class TestRunnerSwitch:
    def test_legacy_flag_selects_alpha_runner(self):
        """--legacy should select AlphaRunner path."""
        from scripts.ops.run_bybit_alpha import _select_runner_class
        cls = _select_runner_class(legacy=True)
        assert cls.__name__ == "AlphaRunner"

    def test_default_selects_live_runner(self):
        """Default (no --legacy) should select LiveRunner path."""
        from scripts.ops.run_bybit_alpha import _select_runner_class
        cls = _select_runner_class(legacy=False)
        # Should return LiveRunner or a wrapper
        assert "LiveRunner" in cls.__name__ or "live" in cls.__name__.lower()

    def test_symbol_config_to_live_config(self):
        """SYMBOL_CONFIG should map correctly to LiveRunnerConfig fields."""
        from scripts.ops.run_bybit_alpha import _build_live_config
        cfg = _build_live_config(
            symbols=["ETHUSDT", "BTCUSDT"],
            ws=True,
            dry_run=False,
        )
        assert "ETHUSDT" in cfg.symbols
        assert "BTCUSDT" in cfg.symbols

    def test_runner_target_uses_real_symbol_and_interval_mapping(self):
        from scripts.ops.run_bybit_alpha import _resolve_runner_target

        assert _resolve_runner_target(
            "ETHUSDT_15m",
            {"ETHUSDT_15m": ("ETHUSDT", "15")},
        ) == ("ETHUSDT", "15")

    def test_runner_target_falls_back_to_runner_key_and_default_interval(self):
        from scripts.ops.run_bybit_alpha import _resolve_runner_target, INTERVAL

        assert _resolve_runner_target("BTCUSDT", None) == ("BTCUSDT", INTERVAL)
