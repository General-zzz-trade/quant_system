"""Test run_bybit_alpha.py LiveRunner/AlphaRunner switching."""

from unittest.mock import MagicMock


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

    def test_combo_entry_price_prefers_fill_price(self):
        from scripts.ops.run_bybit_alpha import _combo_entry_price

        assert _combo_entry_price({"fill_price": 1995.5}, 2000.0) == 1995.5

    def test_combo_entry_price_falls_back_on_bad_fill_price(self):
        from scripts.ops.run_bybit_alpha import _combo_entry_price

        assert _combo_entry_price({"fill_price": "bad"}, 2000.0) == 2000.0

    def test_enforce_portfolio_kill_flattens_runners_and_combo(self):
        from scripts.ops.run_bybit_alpha import _enforce_portfolio_kill

        runner = MagicMock()
        combiner = MagicMock()
        combiner.force_flat.return_value = {"action": "forced_flat", "to": 0}
        combiner._current_position = 0
        pm = MagicMock()
        pm.is_killed = True

        forced = _enforce_portfolio_kill(
            {"ETHUSDT": runner},
            {"ETHUSDT": combiner},
            pm,
            {"ETHUSDT": 2000.0},
        )

        runner.force_flat_local_state.assert_called_once()
        combiner.force_flat.assert_called_once_with(2000.0, reason="portfolio_killed")
        pm.record_position.assert_called_once_with("ETHUSDT", 0, 0, "COMBO_KILLED")
        assert forced == {"ETHUSDT": {"action": "forced_flat", "to": 0}}

    def test_enforce_portfolio_kill_noops_when_not_killed(self):
        from scripts.ops.run_bybit_alpha import _enforce_portfolio_kill

        runner = MagicMock()
        combiner = MagicMock()
        pm = MagicMock()
        pm.is_killed = False

        forced = _enforce_portfolio_kill(
            {"ETHUSDT": runner},
            {"ETHUSDT": combiner},
            pm,
            {"ETHUSDT": 2000.0},
        )

        assert forced == {}
        runner.force_flat_local_state.assert_not_called()
        combiner.force_flat.assert_not_called()

    def test_enforce_portfolio_kill_keeps_pm_truth_when_combo_close_fails(self):
        from scripts.ops.run_bybit_alpha import _enforce_portfolio_kill

        runner = MagicMock()
        combiner = MagicMock()
        combiner.force_flat.return_value = {"action": "forced_flat_failed", "to": 0}
        combiner._current_position = 1
        pm = MagicMock()
        pm.is_killed = True

        forced = _enforce_portfolio_kill(
            {"ETHUSDT": runner},
            {"ETHUSDT": combiner},
            pm,
            {"ETHUSDT": 2000.0},
        )

        assert forced == {"ETHUSDT": {"action": "forced_flat_failed", "to": 0}}
        pm.record_position.assert_not_called()
