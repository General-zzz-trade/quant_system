"""Tests for active run_bybit_alpha runtime wiring."""

import sys
import types
from unittest.mock import MagicMock

import pytest


class TestRunnerSwitch:
    def test_legacy_flag_is_explicitly_rejected_before_runtime_bootstrap(self, monkeypatch, capsys):
        from runner import run_bybit_alpha as mod

        monkeypatch.setattr(mod, "create_adapter", lambda: (_ for _ in ()).throw(AssertionError("should not boot")))

        with pytest.raises(SystemExit) as exc:
            mod.main(["--legacy"])

        assert exc.value.code == 2
        assert "--legacy was removed" in capsys.readouterr().err

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

    def test_runtime_symbols_dedup_timeframe_aliases(self):
        from scripts.ops.run_bybit_alpha import _runtime_symbols

        assert _runtime_symbols(["ETHUSDT", "ETHUSDT_15m", "BTCUSDT"]) == (
            "ETHUSDT",
            "BTCUSDT",
        )

    def test_claim_runtime_symbols_skips_dry_run(self):
        from scripts.ops.run_bybit_alpha import _claim_runtime_symbols

        assert _claim_runtime_symbols(object(), ["ETHUSDT"], dry_run=True) is None

    def test_build_runtime_kill_latch_skips_dry_run(self):
        from scripts.ops.run_bybit_alpha import _build_runtime_kill_latch

        assert _build_runtime_kill_latch(object(), dry_run=True) is None

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

    def test_build_portfolio_combiners_switches_multi_interval_runners_to_signal_only(self, monkeypatch):
        import scripts.ops.run_bybit_alpha as mod

        runners = {
            "ETHUSDT": MagicMock(_state_store=object(), _dry_run=False),
            "ETHUSDT_15m": MagicMock(_state_store=object(), _dry_run=False),
            "BTCUSDT": MagicMock(_state_store=object(), _dry_run=False),
        }
        combo = MagicMock()
        combiner_cls = MagicMock(return_value=combo)
        monkeypatch.setattr(mod, "PortfolioCombiner", combiner_cls)

        combiners = mod._build_portfolio_combiners(
            runners,
            adapter=object(),
            dry_run=False,
            runner_intervals={
                "ETHUSDT": ("ETHUSDT", "60"),
                "ETHUSDT_15m": ("ETHUSDT", "15"),
                "BTCUSDT": ("BTCUSDT", "60"),
            },
        )

        assert combiners == {"ETHUSDT": combo}
        assert runners["ETHUSDT"]._dry_run is True
        assert runners["ETHUSDT_15m"]._dry_run is True
        assert runners["BTCUSDT"]._dry_run is False
        combiner_cls.assert_called_once()

    def test_process_alpha_bar_routes_combo_fill_into_pm(self):
        import scripts.ops.run_bybit_alpha as mod

        runner = MagicMock()
        runner.process_bar.return_value = {
            "action": "signal",
            "bar": 12,
            "close": 2000.0,
            "z": 0.45,
            "signal": 1,
            "hold_count": 3,
            "regime": "active",
            "dz": 0.1,
        }
        combiner = MagicMock()
        combiner.update_signal.return_value = {"to": 1, "fill_price": 1995.5}
        combiner._position_size = 2.0
        pm = MagicMock()
        pm.is_killed = False
        last_prices = {}

        result = mod._process_alpha_bar(
            "ETHUSDT_15m",
            {"close": 2000.0},
            runners={"ETHUSDT_15m": runner},
            runner_intervals={"ETHUSDT_15m": ("ETHUSDT", "15")},
            combiners={"ETHUSDT": combiner},
            last_prices=last_prices,
            portfolio_manager=pm,
            hedge_runner=None,
            mode_label="REST",
        )

        assert result["signal"] == 1
        assert last_prices == {"ETHUSDT": 2000.0}
        combiner.update_signal.assert_called_once_with("ETHUSDT_15m", 1, 2000.0)
        pm.record_position.assert_called_once_with("ETHUSDT", 2.0, 1995.5, "COMBO")

    def test_main_once_mode_routes_rest_bars_through_shared_processor(self, monkeypatch, tmp_path):
        import scripts.ops.run_bybit_alpha as mod

        model_dir = tmp_path / "ETHUSDT_gate_v2"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        shared_state_store = MagicMock()
        fake_hotpath = types.ModuleType("_quant_hotpath")
        fake_hotpath.RustRiskEvaluator = lambda **kwargs: object()
        fake_hotpath.RustKillSwitch = lambda *args, **kwargs: object()
        fake_hotpath.RustStateStore = lambda *args, **kwargs: shared_state_store
        fake_hotpath.rust_pipeline_apply = object()
        fake_hotpath.RustUnifiedPredictor = object
        fake_hotpath.RustTickProcessor = object
        fake_hotpath.RustWsClient = object
        fake_hotpath.RustWsOrderGateway = object
        monkeypatch.setitem(sys.modules, "_quant_hotpath", fake_hotpath)

        monkeypatch.setattr(mod, "MODEL_BASE", tmp_path)
        monkeypatch.setattr(
            mod,
            "SYMBOL_CONFIG",
            {
                "ETHUSDT": {"symbol": "ETHUSDT", "interval": "60", "model_dir": "ETHUSDT_gate_v2", "size": 0.01},
                "ETHUSDT_15m": {"symbol": "ETHUSDT", "interval": "15", "model_dir": "ETHUSDT_gate_v2", "size": 0.01},
            },
        )
        adapter = MagicMock()
        adapter.get_balances.return_value = {"USDT": type("B", (), {"total": 1000.0})()}
        adapter.get_klines.return_value = [{"time": 1, "close": 2000.0}]
        monkeypatch.setattr(mod, "create_adapter", lambda: adapter)
        monkeypatch.setattr(mod, "_build_runtime_kill_latch", lambda *args, **kwargs: None)
        lease = MagicMock()
        monkeypatch.setattr(mod, "_claim_runtime_symbols", lambda *args, **kwargs: lease)
        monkeypatch.setattr(
            mod,
            "load_model",
            lambda _path: {
                "config": {"version": "v-test"},
                "features": [],
                "deadzone": 0.3,
                "min_hold": 1,
                "max_hold": 2,
            },
        )

        built_runners = []

        def fake_runner_ctor(**_kwargs):
            runner = MagicMock()
            runner._state_store = shared_state_store
            runner._dry_run = False
            built_runners.append(runner)
            return runner

        monkeypatch.setattr(mod, "AlphaRunner", fake_runner_ctor)
        pm = MagicMock()
        pm.is_killed = False
        monkeypatch.setattr(mod, "PortfolioManager", lambda *args, **kwargs: pm)
        monkeypatch.setattr(mod, "PortfolioCombiner", MagicMock(return_value=MagicMock()))

        calls = []

        def fake_process(runner_key, bar, **kwargs):
            calls.append((runner_key, bar, kwargs["mode_label"], set(kwargs["combiners"].keys())))
            return {"action": "warmup"}

        monkeypatch.setattr(mod, "_process_alpha_bar", fake_process)
        monkeypatch.setattr(sys, "argv", ["run_bybit_alpha", "--symbols", "ETHUSDT", "ETHUSDT_15m", "--once"])

        mod.main()

        assert [call[0] for call in calls] == ["ETHUSDT", "ETHUSDT_15m"]
        assert all(call[2] == "ONCE" for call in calls)
        assert all(call[3] == {"ETHUSDT"} for call in calls)
        assert all(runner._dry_run is True for runner in built_runners)
        lease.release.assert_called_once()
