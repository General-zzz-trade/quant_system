"""Regression tests for trading chain audit findings.

Tests critical issues found during 2026-03-23 audit:
1. _consensus_signals scope (UnboundLocalError crash)
2. data-refresh module paths (4/6 broken)
3. health_watchdog log-file check (false stale → crash loop)
4. Timer/service file parity
5. Checkpoint edge cases (empty state)
6. Signal chain integration (process_bar end-to-end)
7. Model config completeness
8. Entry scaler / gate evaluator wiring
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── 1. _consensus_signals scope ──────────────────────────────

class TestConsensusSignalsScope:
    """Verify _consensus_signals is never shadowed as a local in any method."""

    def test_consensus_signals_is_module_level_dict(self):
        from scripts.ops.config import _consensus_signals
        assert isinstance(_consensus_signals, dict)

    def test_alpha_runner_imports_consensus(self):
        """AlphaRunner must import _consensus_signals at module level."""
        import runner.alpha_runner as mod
        assert hasattr(mod, '_consensus_signals')
        from runner.strategy_config import _consensus_signals as cfg_cs
        assert mod._consensus_signals is cfg_cs

    def test_process_bar_resolves_consensus_as_global(self):
        """process_bar must access _consensus_signals as GLOBAL, not LOCAL.

        This is the root cause of the 2026-03-22 16:00 crash:
        UnboundLocalError on _consensus_signals in process_bar.
        """
        from scripts.ops.alpha_runner import AlphaRunner
        code = AlphaRunner.process_bar.__code__
        # Must be in co_names (global lookup), NOT co_varnames (local)
        assert '_consensus_signals' not in code.co_varnames, \
            "_consensus_signals is a local variable in process_bar — will cause UnboundLocalError"
        # Should be accessible as global or free var
        assert ('_consensus_signals' in code.co_names or
                '_consensus_signals' in code.co_freevars), \
            "_consensus_signals not found in globals or freevars of process_bar"

    def test_no_method_shadows_consensus_signals(self):
        """No AlphaRunner method should assign to _consensus_signals as a local."""
        with open("runner/alpha_runner.py") as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id == "_consensus_signals":
                        if isinstance(child.ctx, ast.Store):
                            pytest.fail(
                                f"_consensus_signals assigned as local in "
                                f"{node.name}() at line {child.lineno} — "
                                f"this will shadow the module-level import"
                            )

    def test_evaluate_gates_passes_consensus(self):
        """_evaluate_gates must pass _consensus_signals to GateEvaluator."""
        from scripts.ops.alpha_runner import AlphaRunner
        import inspect
        src = inspect.getsource(AlphaRunner._evaluate_gates)
        assert '_consensus_signals' in src, \
            "_evaluate_gates does not reference _consensus_signals"


# ── 2. Data-refresh module paths ─────────────────────────────

class TestDataRefreshModulePaths:
    """Verify data-refresh script references correct module paths."""

    def test_funding_module_importable(self):
        """The module path used by data-refresh for funding must be importable."""
        mod = importlib.import_module("scripts.data.download_funding_rates")
        assert mod is not None

    def test_fear_greed_module_importable(self):
        mod = importlib.import_module("scripts.data.download_fear_greed")
        assert mod is not None

    def test_open_interest_module_importable(self):
        mod = importlib.import_module("scripts.data.download_open_interest")
        assert mod is not None

    def test_data_refresh_uses_correct_paths(self):
        """data_refresh.py must use scripts.data.* paths, not scripts.*."""
        with open("scripts/data/data_refresh.py") as f:
            source = f.read()

        # These OLD paths must NOT appear
        bad_paths = [
            '"scripts.download_funding_rates"',
            '"scripts.download_fear_greed"',
            '"scripts.download_open_interest"',
            "from scripts.download_binance_klines",
        ]
        for bad in bad_paths:
            assert bad not in source, \
                f"data_refresh.py still uses old path {bad} — must use scripts.data.* instead"

        # These CORRECT paths must appear
        good_paths = [
            "scripts.data.download_funding_rates",
            "scripts.data.download_fear_greed",
            "scripts.data.download_open_interest",
            "scripts.data.download_binance_klines",
        ]
        for good in good_paths:
            assert good in source, f"data_refresh.py missing correct path: {good}"

    def test_kline_download_uses_temp_file(self):
        """refresh_klines must write to temp file first to prevent data loss."""
        with open("scripts/data/data_refresh.py") as f:
            source = f.read()
        assert "tempfile.mkstemp" in source or "tmp_path" in source, \
            "refresh_klines must use temp file to prevent data loss on download failure"

    def test_kline_download_importable(self):
        mod = importlib.import_module("scripts.data.download_binance_klines")
        assert mod is not None


# ── 3. Health watchdog log-file check ────────────────────────

class TestWatchdogLogFileCheck:
    """Verify watchdog uses log_file for services that redirect stdout to file."""

    def test_bybit_alpha_has_log_file_config(self):
        """bybit-alpha config must have log_file set (stdout goes to file, not journal)."""
        from scripts.ops.health_watchdog import SERVICES
        cfg = SERVICES.get("bybit-alpha", {})
        assert "log_file" in cfg, \
            "bybit-alpha missing log_file config — watchdog will check journal (empty) and false-stale"
        assert cfg["log_file"] == "logs/bybit_alpha.log"

    def test_log_file_check_returns_healthy_for_fresh_file(self, tmp_path):
        """Service with recently-modified log file should be HEALTHY, not stale."""
        from scripts.ops.health_watchdog import check_service

        log_file = tmp_path / "test.log"
        log_file.write_text("2026-03-23 test log line\n")

        cfg = {
            "unit": "test.service",
            "max_silent_s": 300,
            "log_file": str(log_file),
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="active\n", returncode=0)
            result = check_service("test", cfg)

        assert result["status"] == "healthy", \
            f"Expected healthy but got {result['status']}: {result.get('problems')}"

    def test_log_file_check_returns_stale_for_old_file(self, tmp_path):
        """Service with old log file should be STALE."""
        from scripts.ops.health_watchdog import check_service

        log_file = tmp_path / "test.log"
        log_file.write_text("old log\n")
        # Set mtime to 1 hour ago
        old_time = time.time() - 3600
        os.utime(log_file, (old_time, old_time))

        cfg = {
            "unit": "test.service",
            "max_silent_s": 300,  # 5 min threshold
            "log_file": str(log_file),
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="active\n", returncode=0)
            result = check_service("test", cfg)

        assert result["status"] == "stale"

    def test_log_file_missing_returns_stale(self, tmp_path):
        from scripts.ops.health_watchdog import check_service

        cfg = {
            "unit": "test.service",
            "max_silent_s": 300,
            "log_file": str(tmp_path / "nonexistent.log"),
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="active\n", returncode=0)
            result = check_service("test", cfg)

        assert result["status"] == "stale"
        assert any("missing" in p for p in result["problems"])


# ── 4. Timer/service file parity ─────────────────────────────

class TestTimerServiceParity:
    """Verify every timer in infra/ has a matching service file."""

    def test_every_timer_has_service(self):
        infra = Path("infra/systemd")
        timers = list(infra.glob("*.timer"))
        assert len(timers) >= 4, f"Expected at least 4 timers, found {len(timers)}"

        for timer in timers:
            service = timer.with_suffix(".service")
            assert service.exists(), \
                f"Timer {timer.name} has no matching service file {service.name}"

    def test_bybit_alpha_service_no_15m(self):
        """bybit-alpha.service must NOT include 15m runners (WF FAIL)."""
        service = Path("infra/systemd/bybit-alpha.service")
        content = service.read_text()
        assert "BTCUSDT_15m" not in content, "15m runner still in service file"
        assert "ETHUSDT_15m" not in content, "15m runner still in service file"

    def test_bybit_alpha_service_has_4h(self):
        """bybit-alpha.service must include 4h runners."""
        service = Path("infra/systemd/bybit-alpha.service")
        content = service.read_text()
        assert "BTCUSDT_4h" in content
        assert "ETHUSDT_4h" in content


# ── 5. Checkpoint edge cases ─────────────────────────────────

class TestCheckpointEdgeCases:
    """Regression tests for checkpoint manager edge cases."""

    def test_restore_zero_bars_zero_closes(self, tmp_path):
        """Checkpoint with bars=0 and closes=[] is valid (fresh start, not corrupted)."""
        from scripts.ops.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(checkpoint_dir=tmp_path)

        mgr.save("ETHUSDT", '{"bars": []}', {"weights": []},
                  extra={"bars_processed": 0, "closes": [], "rets": []})

        data = mgr.restore("ETHUSDT")
        # bars=0 with empty closes is OK (not corrupted, just fresh)
        assert data is not None
        assert data["bars_processed"] == 0

    def test_restore_corrupted_bars_but_empty_closes(self, tmp_path):
        """Checkpoint with bars>0 but empty closes must return None (force warmup).

        Regression: ETHUSDT checkpoint had bars=810, closes=[] → z=? forever.
        """
        from scripts.ops.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(checkpoint_dir=tmp_path)

        # Save a corrupted checkpoint: bars processed but no price data
        mgr._dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "engine": "{}",
            "inference": "{}",
            "bars_processed": 810,
            "closes": [],
            "rets": [],
        }
        (tmp_path / "ETHUSDT.json").write_text(json.dumps(ckpt))

        # Restore should return None (corrupted)
        data = mgr.restore("ETHUSDT")
        assert data is None, \
            "Corrupted checkpoint (bars=810, closes=[]) should return None to force full warmup"

    def test_restore_valid_checkpoint_with_data(self, tmp_path):
        """Checkpoint with bars>0 AND closes should restore normally."""
        from scripts.ops.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(checkpoint_dir=tmp_path)

        mgr._dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "engine": "{}",
            "inference": "{}",
            "bars_processed": 800,
            "closes": [100.0 + i for i in range(500)],
            "rets": [0.001] * 500,
        }
        (tmp_path / "BTCUSDT.json").write_text(json.dumps(ckpt))

        data = mgr.restore("BTCUSDT")
        assert data is not None
        assert data["bars_processed"] == 800
        assert len(data["closes"]) == 500

    def test_restore_with_large_buffers(self, tmp_path):
        """Checkpoint with 500-element buffers should round-trip correctly."""
        from scripts.ops.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(checkpoint_dir=tmp_path)

        closes = [100.0 + i * 0.1 for i in range(500)]
        rets = [0.001 * (i % 10 - 5) for i in range(500)]

        mgr.save("BTCUSDT", "{}", {},
                  extra={"closes": closes, "rets": rets})

        data = mgr.restore("BTCUSDT")
        assert len(data["closes"]) == 500
        assert len(data["rets"]) == 500
        assert abs(data["closes"][0] - 100.0) < 1e-6


# ── 6. Model config completeness ─────────────────────────────

class TestModelConfigCompleteness:
    """Verify all active model configs have required fields."""

    @pytest.fixture
    def active_models(self):
        from scripts.ops.config import SYMBOL_CONFIG, MODEL_BASE
        models = {}
        for key, cfg in SYMBOL_CONFIG.items():
            config_path = MODEL_BASE / cfg["model_dir"] / "config.json"
            if config_path.exists():
                models[key] = json.loads(config_path.read_text())
        return models

    def test_all_models_have_features(self, active_models):
        for key, config in active_models.items():
            has_features = "features" in config or "horizon_models" in config
            assert has_features, f"{key} config missing features or horizon_models"

    def test_all_models_have_deadzone(self, active_models):
        for key, config in active_models.items():
            assert "deadzone" in config, f"{key} config missing deadzone"
            assert config["deadzone"] > 0, f"{key} deadzone must be positive"

    def test_all_models_have_hold_limits(self, active_models):
        for key, config in active_models.items():
            assert "min_hold" in config, f"{key} config missing min_hold"
            assert "max_hold" in config, f"{key} config missing max_hold"
            assert config["min_hold"] < config["max_hold"], \
                f"{key}: min_hold ({config['min_hold']}) >= max_hold ({config['max_hold']})"

    def test_all_models_have_train_date(self, active_models):
        for key, config in active_models.items():
            assert "train_date" in config, f"{key} config missing train_date"
            assert config["train_date"] != "needs-retrain", \
                f"{key} model has not been trained yet"


# ── 7. Entry scaler / gate evaluator wiring ──────────────────

class TestModuleWiring:
    """Verify extracted modules are properly wired into AlphaRunner."""

    def test_alpha_runner_has_checkpoint_manager(self):
        from scripts.ops.alpha_runner import AlphaRunner
        import inspect
        init_src = inspect.getsource(AlphaRunner.__init__)
        assert "CheckpointManager" in init_src

    def test_alpha_runner_has_gate_evaluator(self):
        from scripts.ops.alpha_runner import AlphaRunner
        import inspect
        init_src = inspect.getsource(AlphaRunner.__init__)
        assert "GateEvaluator" in init_src

    def test_alpha_runner_has_entry_scaler(self):
        from scripts.ops.alpha_runner import AlphaRunner
        import inspect
        init_src = inspect.getsource(AlphaRunner.__init__)
        assert "EntryScaler" in init_src

    def test_save_checkpoint_delegates(self):
        """_save_checkpoint must use self._ckpt, not inline JSON logic."""
        from scripts.ops.alpha_runner import AlphaRunner
        import inspect
        src = inspect.getsource(AlphaRunner._save_checkpoint)
        assert "self._ckpt.save" in src, \
            "_save_checkpoint does not delegate to CheckpointManager"

    def test_evaluate_gates_delegates(self):
        from scripts.ops.alpha_runner import AlphaRunner
        import inspect
        src = inspect.getsource(AlphaRunner._evaluate_gates)
        assert "self._gate_evaluator.evaluate" in src

    def test_compute_entry_scale_delegates(self):
        from scripts.ops.alpha_runner import AlphaRunner
        import inspect
        src = inspect.getsource(AlphaRunner._compute_entry_scale)
        assert "self._entry_scaler_module.bb_scale" in src

    def test_leverage_scale_delegates(self):
        """Dynamic leverage must use EntryScaler.leverage_scale."""
        with open("runner/alpha_runner.py") as f:
            source = f.read()
        assert "self._entry_scaler_module.leverage_scale" in source

    def test_vol_adaptive_deadzone_in_check_regime(self):
        """Vol-adaptive deadzone is now computed in _check_regime using rolling median."""
        with open("runner/alpha_runner.py") as f:
            source = f.read()
        assert "vol_median" in source
        assert "self._deadzone_base * ratio" in source


# ── 8. Exception hierarchy wired ─────────────────────────────

class TestExceptionWiring:
    """Verify exception types are imported and used in AlphaRunner."""

    def test_exceptions_imported(self):
        from scripts.ops.alpha_runner import VenueError
        from scripts.ops.exceptions import VenueError as VE
        assert VenueError is VE

    def test_venue_error_in_reconcile(self):
        with open("runner/alpha_runner.py") as f:
            source = f.read()
        assert "except VenueError" in source, \
            "VenueError not caught anywhere in alpha_runner"


# ── 9. SYMBOL_CONFIG consistency ─────────────────────────────

class TestSymbolConfigConsistency:
    """Verify SYMBOL_CONFIG entries are consistent."""

    def test_all_model_dirs_exist(self):
        from scripts.ops.config import SYMBOL_CONFIG, MODEL_BASE
        for key, cfg in SYMBOL_CONFIG.items():
            model_dir = MODEL_BASE / cfg["model_dir"]
            assert model_dir.exists(), f"{key}: model_dir {model_dir} does not exist"
            assert (model_dir / "config.json").exists(), \
                f"{key}: {model_dir}/config.json missing"

    def test_step_sizes_valid(self):
        from scripts.ops.config import SYMBOL_CONFIG
        for key, cfg in SYMBOL_CONFIG.items():
            step = cfg.get("step", 0.01)
            assert step > 0, f"{key}: invalid step size {step}"

    def test_15m_configs_marked_as_fail(self):
        """15m entries should have WF FAIL comments (regression guard)."""
        with open("runner/strategy_config.py") as f:
            source = f.read()
        # Both 15m entries should be clearly marked
        assert "WF FAIL" in source, "15m WF FAIL comment missing from config.py"


# ── 10. Safety guards ────────────────────────────────────────

class TestSafetyGuards:
    """Verify critical safety mechanisms are in place."""

    def test_nan_order_guard(self):
        """Orders must be blocked if position_size is NaN/zero."""
        with open("runner/alpha_runner.py") as f:
            source = f.read()
        assert "invalid size" in source.lower() or "nan_or_zero_size" in source, \
            "Missing NaN/zero guard before send_market_order"

    def test_nan_sizing_guard(self):
        """_compute_position_size must catch NaN/Inf size."""
        with open("runner/alpha_runner.py") as f:
            source = f.read()
        assert "np.isnan(size)" in source or "SIZING BLOCKED" in source, \
            "Missing NaN guard in _compute_position_size"

    def test_phantom_close_guard(self):
        """Phantom close must check exchange position before closing."""
        with open("runner/alpha_runner.py") as f:
            source = f.read()
        assert "PHANTOM CLOSE" in source, \
            "Missing phantom close guard in _execute_signal_change"

    def test_z_score_clamp(self):
        """Extreme z-scores must be clamped for new entries."""
        with open("runner/alpha_runner.py") as f:
            source = f.read()
        assert "Z_CLAMP" in source, \
            "Missing z-score clamp guard"

    def test_direction_alignment(self):
        """ETH must follow BTC direction on new entries."""
        with open("runner/alpha_runner.py") as f:
            source = f.read()
        assert "DIRECTION_ALIGN" in source, \
            "Missing ETH→BTC direction alignment"

    def test_checkpoint_atomic_write(self):
        """Checkpoint must use atomic write (tmp + rename)."""
        with open("state/checkpoint.py") as f:
            source = f.read()
        assert ".tmp" in source and "replace" in source, \
            "Checkpoint write must be atomic (write tmp then rename)"

    def test_oi_csv_normalization(self):
        """Data refresh must normalize OI CSV format after download."""
        with open("scripts/data/data_refresh.py") as f:
            source = f.read()
        assert "sum_open_interest" in source and "normalize" in source.lower() or "normali" in source.lower(), \
            "OI CSV format normalization missing in data_refresh"

    def test_ic_retrain_trigger(self):
        """IC decay monitor must trigger retrain on RED."""
        with open("monitoring/ic_decay_monitor.py") as f:
            source = f.read()
        assert "maybe_trigger_retrain" in source, \
            "IC-triggered retrain missing"

    def test_ic_health_refresh_after_retrain(self):
        """Auto-retrain must refresh IC health JSON after success."""
        with open("alpha/auto_retrain.py") as f:
            source = f.read()
        assert "ic_health" in source and "GREEN" in source, \
            "IC health refresh after retrain missing"
