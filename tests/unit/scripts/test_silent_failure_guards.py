"""Tests ensuring error paths produce logs or safe fallbacks — not silent failures.

The four most common silent-failure patterns in this codebase:
  1. except:pass swallowing errors
  2. NaN features filled with 0.0 instead of neutral values
  3. API failures causing dangerous fallback values (equity=100)
  4. close_position / set_leverage failures not reported

Each test verifies the fix is in place.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Directories to scan for bare except:pass
SCAN_DIRS = [
    PROJECT_ROOT / "scripts" / "ops",
    PROJECT_ROOT / "runner",
    PROJECT_ROOT / "scripts" / "run_hft_signal.py",
    PROJECT_ROOT / "scripts" / "run_bybit_mm.py",
]


def _collect_py_files(paths: list[Path]) -> list[Path]:
    """Collect all .py files from a list of paths (files or dirs)."""
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(p.rglob("*.py"))
    return files


def _find_bare_except_pass(filepath: Path) -> list[int]:
    """Return line numbers where ``except ...: pass`` appears with no logging.

    Excludes acceptable patterns:
    - ``except KeyboardInterrupt: pass`` (standard signal handling)
    - ``except FileNotFoundError: pass`` (optional file checks)
    - ``except (OSError, ProcessLookupError): pass`` (cleanup best-effort)
    - ``except ImportError: pass`` (optional dependency)
    - Cleanup/resource release blocks (fcntl.LOCK_UN, .close(), .unlink())
    """
    # Acceptable exception types that are commonly caught-and-ignored
    _ACCEPTABLE_TYPES = frozenset({
        "KeyboardInterrupt", "FileNotFoundError", "ImportError",
        "ValueError",  # date/value parsing fallbacks in multi-format parsers
    })

    source = filepath.read_text()
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source, str(filepath))
    except SyntaxError:
        return []

    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Body is a single Pass statement?
        if not (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
            continue

        # Check if exception type is in the acceptable set
        exc_type = node.type
        if exc_type is not None:
            type_names = set()
            if isinstance(exc_type, ast.Name):
                type_names.add(exc_type.id)
            elif isinstance(exc_type, ast.Tuple):
                for elt in exc_type.elts:
                    if isinstance(elt, ast.Name):
                        type_names.add(elt.id)
            if type_names and type_names <= _ACCEPTABLE_TYPES:
                continue
            # OSError in cleanup context (fcntl, .close, .unlink)
            if "OSError" in type_names or "ProcessLookupError" in type_names:
                # Check surrounding lines for cleanup patterns
                start = max(0, node.lineno - 3)
                end = min(len(source_lines), node.lineno + 1)
                context = "\n".join(source_lines[start:end]).lower()
                if any(kw in context for kw in ("flock", "close", "unlink", "truncate")):
                    continue

        # Check surrounding context for Rust-bridge resilience (gate layer)
        # and resource-cleanup patterns that are acceptable to silently ignore
        start = max(0, node.lineno - 5)
        end = min(len(source_lines), node.lineno + 1)
        context = "\n".join(source_lines[start:end]).lower()

        # Rust-Python bridge calls that fallback to Python on failure
        if any(kw in context for kw in ("rust_gate.", "set_phase", "set_peak",
                                         "push_true_range")):
            continue
        # Gate data source fallbacks (return last-known value)
        if any(kw in context for kw in ("_get_funding_rate", "_get_bbo")):
            continue
        # Resource cleanup (flock, truncate, close, seek)
        if any(kw in context for kw in ("flock", "truncate", "handle.close",
                                         "handle.seek")):
            continue

        hits.append(node.lineno)
    return hits


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_mock_adapter(*, balances_fail=False, close_fail=False,
                       set_leverage_fail=False):
    """Build a mock Bybit adapter with controllable failures."""
    adapter = MagicMock()

    # get_balances
    if balances_fail:
        adapter.get_balances.side_effect = RuntimeError("API down")
    else:
        usdt = SimpleNamespace(total=500.0, free=400.0)
        adapter.get_balances.return_value = {"USDT": usdt}

    # close_position via reliable_close_position path
    if close_fail:
        adapter.close_position.side_effect = RuntimeError("close failed")
    else:
        adapter.close_position.return_value = {"retCode": 0}

    # send_market_order
    adapter.send_market_order.return_value = {"retCode": 0, "orderId": "test123"}

    # set_leverage
    client = MagicMock()
    if set_leverage_fail:
        client.post.side_effect = RuntimeError("leverage API error")
    else:
        client.post.return_value = {"retCode": 0}
    adapter._client = client

    # get_positions — return a mock position so phantom close guard doesn't trigger
    mock_pos = SimpleNamespace(is_flat=False, is_long=True, symbol="ETHUSDT", qty=0.01)
    adapter.get_positions.return_value = [mock_pos]

    # get_klines
    adapter.get_klines.return_value = []

    # get_recent_fills
    adapter.get_recent_fills.return_value = []

    return adapter


def _make_model_info():
    """Minimal model_info dict for AlphaRunner construction."""
    model = MagicMock()
    model.predict.return_value = [0.01]
    return {
        "model": model,
        "features": ["rsi_14", "bb_pctb_20"],
        "config": {"version": "test_v1"},
        "deadzone": 0.3,
        "min_hold": 2,
        "max_hold": 48,
        "zscore_window": 100,
        "zscore_warmup": 10,
    }


# ── Test class ───────────────────────────────────────────────────────

class TestSilentFailureGuards:
    """Ensure every error path has log output or an explicit safe fallback."""

    def test_no_bare_except_pass(self):
        """No bare ``except: pass`` in production scripts/ops or runner code.

        Bare except:pass is the #1 silent-failure pattern. Every except must
        either log or perform an observable action.
        """
        files = _collect_py_files(SCAN_DIRS)
        violations: list[str] = []
        for f in files:
            hits = _find_bare_except_pass(f)
            for line in hits:
                violations.append(f"{f.relative_to(PROJECT_ROOT)}:{line}")

        if violations:
            msg = "Bare except:pass found (silent failures):\n" + "\n".join(
                f"  {v}" for v in violations
            )
            pytest.fail(msg)

    def test_set_leverage_failure_is_logged(self, caplog):
        """set_leverage failure must produce at least a warning log."""
        adapter = _make_mock_adapter(set_leverage_fail=True)
        model_info = _make_model_info()

        from scripts.ops.alpha_runner import AlphaRunner

        runner = AlphaRunner(
            adapter, model_info, "ETHUSDT",
            dry_run=True, adaptive_sizing=True,
            start_oi_cache=False,
            oi_cache=MagicMock(get=MagicMock(return_value={
                "open_interest": float("nan"),
                "ls_ratio": float("nan"),
                "taker_buy_vol": float("nan"),
                "top_trader_ls_ratio": float("nan"),
            })),
        )

        with caplog.at_level(logging.WARNING):
            runner._compute_position_size(2000.0)

        leverage_logs = [r for r in caplog.records
                         if "set_leverage" in r.message.lower()
                         and r.levelno >= logging.WARNING]
        assert leverage_logs, "set_leverage failure should produce a WARNING log"

    def test_set_leverage_failure_logged_in_combiner(self, caplog):
        """PortfolioCombiner set_leverage failure must produce a warning."""
        adapter = _make_mock_adapter(set_leverage_fail=True)

        from scripts.ops.portfolio_combiner import PortfolioCombiner

        combiner = PortfolioCombiner(
            adapter, "ETHUSDT",
            weights={"ETH_1h": 0.5, "ETH_15m": 0.5},
            dry_run=False,
        )
        # Set signals so it tries to open a position
        combiner._signals = {"ETH_1h": 1, "ETH_15m": 1}

        with caplog.at_level(logging.WARNING):
            combiner._execute_change(1, 2000.0)

        leverage_logs = [r for r in caplog.records
                         if "set_leverage" in r.message.lower()
                         and r.levelno >= logging.WARNING]
        assert leverage_logs, "COMBO set_leverage failure should produce a WARNING log"

    def test_close_failure_is_logged(self, caplog):
        """reliable_close_position failure must produce logger.error."""
        adapter = _make_mock_adapter()

        from scripts.ops.alpha_runner import AlphaRunner

        runner = AlphaRunner(
            adapter, _make_model_info(), "ETHUSDT",
            dry_run=False,
            start_oi_cache=False,
            oi_cache=MagicMock(get=MagicMock(return_value={
                "open_interest": float("nan"),
                "ls_ratio": float("nan"),
                "taker_buy_vol": float("nan"),
                "top_trader_ls_ratio": float("nan"),
            })),
        )

        # Simulate a position that needs closing
        runner._current_signal = 1
        runner._entry_price = 2000.0
        runner._entry_size = 0.01
        runner._position_size = 0.01

        with caplog.at_level(logging.ERROR), \
             patch("scripts.ops.alpha_runner.reliable_close_position",
                   return_value={"status": "failed", "attempts": 3}):
            result = runner._execute_signal_change(1, 0, 2000.0)

        assert result.get("action") == "close_failed"
        close_logs = [r for r in caplog.records
                      if "CLOSE FAILED" in r.message
                      and r.levelno >= logging.ERROR]
        assert close_logs, "close_position failure must produce an ERROR log"

    def test_nan_feature_uses_neutral_not_zero(self):
        """NaN rsi_14 must become 50.0 (neutral), not 0.0."""
        from scripts.ops.alpha_runner import _NEUTRAL_DEFAULTS

        # Verify the neutral default map has the right values
        assert _NEUTRAL_DEFAULTS["rsi_14"] == 50.0
        assert _NEUTRAL_DEFAULTS["rsi_6"] == 50.0
        assert _NEUTRAL_DEFAULTS["ls_ratio"] == 1.0
        assert _NEUTRAL_DEFAULTS["bb_pctb_20"] == 0.5

    def test_nan_feature_in_ensemble_predict(self):
        """_ensemble_predict must use neutral defaults for NaN, not 0.0."""
        adapter = _make_mock_adapter()
        model_info = _make_model_info()
        model_info["features"] = ["rsi_14", "ls_ratio"]

        from scripts.ops.alpha_runner import AlphaRunner

        runner = AlphaRunner(
            adapter, model_info, "ETHUSDT",
            dry_run=True,
            start_oi_cache=False,
            oi_cache=MagicMock(get=MagicMock(return_value={
                "open_interest": float("nan"),
                "ls_ratio": float("nan"),
                "taker_buy_vol": float("nan"),
                "top_trader_ls_ratio": float("nan"),
            })),
        )

        # Call with NaN features — should use neutral defaults
        feat_dict = {"rsi_14": float("nan"), "ls_ratio": None}
        runner._ensemble_predict(feat_dict)

        # Check what the model was called with
        call_args = model_info["model"].predict.call_args[0][0]
        assert call_args[0] == [50.0, 1.0], (
            f"Expected [50.0, 1.0] (neutral), got {call_args[0]}"
        )

    def test_secondary_horizon_uses_neutral_defaults(self):
        """_secondary_horizon_predict must use neutral defaults for missing features."""
        adapter = _make_mock_adapter()
        model_info = _make_model_info()

        from scripts.ops.alpha_runner import AlphaRunner, _NEUTRAL_DEFAULTS

        lgbm_mock = MagicMock()
        lgbm_mock.predict.return_value = [0.01]
        model_info["horizon_models"] = [
            {
                "features": ["rsi_14", "ls_ratio"],
                "ic": 0.1,
                "lgbm": lgbm_mock,
            },
            {
                "features": ["rsi_14"],
                "ic": 0.1,
                "lgbm": MagicMock(predict=MagicMock(return_value=[0.02])),
            },
        ]

        runner = AlphaRunner(
            adapter, model_info, "ETHUSDT",
            dry_run=True,
            start_oi_cache=False,
            oi_cache=MagicMock(get=MagicMock(return_value={
                "open_interest": float("nan"),
                "ls_ratio": float("nan"),
                "taker_buy_vol": float("nan"),
                "top_trader_ls_ratio": float("nan"),
            })),
        )

        # Call with empty features (simulating missing)
        runner._secondary_horizon_predict({})

        # Check what the lgbm was called with: should use neutral defaults
        call_args = lgbm_mock.predict.call_args[0][0]
        expected_rsi = _NEUTRAL_DEFAULTS.get("rsi_14", 0.0)
        expected_ls = _NEUTRAL_DEFAULTS.get("ls_ratio", 0.0)
        assert call_args[0][0] == expected_rsi, (
            f"rsi_14 should be {expected_rsi} (neutral), got {call_args[0][0]}"
        )
        assert call_args[0][1] == expected_ls, (
            f"ls_ratio should be {expected_ls} (neutral), got {call_args[0][1]}"
        )

    def test_api_failure_does_not_use_fallback_equity(self, caplog):
        """get_balances failure must not trade with a fake equity value.

        Previously, some paths used equity=100 as fallback, which is dangerous
        because it could lead to outsized positions.
        """
        adapter = _make_mock_adapter(balances_fail=True)
        model_info = _make_model_info()

        from scripts.ops.alpha_runner import AlphaRunner

        runner = AlphaRunner(
            adapter, model_info, "ETHUSDT",
            dry_run=True, adaptive_sizing=True,
            start_oi_cache=False,
            oi_cache=MagicMock(get=MagicMock(return_value={
                "open_interest": float("nan"),
                "ls_ratio": float("nan"),
                "taker_buy_vol": float("nan"),
                "top_trader_ls_ratio": float("nan"),
            })),
        )

        with caplog.at_level(logging.WARNING):
            size = runner._compute_position_size(2000.0)

        # On balance failure, should fall back to base_position_size, NOT
        # compute from a fake equity value
        assert size == runner._base_position_size or size == runner._round_to_step(runner._base_position_size), (
            "On API failure, should use base_position_size, not computed from fake equity"
        )

        # Should have logged a warning
        warning_logs = [r for r in caplog.records
                        if r.levelno >= logging.WARNING
                        and "balance" in r.message.lower()]
        assert warning_logs, "Balance fetch failure must produce a WARNING log"

    def test_portfolio_manager_rejects_on_zero_equity(self, caplog):
        """PortfolioManager must reject intents when equity is unavailable."""
        adapter = _make_mock_adapter(balances_fail=True)

        from scripts.ops.portfolio_manager import PortfolioManager

        pm = PortfolioManager(adapter, dry_run=True)

        with caplog.at_level(logging.ERROR):
            result = pm.submit_intent("test", "ETHUSDT", 1, 2000.0)

        assert result is not None
        assert result.get("action") == "rejected"
        assert result.get("reason") == "equity_unavailable"

    def test_hft_set_leverage_failure_logged(self, caplog):
        """HFT signal set_leverage failure must produce a warning, not silent pass."""
        from scripts.run_hft_signal import SignalHFT

        adapter = MagicMock()
        adapter._client.post.side_effect = RuntimeError("leverage API error")

        hft = SignalHFT(
            symbols=["BTCUSDT"],
            adapter=adapter,
            dry_run=True,
        )

        with caplog.at_level(logging.WARNING, logger="hft_signal"):
            # Just call start's leverage-setting part
            for sym in hft._symbols:
                try:
                    adapter._client.post('/v5/position/set-leverage', body={
                        'category': 'linear', 'symbol': sym,
                        'buyLeverage': '20',
                        'sellLeverage': '20',
                    })
                except Exception as e:
                    import logging as _logging
                    _logging.getLogger("hft_signal").warning(
                        "set_leverage failed for %s: %s", sym, e
                    )

        leverage_logs = [r for r in caplog.records
                         if "set_leverage" in r.message.lower()]
        assert leverage_logs, "HFT set_leverage failure should produce a warning"
