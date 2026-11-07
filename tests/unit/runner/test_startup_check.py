"""Installation smoke checks must never create account clients or load models."""
import json
import socket
import subprocess
import sys
from unittest.mock import patch

import pytest

from runner.alpha_cli import parse_args
from runner.startup_check import check_startup, no_network


def test_check_startup_is_explicitly_offline(monkeypatch):
    monkeypatch.delenv("RISK_SCOPE", raising=False)
    assert parse_args(["--check-startup"]).dry_run
    with pytest.raises(SystemExit) as error:
        parse_args(["--check-startup", "--enable-execution"])
    assert error.value.code == 2


def test_network_guard_blocks_connections_and_restores_socket_methods():
    original = socket.create_connection
    with no_network():
        with pytest.raises(RuntimeError, match="network is forbidden"):
            socket.create_connection(("example.invalid", 443))
        with pytest.raises(RuntimeError, match="network is forbidden"):
            socket.getaddrinfo("example.invalid", 443)
        with socket.socket() as sock:
            with pytest.raises(RuntimeError, match="network is forbidden"):
                sock.connect_ex(("127.0.0.1", 1))
    assert socket.create_connection is original


def test_real_native_market_state_advances():
    result = check_startup()
    assert result["status"] == "passed"
    assert result["event_index"] == 1
    assert result["close"] == "101.0" or result["close"] == "101"
    assert result["models_checked"] is False
    assert result["exchange_checked"] is False


def test_entry_check_returns_before_models_cache_or_account_access(monkeypatch, capsys):
    from runner import alpha_main
    monkeypatch.setattr(sys, "argv", ["alpha_main", "--check-startup"])
    monkeypatch.setenv("RUNTIME_STATE_DIR", "/deliberately-missing-offline-state")
    # A missing online cache must not affect the offline engine check.
    monkeypatch.setitem(sys.modules, "data.oi_cache", None)
    with (
        patch.object(alpha_main, "create_adapter", side_effect=AssertionError("account accessed")),
        patch.object(alpha_main, "_create_binance_adapter", side_effect=AssertionError("account accessed")),
        patch.object(alpha_main, "_create_okx_adapter", side_effect=AssertionError("account accessed")),
        patch.object(alpha_main, "load_model", side_effect=AssertionError("model accessed")),
    ):
        alpha_main.main()
    assert json.loads(capsys.readouterr().out)["scope"] == "offline-native-market-to-state"


def test_missing_online_input_fails_before_account_access(monkeypatch):
    from runner import alpha_main
    monkeypatch.setattr(sys, "argv", ["alpha_main", "--dry-run"])
    monkeypatch.setitem(sys.modules, "data.oi_cache", None)
    with patch.object(alpha_main, "create_adapter", side_effect=AssertionError("account accessed")):
        with pytest.raises(RuntimeError, match="Online startup blocked"):
            alpha_main.main()


@pytest.mark.parametrize("defect", ["missing", "wrong_scope", "corrupt", "journal_outside", "drawdown_outside"])
def test_managed_state_failure_blocks_before_account_or_online_input(monkeypatch, tmp_path, defect):
    import sqlite3
    from runner import alpha_main
    from shared.runtime_state import initialize_runtime_state

    monkeypatch.chdir(tmp_path)
    state = tmp_path / "data" / "runtime"
    initialize_runtime_state(state, "paper-test")
    monkeypatch.setattr(sys, "argv", ["alpha_main", "--dry-run"])
    monkeypatch.setenv("RUNTIME_STATE_DIR", str(state))
    monkeypatch.setenv("RISK_SCOPE", "wrong" if defect == "wrong_scope" else "paper-test")
    monkeypatch.setenv("ORDER_JOURNAL_PATH", str(state / "orders.sqlite"))
    monkeypatch.setenv("DRAWDOWN_STATE_PATH", str(state / "drawdown.sqlite"))
    monkeypatch.setitem(sys.modules, "data.oi_cache", None)
    if defect == "missing":
        (state / "orders.sqlite").unlink()
    elif defect == "corrupt":
        (state / "orders.sqlite").write_bytes(b"not a SQLite database")
    elif defect == "journal_outside":
        monkeypatch.setenv("ORDER_JOURNAL_PATH", str(tmp_path / "other.sqlite"))
    elif defect == "drawdown_outside":
        monkeypatch.setenv("DRAWDOWN_STATE_PATH", str(tmp_path / "drawdown.sqlite"))
    with patch.object(alpha_main, "create_adapter", side_effect=AssertionError("account accessed")):
        with pytest.raises((ValueError, RuntimeError, sqlite3.Error)) as error:
            alpha_main.main()
    assert "Online startup blocked" not in str(error.value)
    if defect == "missing":
        assert not (state / "orders.sqlite").exists()


def test_managed_paper_startup_holds_lease_before_online_inputs(monkeypatch, tmp_path):
    import atexit
    from runner import alpha_main
    from shared.runtime_state import initialize_runtime_state, RuntimeStateLease

    monkeypatch.chdir(tmp_path)
    state = tmp_path / "data" / "runtime"
    initialize_runtime_state(state, "paper-test")
    monkeypatch.setattr(sys, "argv", ["alpha_main", "--dry-run"])
    monkeypatch.setenv("RUNTIME_STATE_DIR", str(state))
    monkeypatch.setenv("RISK_SCOPE", "paper-test")
    monkeypatch.setenv("ORDER_JOURNAL_PATH", str(state / "orders.sqlite"))
    monkeypatch.setenv("DRAWDOWN_STATE_PATH", str(state / "drawdown.sqlite"))
    monkeypatch.setitem(sys.modules, "data.oi_cache", None)
    callbacks = []
    monkeypatch.setattr(atexit, "register", callbacks.append)
    try:
        with patch.object(alpha_main, "create_adapter", side_effect=AssertionError("account accessed")):
            with pytest.raises(RuntimeError, match="Online startup blocked"):
                alpha_main.main()
        assert len(callbacks) == 1
        with pytest.raises(RuntimeError, match="already has an execution process"):
            RuntimeStateLease(state, "paper-test")
    finally:
        for callback in callbacks:
            callback()
    with RuntimeStateLease(state, "paper-test"):
        pass


def test_missing_native_extension_cannot_report_success():
    source = '''
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "_quant_hotpath":
        raise ModuleNotFoundError("deliberately missing native extension")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from runner.startup_check import check_startup
check_startup()
'''
    result = subprocess.run([sys.executable, "-B", "-c", source], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "deliberately missing native extension" in result.stderr


@pytest.mark.parametrize("entry", ["runner.alpha_main", "scripts.run_bybit_alpha"])
def test_canonical_and_compatibility_entrypoints_have_help(entry):
    result = subprocess.run([sys.executable, "-B", "-m", entry, "--help"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--check-startup" in result.stdout


def test_compatibility_entrypoint_runs_the_offline_check():
    result = subprocess.run([sys.executable, "-B", "-m", "scripts.run_bybit_alpha", "--check-startup"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["scope"] == "offline-native-market-to-state"
