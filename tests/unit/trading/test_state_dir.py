"""A managed state directory and one kill-file path (migration step 13c; ADR-0007 1c, ADR-0005 5b and 6b).

Execution mode starts only from a directory a person initialized: a manifest naming the account scope, every
store, and a kill file. In such a directory a kill file that disappears while running reads as active. The
kill command takes the state directory (no default) and writes where the runtime reads; the runtime never
writes the kill file itself.
"""
from __future__ import annotations

import ast
import io
import json
from decimal import Decimal as D
from pathlib import Path

import pytest

from trading.adapters.files.kill_switch import KillFile
from trading.infrastructure import kill as kill_cli
from trading.infrastructure import state_dir as managed
from trading.infrastructure.state_dir import MANIFEST, STORES, kill_file_path

ROOT = Path(__file__).resolve().parents[3]
SCOPE = "binance-testnet-ut"
ENV = {"RISK_SCOPE": SCOPE, "BINANCE_TESTNET": "1"}


def _init(path, env=ENV):
    out = io.StringIO()
    return managed.main(["--state-dir", str(path), "init"], stdout=out, env=env), out.getvalue()


def _spec(tmp_path):
    from lab.strategy import StrategySpec
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": ["BTCUSDT"],
                                "spec": StrategySpec().record(), "passed": False}))
    return spec


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", SCOPE)


# -- initialisation ------------------------------------------------------------------------------------------
def test_init_creates_the_manifest_every_store_and_an_inactive_kill_file(tmp_path):
    code, out = _init(tmp_path / "state")
    assert code == 0
    manifest = json.loads((tmp_path / "state" / MANIFEST).read_text())
    assert (manifest["schema"], manifest["scope"], manifest["testnet"]) == (1, SCOPE, True)
    assert all((tmp_path / "state" / name).is_file() for name in STORES)
    assert KillFile(kill_file_path(tmp_path / "state"), missing_is_active=True).state().active is False
    managed.verify(tmp_path / "state", SCOPE)                                 # a fresh directory passes


def test_init_never_overwrites_an_initialized_directory(tmp_path):
    _init(tmp_path / "state")
    KillFile(kill_file_path(tmp_path / "state"), missing_is_active=True).write(active=True, reason="kept")
    code, out = _init(tmp_path / "state")
    assert code == 1 and "already initialized" in out
    assert KillFile(kill_file_path(tmp_path / "state"), missing_is_active=True).state().active is True


def test_init_needs_a_binance_scope(tmp_path):
    with pytest.raises(RuntimeError, match="RISK_SCOPE"):
        _init(tmp_path / "state", env={"BINANCE_TESTNET": "1"})
    assert not (tmp_path / "state").exists()


# -- the start-up check (R6) -----------------------------------------------------------------------------------
@pytest.mark.parametrize("damage,reason", [
    (lambda d: (d / MANIFEST).unlink(), "no manifest"),
    (lambda d: (d / "state.sqlite").unlink(), "state.sqlite is missing"),
    (lambda d: (d / "state.sqlite").write_bytes(b"not a database" * 100), "state.sqlite is unreadable"),
    (lambda d: (d / "kill_switch.json").unlink(), "kill_switch.json is missing"),
    (lambda d: (d / "kill_switch.json").write_text("{broken"), "kill_switch.json is unreadable"),
])
def test_a_damaged_state_directory_refuses_to_start(tmp_path, damage, reason):
    state = tmp_path / "state"
    _init(state)
    damage(state)
    with pytest.raises(RuntimeError, match=reason):
        managed.verify(state, SCOPE)


def test_a_state_directory_of_another_scope_refuses_to_start(tmp_path):
    _init(tmp_path / "state")
    with pytest.raises(RuntimeError, match="belongs to scope binance-testnet-ut, not binance-testnet-other"):
        managed.verify(tmp_path / "state", "binance-testnet-other")


def test_execution_preflight_checks_the_state_directory_before_any_account_access(tmp_path):
    from runner.binance_target import preflight
    spec = _spec(tmp_path)
    with pytest.raises(RuntimeError, match="not initialized"):
        preflight(spec, execution=True, testnet=True, stop_pct=D("0.08"), state_dir=tmp_path / "state")
    assert not (tmp_path / "state").exists()                                  # nothing is created
    _init(tmp_path / "state")
    preflight(spec, execution=True, testnet=True, stop_pct=D("0.08"), state_dir=tmp_path / "state")
    preflight(spec, execution=False, testnet=True, stop_pct=D("0.08"), state_dir=tmp_path / "paper")   # paper


# -- the kill file (R5, 5b, 6b) ----------------------------------------------------------------------------------
def test_in_a_managed_directory_a_kill_file_that_disappears_reads_as_active(tmp_path):
    _init(tmp_path / "state")
    path = kill_file_path(tmp_path / "state")
    path.unlink()
    state = KillFile(path, missing_is_active=True).state()
    assert (state.active, state.flatten) == (True, False) and "missing" in state.reason
    assert KillFile(tmp_path / "unmanaged.json", missing_is_active=False).state().active is False


def test_the_runtime_reads_the_kill_file_the_command_writes(tmp_path):
    from runner.binance_target import TargetRuntime
    from tests.unit.lab.test_target_runtime import _world
    from trading.domain.strategy.overlay import StrategySpec
    state = tmp_path / "state"
    _init(state)
    world = _world(tmp_path, StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05))
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=state, mode="execution", market=world["adapter"],
                            account=world["adapter"], clock=lambda: world["clock"]["now"], testnet=True)
    assert kill_cli.main(["--state-dir", str(state), "activate", "--reason", "test", "--flatten"],
                         stdout=io.StringIO()) == 0
    assert (runtime.kill.state().active, runtime.kill.state().flatten) == (True, True)
    assert kill_cli.main(["--state-dir", str(state), "clear"], stdout=io.StringIO()) == 0
    assert runtime.kill.state().active is False
    kill_file_path(state).unlink()
    assert runtime.kill.state().active is True                                # managed: missing means active


def test_the_kill_command_needs_an_initialized_state_directory(tmp_path):
    with pytest.raises(SystemExit):
        kill_cli.main(["activate"], stdout=io.StringIO())                     # --state-dir is required
    out = io.StringIO()
    assert kill_cli.main(["--state-dir", str(tmp_path / "nowhere"), "activate"], stdout=out) == 2
    assert not (tmp_path / "nowhere").exists() and "not initialized" in out.getvalue()


def test_only_the_kill_command_and_the_loss_stop_s_trip_write_a_kill_file():
    """Owner 2026-11-16: the strategy loss stop sets the kill switch (``KillFile.trip``). It is the runtime's one
    write there, it can only activate, and only the loss stop calls it."""
    writers, trips, trip_writes = [], [], []
    files = [*sorted((ROOT / "trading").rglob("*.py")), ROOT / "runner" / "binance_target.py"]
    for path in files:
        tree, name = ast.parse(path.read_text(encoding="utf-8")), path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr == "write" and any(k.arg == "active" for k in node.keywords):
                writers.append(name)
                if name == "trading/adapters/files/kill_switch.py":
                    trip_writes.append(ast.unparse(next(k.value for k in node.keywords if k.arg == "active")))
            if node.func.attr == "trip":
                trips.append(name)
    assert sorted(set(writers)) == ["trading/adapters/files/kill_switch.py", "trading/infrastructure/kill.py",
                                    "trading/infrastructure/state_dir.py"]
    assert trip_writes == ["True"]                                   # the trip only ever activates
    assert sorted(set(trips)) == ["trading/application/risk/loss_stop.py"]


def test_nothing_else_copies_moves_or_writes_over_the_kill_file():
    """Design audit M1 (S8): the test above only finds ``write(active=...)``; a restore or backup that copied or moved
    the kill file would slip past it. Any copy, move, rename or raw write whose arguments name the kill file is
    allowed only in the kill file's own adapter."""
    allowed = {"trading/adapters/files/kill_switch.py"}
    calls = {"copy", "copy2", "copyfile", "copyfileobj", "move", "replace", "rename", "write_text", "write_bytes",
             "unlink", "symlink_to", "hardlink_to"}

    def names_kill(node):
        return any((isinstance(n, ast.Name) and "kill" in n.id.lower())
                   or (isinstance(n, ast.Attribute) and "kill" in n.attr.lower())
                   or (isinstance(n, ast.Constant) and isinstance(n.value, str) and "kill_switch" in n.value)
                   for n in ast.walk(node))
    found = []
    files = [*sorted((ROOT / "trading").rglob("*.py")), *sorted((ROOT / "runner").rglob("*.py"))]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
            if name in calls and (any(names_kill(a) for a in node.args)
                                  or (receiver is not None and names_kill(receiver))):
                found.append(rel)
    assert sorted(set(found) - allowed) == []


def test_the_entry_point_refuses_execution_from_an_uninitialized_directory_and_creates_nothing(tmp_path):
    from runner.binance_target import main
    spec = _spec(tmp_path)
    with pytest.raises(RuntimeError, match="not initialized"):
        main(["--spec", str(spec), "--enable-execution", "--state-dir", str(tmp_path / "state")])
    assert not (tmp_path / "state").exists()
