"""ADR-0016 point 5 (S6): the execution runtime takes the host lock of its account identity after the local checks
and before any account request, holds it while it runs and releases it when it ends. ``state_dir init`` creates
the lock directory; paper mode needs none. Driven through the entry point with the fake venue (ADR-0010 harness).
"""
from __future__ import annotations

import io

import pytest

from tests.characterization.test_startup_account_check import Started, world  # noqa: F401  (fixture)
from trading.adapters.files import host_lock
from trading.adapters.files.host_lock import HostLock
from trading.infrastructure import state_dir as managed
from trading.infrastructure.config import account_identity

IDENTITY = account_identity(testnet=True, api_key="k")        # the fixture's network and key


def test_a_second_process_for_the_same_identity_is_refused_before_any_account_request(world):  # noqa: F811
    with HostLock(IDENTITY, holder="the first runtime"):
        with pytest.raises(RuntimeError, match="already has an execution process"):
            world["main"](world["argv"])
    assert world["fake"].requests == []


def test_two_risk_scopes_with_the_same_key_cannot_run_together(world, tmp_path, monkeypatch):  # noqa: F811
    other = tmp_path / "other"
    assert managed.main(["--state-dir", str(other), "init"], stdout=io.StringIO(),
                        env={"RISK_SCOPE": "binance-testnet-ut2", "BINANCE_TESTNET": "1"}) == 0
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut2")
    argv = world["argv"][:-1] + [str(other)]
    with HostLock(IDENTITY, holder="the runtime of binance-testnet-ut"):
        with pytest.raises(RuntimeError, match="already has an execution process"):
            world["main"](argv)
    assert world["fake"].requests == []


def test_the_runtime_holds_the_lock_while_it_runs_and_it_is_released_after(world, monkeypatch):  # noqa: F811
    import runner.binance_target as entry
    seen = {}

    def stand_in(**kwargs):
        try:
            HostLock(IDENTITY).close()
        except RuntimeError as exc:
            seen["refused"] = str(exc)
        raise Started()
    monkeypatch.setattr(entry, "TargetRuntime", stand_in)
    with pytest.raises(Started):
        world["main"](world["argv"])
    assert "already has an execution process" in seen["refused"]
    HostLock(IDENTITY).close()                                 # released although the start-up ended by an error


def test_a_missing_lock_directory_refuses_the_start_before_any_account_request(world, tmp_path,  # noqa: F811
                                                                                monkeypatch):
    monkeypatch.setattr(host_lock, "LOCK_DIR", tmp_path / "no-locks")
    with pytest.raises(RuntimeError, match="does not exist"):
        world["main"](world["argv"])
    assert world["fake"].requests == [] and not (tmp_path / "no-locks").exists()


def test_paper_mode_needs_no_lock_directory(world, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(host_lock, "LOCK_DIR", tmp_path / "no-locks")
    with pytest.raises(Started):
        world["main"]([a for a in world["argv"] if a != "--enable-execution"])


def test_init_creates_the_lock_directory(tmp_path, monkeypatch):
    locks = tmp_path / "var" / "lib" / "quant" / "locks"
    monkeypatch.setattr(host_lock, "LOCK_DIR", locks)
    assert managed.main(["--state-dir", str(tmp_path / "state"), "init"], stdout=io.StringIO(),
                        env={"RISK_SCOPE": "binance-testnet-ut", "BINANCE_TESTNET": "1"}) == 0
    assert locks.is_dir()


def test_init_that_cannot_create_the_lock_directory_creates_nothing(tmp_path, monkeypatch):
    (tmp_path / "file").write_text("")
    monkeypatch.setattr(host_lock, "LOCK_DIR", tmp_path / "file" / "locks")
    out = io.StringIO()
    code = managed.main(["--state-dir", str(tmp_path / "state"), "init"], stdout=out,
                        env={"RISK_SCOPE": "binance-testnet-ut", "BINANCE_TESTNET": "1"})
    assert code != 0 and "lock directory" in out.getvalue()
    assert not (tmp_path / "state").exists()
