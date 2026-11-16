"""ADR-0016 point 5 (S6): one execution process per account identity on this host.

The lock file is ``<lock dir>/<identity>.lock``; the lock directory is fixed on the host, apart from any state
directory, and only ``state_dir init`` creates it. The operating system releases the lock when its holder ends.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from trading.adapters.files import host_lock
from trading.adapters.files.host_lock import HostLock, prepare_lock_dir
from trading.infrastructure.config import account_identity

ROOT = Path(__file__).resolve().parents[3]
A = account_identity(testnet=True, api_key="key-a")
B = account_identity(testnet=True, api_key="key-b")


def test_a_second_lock_for_the_same_identity_is_refused(tmp_path):
    with HostLock(A, lock_dir=tmp_path):
        with pytest.raises(RuntimeError, match="already has an execution process"):
            HostLock(A, lock_dir=tmp_path)


def test_other_keys_and_the_other_network_are_other_identities(tmp_path):
    mainnet = account_identity(testnet=False, api_key="key-a")
    with HostLock(A, lock_dir=tmp_path), HostLock(B, lock_dir=tmp_path), HostLock(mainnet, lock_dir=tmp_path):
        pass


def test_the_lock_is_released_on_close(tmp_path):
    HostLock(A, lock_dir=tmp_path).close()
    HostLock(A, lock_dir=tmp_path).close()


def test_another_process_is_refused_and_the_lock_goes_with_its_holder(tmp_path):
    code = ("import sys\n"
            "from pathlib import Path\n"
            "from trading.adapters.files.host_lock import HostLock\n"
            "try:\n"
            "    HostLock(sys.argv[1], lock_dir=Path(sys.argv[2])).close()\n"
            "except RuntimeError:\n"
            "    sys.exit(42)\n")

    def attempt():
        return subprocess.run([sys.executable, "-c", code, A, str(tmp_path)], cwd=ROOT, capture_output=True,
                              timeout=30).returncode
    with HostLock(A, lock_dir=tmp_path):
        assert attempt() == 42
    assert attempt() == 0


def test_the_refusal_names_the_holder(tmp_path):
    with HostLock(A, lock_dir=tmp_path, holder="binance_target --state-dir /srv/a"):
        with pytest.raises(RuntimeError, match=f"pid={os.getpid()} .*binance_target --state-dir /srv/a"):
            HostLock(A, lock_dir=tmp_path)


def test_a_missing_lock_directory_is_refused_and_never_created(tmp_path):
    missing = tmp_path / "locks"
    with pytest.raises(RuntimeError, match="does not exist.*state_dir"):
        HostLock(A, lock_dir=missing)
    assert not missing.exists()


def test_a_lock_directory_that_is_a_file_is_refused(tmp_path):
    (tmp_path / "locks").write_text("")
    with pytest.raises(RuntimeError, match="not usable"):
        HostLock(A, lock_dir=tmp_path / "locks")


def test_a_lock_file_that_cannot_be_opened_is_refused(tmp_path, monkeypatch):
    # Permission bits do not stop root, so the open itself is made to fail.
    def denied(path):
        raise PermissionError(13, "Permission denied", str(path))
    monkeypatch.setattr(host_lock, "_open", denied)
    with pytest.raises(RuntimeError, match="not usable"):
        HostLock(A, lock_dir=tmp_path)


def test_only_an_identity_digest_names_a_lock_file(tmp_path):
    for bad in ("", "../escape", "k", A.upper(), A[:-1]):
        with pytest.raises(ValueError):
            HostLock(bad, lock_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_the_default_lock_directory_is_the_fixed_host_directory():
    assert host_lock.DEFAULT_LOCK_DIR == Path("/var/lib/quant/locks")


def test_prepare_creates_the_directory_and_can_be_repeated(tmp_path):
    locks = tmp_path / "var" / "lib" / "quant" / "locks"
    assert prepare_lock_dir(locks) == locks and locks.is_dir()
    assert prepare_lock_dir(locks) == locks
    HostLock(A, lock_dir=locks).close()


def test_prepare_refuses_a_directory_it_cannot_create(tmp_path):
    (tmp_path / "file").write_text("")
    with pytest.raises(RuntimeError, match="lock directory"):
        prepare_lock_dir(tmp_path / "file" / "locks")


def test_a_new_holder_replaces_the_old_text_entirely(tmp_path):
    # S6-S8 audit, mutant C7: a shorter holder must not leave the end of a longer one behind.
    HostLock(A, lock_dir=tmp_path, holder="x" * 200).close()
    with HostLock(A, lock_dir=tmp_path, holder="short"):
        assert (tmp_path / f"{A}.lock").read_text() == f"pid={os.getpid()} short\n"
