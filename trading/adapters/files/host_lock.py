"""One execution process per account identity on this host (ADR-0016 point 5, S6; R20).

The lock file is ``<lock dir>/<identity>.lock``, where the identity is sha256(network:API key)
(``trading.infrastructure.config.account_identity``). The lock directory is fixed on the host, apart from any
state directory, so two state directories or two RISK_SCOPEs with the same key cannot run together. The
execution runtime, the manual tools and the restore command take it; paper mode does not.

Only ``state_dir init`` (a person) creates the directory, through ``prepare_lock_dir``; taking the lock never
creates it, and a missing or unusable directory refuses. The operating system releases the lock when its holder
ends, however it ends. Another host cannot be stopped from here: the runbook says one host per account identity.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import IO, Any

DEFAULT_LOCK_DIR = Path("/var/lib/quant/locks")
LOCK_DIR = DEFAULT_LOCK_DIR             # read at each call; tests point it at their own directory
_IDENTITY = re.compile(r"[0-9a-f]{64}")
_INIT = "python -m trading.infrastructure.state_dir --state-dir <state dir> init"


def _open(path: Path) -> IO[bytes]:
    return open(os.open(path, os.O_RDWR | os.O_CREAT, 0o644), "r+b")


def _lock(handle: IO[bytes]) -> None:
    if sys.platform == "win32":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _holder(path: Path) -> str:
    try:
        return path.read_text(errors="replace").strip() or "unknown"
    except OSError:
        return "unknown"


class HostLock:
    """Held from construction until ``close`` (or the end of the process). Raises RuntimeError when another
    process holds the identity or the lock directory is missing or unusable; ValueError for a malformed identity."""

    def __init__(self, identity: str, *, holder: str = "", lock_dir: Path | None = None) -> None:
        if not isinstance(identity, str) or not _IDENTITY.fullmatch(identity):
            raise ValueError("an account identity is a sha256 hex digest")
        directory = Path(LOCK_DIR if lock_dir is None else lock_dir)
        if not directory.exists():
            raise RuntimeError(f"host lock directory {directory} does not exist; a person creates it with: {_INIT}")
        self.path = directory / f"{identity}.lock"
        try:
            if not directory.is_dir():
                raise NotADirectoryError(f"{directory} is not a directory")
            self._file: Any = _open(self.path)
        except OSError as exc:
            raise RuntimeError(f"host lock directory {directory} is not usable ({exc})") from None
        try:
            _lock(self._file)
        except OSError:
            self._file.close()
            raise RuntimeError(f"this account identity already has an execution process on this host "
                               f"(lock {self.path}, held by {_holder(self.path)})") from None
        self._file.seek(0)
        self._file.truncate()
        self._file.write(f"pid={os.getpid()} {holder}".rstrip().encode() + b"\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()          # the operating system releases the lock

    def __enter__(self) -> HostLock:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def prepare_lock_dir(lock_dir: Path | None = None) -> Path:
    """Create the host lock directory if it is missing (``state_dir init`` only). Raises RuntimeError when it
    cannot be created or written."""
    directory = Path(LOCK_DIR if lock_dir is None else lock_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise PermissionError(f"{directory} is not writable")
    except OSError as exc:
        raise RuntimeError(f"cannot prepare the host lock directory {directory} ({exc}); create it as root "
                           "and give it to the user the service runs as") from None
    return directory


__all__ = ["DEFAULT_LOCK_DIR", "LOCK_DIR", "HostLock", "prepare_lock_dir"]
