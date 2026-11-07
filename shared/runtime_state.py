"""Explicit initialization and fail-closed validation of local execution state.

The state directory must live on a persistent local filesystem. This module
does not infer persistence from a path; the deployment preflight checks mounts.
Checking an existing state never creates an empty replacement order database.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any

from shared.drawdown_guard import DrawdownGuard
from shared.execution_journal import ExecutionJournal
from shared.execution_lease import ExecutionLease

MANIFEST_NAME = "runtime-state.json"
JOURNAL_NAME = "orders.sqlite"
DATABASE_NAMES = (JOURNAL_NAME, "drawdown.sqlite")
_REQUIRED_COLUMNS = {
    "submissions": {"scope", "intent_id", "symbol", "client_id", "order_id", "settled",
                    "side", "qty", "reducing", "owner", "parent", "status", "order_type"},
    "execution_fills": {"scope", "fill_id", "intent_id", "owner", "payload", "delivered"},
    "execution_consumers": {"scope", "owner", "checkpoint"},
}
_DRAWDOWN_COLUMNS = {
    "drawdown_periods": {"scope", "kind", "period", "baseline", "limit_pct", "tripped"},
    "equity_observations": {"scope", "equity", "observed_at"},
}


def _scope(scope: str) -> str:
    if not isinstance(scope, str) or not scope or scope != scope.strip() or "\x00" in scope:
        raise ValueError("Runtime state requires an explicit scope without surrounding whitespace")
    return scope


def _path(path: Path) -> Path:
    value = Path(path).absolute()
    if ".." in value.parts:
        raise ValueError("Runtime state paths cannot contain parent traversal")
    for candidate in (*reversed(value.parents), value):
        if candidate.is_symlink():
            raise ValueError("Runtime state paths cannot contain symbolic links")
    return value


def _lock_path(journal: Path, scope: str) -> Path:
    key = hashlib.sha256(scope.encode()).hexdigest()[:24]
    return journal.with_name(f"execution-{key}.lock")


def _json(text: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Runtime state JSON contains duplicate keys")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError("Runtime state JSON contains a non-finite number")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Runtime state JSON must be an object")
    return value


def _layout(state_dir: Path, scope: str, journal_path: Path | None) -> tuple[Path, Path, dict[str, Any]]:
    _scope(scope)
    directory = _path(state_dir)
    if not directory.is_dir():
        raise ValueError("Runtime state directory is missing; explicit initialization or restore required")
    journal = _path(directory / JOURNAL_NAME)
    if journal_path is not None and _path(journal_path) != journal:
        raise ValueError("Order journal path must be orders.sqlite in the configured runtime state directory")
    manifest_path = _path(directory / MANIFEST_NAME)
    if not manifest_path.is_file() or not journal.is_file():
        raise ValueError("Runtime state identity or order database is missing; refusing an empty replacement")
    if not stat.S_ISREG(manifest_path.stat().st_mode) or not stat.S_ISREG(journal.stat().st_mode):
        raise ValueError("Runtime identity and order database must be regular files")
    manifest = _json(manifest_path.read_text(encoding="utf-8"))
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("Unsupported runtime state identity version")
    if manifest.get("scope") != scope or manifest.get("journal") != JOURNAL_NAME:
        raise ValueError("Runtime state identity does not match the configured scope or journal")
    if manifest.get("databases") != list(DATABASE_NAMES):
        raise ValueError("Runtime state identity must declare the order and drawdown databases")
    state_id = manifest.get("state_id")
    if not isinstance(state_id, str) or len(state_id) != 32 or uuid.UUID(hex=state_id).hex != state_id:
        raise ValueError("Invalid runtime state identity")
    for name in DATABASE_NAMES:
        database = _path(directory / name)
        if not database.is_file():
            raise ValueError(f"Required runtime database {name} is missing; refusing an empty replacement")
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = _path(Path(str(database) + suffix))
            if sidecar.exists() and not sidecar.is_file():
                raise ValueError("SQLite sidecars must be regular files")
    lock = _path(_lock_path(journal, scope))
    if lock.exists() and not lock.is_file():
        raise ValueError("Execution lease must be a regular file")
    return directory, journal, manifest


def _writable(directory: Path, journal: Path | None = None) -> None:
    # Check mode bits as well as a real write, including when tests run as root.
    if not directory.stat().st_mode & 0o222 or not os.access(directory, os.W_OK | os.X_OK):
        raise ValueError("Runtime state directory is not writable")
    if journal is not None and (not journal.stat().st_mode & 0o222 or not os.access(journal, os.W_OK)):
        raise ValueError("Runtime database is not writable")
    with tempfile.NamedTemporaryFile(prefix=".runtime-write-check-", dir=directory) as probe:
        probe.write(b"runtime state write check")
        probe.flush()
        os.fsync(probe.fileno())


def validate_runtime_state(
    state_dir: Path, scope: str, *, journal_path: Path | None = None, require_writable: bool = True,
) -> Path:
    """Validate an existing state without acquiring a long-lived execution lease.

    Callers starting execution must use RuntimeStateLease for the full process
    lifetime. This function is also usable while validating backup snapshots.
    """
    directory, journal, manifest = _layout(state_dir, scope, journal_path)
    mode = "rw" if require_writable else "ro"
    for name in DATABASE_NAMES:
        database = directory / name
        if require_writable:
            _writable(directory, database)
        db = sqlite3.connect(f"{database.as_uri()}?mode={mode}", uri=True, timeout=1)
        try:
            if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError(f"Runtime database {name} failed SQLite integrity check")
            if name == JOURNAL_NAME and str(db.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
                raise ValueError("Order database must preserve WAL journal mode")
            identities = db.execute("SELECT singleton, state_id, scope FROM runtime_identity").fetchall()
            if identities != [(1, manifest["state_id"], scope)]:
                raise ValueError(f"Runtime database {name} identity differs from runtime state identity")
            required_columns = _REQUIRED_COLUMNS if name == JOURNAL_NAME else _DRAWDOWN_COLUMNS
            for table, required in required_columns.items():
                columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                if not required <= columns:
                    raise ValueError(f"Runtime database {name} has an incomplete {table} schema")
                foreign = db.execute(f"SELECT 1 FROM {table} WHERE scope IS NULL OR scope != ? LIMIT 1", (scope,))
                if foreign.fetchone():
                    raise ValueError(f"Runtime database {name} contains data from a different account scope")
            if name == JOURNAL_NAME:
                for table, column in (("execution_fills", "payload"), ("execution_consumers", "checkpoint")):
                    for row in db.execute(f"SELECT {column} FROM {table}"):
                        _json(row[0])
            if require_writable:
                # Exercise each database's real write path without changing a record.
                db.execute("BEGIN IMMEDIATE")
                db.rollback()
        finally:
            db.close()
    return journal


class RuntimeStateLease:
    """Validate and hold the account's existing single-writer lease until close."""

    def __init__(self, state_dir: Path, scope: str, *, journal_path: Path | None = None) -> None:
        _, journal, _ = _layout(state_dir, scope, journal_path)
        self._lease = ExecutionLease(journal, scope)
        try:
            self.journal_path = validate_runtime_state(state_dir, scope, journal_path=journal)
        except BaseException:
            self._lease.close()
            raise

    def close(self) -> None:
        self._lease.close()

    def __enter__(self) -> RuntimeStateLease:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def initialize_runtime_state(state_dir: Path, scope: str) -> Path:
    """Create a new empty account journal only through an explicit init action.

    Existing or partial state is never overwritten or silently adopted.
    """
    _scope(scope)
    directory = _path(state_dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Runtime state initialization requires an empty directory")
    _writable(directory)
    journal = directory / JOURNAL_NAME
    with ExecutionLease(journal, scope):
        if set(directory.iterdir()) != {_lock_path(journal, scope)}:
            raise ValueError("Runtime state appeared during initialization; refusing to overwrite")
        state_id = uuid.uuid4().hex
        ExecutionJournal(journal, scope)
        # Constructor creates schema only: no equity observation or risk calculation.
        DrawdownGuard(directory / "drawdown.sqlite", scope)
        for name in DATABASE_NAMES:
            db = sqlite3.connect(directory / name)
            try:
                with db:
                    db.execute("PRAGMA synchronous=FULL")
                    db.execute("CREATE TABLE runtime_identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                               "state_id TEXT NOT NULL, scope TEXT NOT NULL)")
                    db.execute("INSERT INTO runtime_identity VALUES (1, ?, ?)", (state_id, scope))
            finally:
                db.close()
        manifest = {"version": 1, "scope": scope, "journal": JOURNAL_NAME, "state_id": state_id,
                    "databases": list(DATABASE_NAMES)}
        with (directory / MANIFEST_NAME).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return validate_runtime_state(directory, scope)
