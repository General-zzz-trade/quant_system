"""Offline migration of one account runtime into a new identified directory.

The source is locked and read only. SQLite is copied through its backup API so
committed WAL records are included. Publication is one directory rename; an
error never leaves a target that could be mistaken for complete state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

from shared.drawdown_guard import DrawdownGuard
from shared.execution_journal import ExecutionJournal
from shared.execution_lease import ExecutionLease
from shared.runtime_backup import (
    _LOCK_NAME,
    _SIDECARS,
    _destination,
    _digest,
    _inventory,
    _relative,
    _sqlite,
    _sqlite_snapshot,
    _sync_directory,
    _sync_file,
)
from shared.runtime_state import (
    DATABASE_NAMES,
    JOURNAL_NAME,
    MANIFEST_NAME,
    RuntimeStateLease,
    _json,
    _path,
    _scope,
    _DRAWDOWN_COLUMNS,
    validate_runtime_state,
)

MIGRATION_NAME = "runtime-migration.json"


@dataclass(frozen=True)
class MigrationResult:
    target: str
    source_kind: str
    scope: str
    state_id: str
    execution_ready: bool


def _database_inventory(files: dict[str, Path]) -> tuple[set[str], dict[str, Path]]:
    databases = {name for name, path in files.items()
                 if _sqlite(path) and not name.endswith(_SIDECARS)}
    copied: dict[str, Path] = {}
    for name, path in files.items():
        if _LOCK_NAME.fullmatch(path.name):
            continue
        if name.endswith(_SIDECARS):
            base = next(name[:-len(suffix)] for suffix in _SIDECARS if name.endswith(suffix))
            if base not in databases:
                raise ValueError("Runtime migration found an orphan SQLite sidecar")
            continue
        copied[name] = path
    return databases, copied


def _copy_state(source: Path, stage: Path) -> list[dict[str, object]]:
    source_files, directories = _inventory(source)
    databases, files = _database_inventory(source_files)
    for name in directories:
        (stage / name).mkdir(mode=0o700)
    entries = []
    for name, path in sorted(files.items()):
        output = stage / name
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        kind = "sqlite" if name in databases else "file"
        if kind == "sqlite":
            _sqlite_snapshot(path, output)
        else:
            shutil.copyfile(path, output)
        _sync_file(output)
        entries.append({"path": name, "kind": kind, "size": output.stat().st_size,
                        "sha256": _digest(output)})
    for name in reversed(directories):
        _sync_directory(stage / name)
    return entries


def _tables(path: Path) -> dict[str, int]:
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("Migrated SQLite database failed integrity check")
        result = {}
        names = [row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        for name in names:
            if not name.replace("_", "").isalnum():
                raise ValueError("SQLite table name cannot be inventoried safely")
            result[name] = int(db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
        return result
    finally:
        db.close()


def _scope_rows(db: sqlite3.Connection, table: str, scope: str) -> None:
    if not table.replace("_", "").isalnum():
        raise ValueError("SQLite table name cannot be validated safely")
    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    if not columns:
        raise ValueError(f"Legacy runtime is missing {table}")
    if "scope" not in columns:
        raise ValueError(f"Legacy {table} has no account scope; refusing to infer ownership")
    if db.execute(f"SELECT 1 FROM {table} WHERE scope IS NULL OR scope!=? LIMIT 1", (scope,)).fetchone():
        raise ValueError(f"Legacy {table} contains a foreign or unknown account scope")


def _legacy_preflight(source: Path, scope: str) -> None:
    for name in DATABASE_NAMES:
        path = source / name
        if not path.is_file():
            raise ValueError(f"Legacy runtime requires {name}")
        db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError(f"Legacy {name} failed integrity check")
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_identity'").fetchone():
                raise ValueError("Partial runtime identity must be repaired explicitly")
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            expected = ("submissions",) if name == JOURNAL_NAME else tuple(_DRAWDOWN_COLUMNS)
            for table in expected:
                _scope_rows(db, table, scope)
                required = ({"scope", "intent_id", "symbol", "client_id", "order_id", "settled"}
                            if name == JOURNAL_NAME else _DRAWDOWN_COLUMNS[table])
                columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                if not required <= columns:
                    raise ValueError(f"Legacy runtime has an incomplete {table} schema")
            for table in tables:
                if not table.replace("_", "").isalnum():
                    raise ValueError("SQLite table name cannot be validated safely")
                columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                if "scope" in columns:
                    _scope_rows(db, table, scope)
        finally:
            db.close()


def _account_database_scopes(source: Path, scope: str) -> None:
    files, _ = _inventory(source)
    databases, _ = _database_inventory(files)
    for name in databases:
        db = sqlite3.connect(files[name].as_uri() + "?mode=ro", uri=True)
        try:
            tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            if "portfolio_metadata" in tables:
                raise ValueError("Shared portfolio state requires a coordinated group backup")
            for table in tables:
                if not table.replace("_", "").isalnum():
                    raise ValueError("SQLite table name cannot be validated safely")
                columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                if "scope" in columns:
                    _scope_rows(db, table, scope)
        finally:
            db.close()


def _final_inventory(stage: Path) -> list[dict[str, object]]:
    files, _ = _inventory(stage)
    databases, copied = _database_inventory(files)
    entries = []
    for name, path in sorted(copied.items()):
        _sync_file(path)
        entries.append({"path": name, "kind": "sqlite" if name in databases else "file",
                        "size": path.stat().st_size, "sha256": _digest(path)})
    return entries


def _adopt_legacy(stage: Path, scope: str) -> str:
    """Add current identity/schema in the target, preserving every legacy row."""
    state_id = uuid.uuid4().hex
    ExecutionJournal(stage / JOURNAL_NAME, scope)
    DrawdownGuard(stage / "drawdown.sqlite", scope)
    for name in DATABASE_NAMES:
        db = sqlite3.connect(stage / name)
        try:
            with db:
                db.execute("PRAGMA synchronous=FULL")
                if db.execute("SELECT 1 FROM sqlite_master WHERE name='runtime_identity'").fetchone():
                    raise ValueError("Legacy target unexpectedly contains runtime identity")
                db.execute("CREATE TABLE runtime_identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1),"
                           "state_id TEXT NOT NULL,scope TEXT NOT NULL)")
                db.execute("INSERT INTO runtime_identity VALUES (1,?,?)", (state_id, scope))
        finally:
            db.close()
    manifest = {"version": 1, "scope": scope, "journal": JOURNAL_NAME,
                "state_id": state_id, "databases": list(DATABASE_NAMES)}
    path = stage / MANIFEST_NAME
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return state_id


def _facts(stage: Path) -> dict[str, object]:
    journal = sqlite3.connect(stage / JOURNAL_NAME)
    journal.row_factory = sqlite3.Row
    try:
        unresolved = journal.execute("SELECT COUNT(*) FROM submissions WHERE settled=0").fetchone()[0]
        unknown = journal.execute(
            "SELECT COUNT(*) FROM submissions WHERE status IS NULL OR status NOT IN "
            "('filled','canceled','mmp_canceled','rejected','expired')"
        ).fetchone()[0]
        unconsumed = journal.execute("SELECT COUNT(*) FROM execution_fills WHERE delivered=0").fetchone()[0]
        wallet_tables = {row[0] for row in journal.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'wallet_%'"
        )}
    finally:
        journal.close()
    return {"unresolved_submissions": unresolved, "unknown_submissions": unknown,
            "unconsumed_fills": unconsumed, "wallet_tables": sorted(wallet_tables),
            "wallet_reconciliation_required": True}


def _invalidate_migrated_wallet(stage: Path, scope: str, state_id: str) -> None:
    db = sqlite3.connect((stage / JOURNAL_NAME).as_uri() + "?mode=ro", uri=True)
    try:
        present = db.execute("SELECT 1 FROM sqlite_master WHERE name='wallet_metadata'").fetchone()
    finally:
        db.close()
    if present:
        from shared.account_ledger import AccountLedger
        ledger = AccountLedger(stage / JOURNAL_NAME, scope=scope, state_id=state_id)
        ledger.invalidate("runtime migration requires fresh wallet reconciliation")


def _metadata(stage: Path, *, source_kind: str, scope: str, state_id: str,
              entries: list[dict[str, object]]) -> dict[str, object]:
    databases = {}
    for entry in entries:
        if entry["kind"] == "sqlite":
            databases[str(entry["path"])] = _tables(stage / str(entry["path"]))
    return {"version": 1, "source_kind": source_kind, "scope": scope, "state_id": state_id,
            "created_at": int(time.time()), "files": entries, "database_rows": databases,
            "facts": _facts(stage), "execution_ready": False,
            "risk_group_verified": False}


def migrate_runtime_state(source_dir: Path, target_dir: Path, scope: str) -> MigrationResult:
    """Migrate source into a new target without changing or deleting source data."""
    scope = _scope(scope)
    source, target = _path(source_dir), _path(target_dir)
    if not source.is_dir():
        raise ValueError("Runtime migration source directory is missing")
    if source == target or source in target.parents:
        raise ValueError("Runtime migration target must be outside the source")
    if target.exists() or target.is_symlink():
        raise FileExistsError("Runtime migration target already exists")
    managed = (source / MANIFEST_NAME).exists()
    source_kind = "managed" if managed else "legacy"
    # The source must be safe before creating/opening an execution lock; a
    # symlink lock file or database could otherwise reach outside its directory.
    _inventory(source)
    if (source / MIGRATION_NAME).exists():
        raise ValueError("Source already has migration provenance; use the verified backup/restore workflow")
    lease = RuntimeStateLease(source, scope) if managed else ExecutionLease(source / JOURNAL_NAME, scope)
    with lease:
        if not managed:
            _legacy_preflight(source, scope)
        _account_database_scopes(source, scope)
        with _destination(target) as stage:
            _copy_state(source, stage)
            if managed:
                identity = _json((stage / MANIFEST_NAME).read_text(encoding="utf-8"))
                state_id = identity["state_id"]
            else:
                state_id = _adopt_legacy(stage, scope)
            validate_runtime_state(stage, scope)
            _invalidate_migrated_wallet(stage, scope, state_id)
            # Adoption/validation may change database bytes. Inventory the
            # completed target, including its new identity, after those writes.
            entries = _final_inventory(stage)
            metadata = _metadata(stage, source_kind=source_kind, scope=scope,
                                 state_id=state_id, entries=sorted(entries, key=lambda row: str(row["path"])))
            provenance = stage / MIGRATION_NAME
            provenance.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _sync_file(provenance)
            _sync_directory(stage)
    return MigrationResult(str(target), source_kind, scope, state_id, False)


def inspect_migration(target_dir: Path, scope: str) -> dict[str, object]:
    target = _path(target_dir)
    validate_runtime_state(target, _scope(scope), require_writable=False)
    metadata = _json((target / MIGRATION_NAME).read_text(encoding="utf-8"))
    if (metadata.get("version") != 1 or metadata.get("scope") != scope
            or metadata.get("execution_ready") is not False):
        raise ValueError("Runtime migration provenance is invalid")
    identity = _json((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    if metadata.get("state_id") != identity.get("state_id"):
        raise ValueError("Runtime migration provenance identity mismatch")
    expected = metadata.get("files")
    if not isinstance(expected, list):
        raise ValueError("Runtime migration file inventory is missing")
    files, _ = _inventory(target)
    # Provenance hashes describe the checkpointed files at migration completion.
    # A committed WAL-only UPDATE can leave both main-file hash and row counts
    # unchanged. It must not be accepted as that original snapshot.
    if any(name.endswith(("-wal", "-journal")) and path.stat().st_size > 0
           for name, path in files.items()):
        raise ValueError("Runtime migration contains uncheckpointed SQLite changes")
    databases, copied = _database_inventory(files)
    copied.pop(MIGRATION_NAME, None)
    seen = set()
    for entry in expected:
        if not isinstance(entry, dict):
            raise ValueError("Runtime migration file entry is invalid")
        name = _relative(entry.get("path"))
        if name in seen or name not in copied:
            raise ValueError("Runtime migration file inventory differs")
        seen.add(name)
        path = copied[name]
        if (entry.get("kind") != ("sqlite" if name in databases else "file")
                or type(entry.get("size")) is not int or entry["size"] != path.stat().st_size
                or entry.get("sha256") != _digest(path)):
            raise ValueError("Runtime migration checksum or file kind differs")
    if seen != set(copied):
        raise ValueError("Runtime migration file inventory is incomplete")
    rows = {name: _tables(copied[name]) for name in sorted(databases)}
    if metadata.get("database_rows") != rows or metadata.get("facts") != _facts(target):
        raise ValueError("Runtime migration row inventory differs")
    return metadata


def migration_json(result: MigrationResult) -> str:
    return json.dumps(asdict(result), sort_keys=True)
