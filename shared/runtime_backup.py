"""Offline snapshots for initialized runtime state on a local filesystem.

Stop all state writers before taking a snapshot. The execution lease excludes
cooperating runners; SQLite's backup API includes committed WAL records. Each
database is consistent individually, not a distributed transaction across files.
Checksums detect accidental damage; snapshots are not cryptographically signed.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import tempfile
import time
from typing import Iterator

from shared.runtime_state import RuntimeStateLease, validate_runtime_state


_LOCK_NAME = re.compile(r"execution-[0-9a-f]{24}\.lock\Z")
_SQLITE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}
_SIDECARS = ("-wal", "-shm", "-journal")
_BACKUP_TIMEOUT = 30.0


def _path(path: Path) -> Path:
    result = Path(os.path.abspath(path))
    for part in (result, *result.parents):
        if part.is_symlink():
            raise ValueError("Runtime snapshot paths must not use symbolic links")
    return result


def _sensitive(name: str) -> bool:
    lower = name.lower()
    return (lower == ".env" or lower.startswith(".env.")
            or lower in {"id_rsa", "id_ed25519", "credentials", "credentials.json"}
            or lower.endswith((".pem", ".key", ".p12", ".pfx")))


def _inventory(root: Path) -> tuple[dict[str, Path], list[str]]:
    files: dict[str, Path] = {}
    directories: list[str] = []

    def fail(exc: OSError) -> None:
        raise exc

    for current, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        for name in sorted(dirs + names):
            path = Path(current) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError("Runtime snapshots reject symbolic links and special files")
            if _sensitive(name):
                raise ValueError("Runtime state contains a credential file; move it outside state")
            rel = path.relative_to(root).as_posix()
            if stat.S_ISDIR(mode):
                directories.append(rel)
            else:
                files[rel] = path
    return files, sorted(directories)


def _sqlite(path: Path) -> bool:
    if path.suffix in _SQLITE_SUFFIXES:
        return True
    with path.open("rb") as stream:
        return stream.read(16) == b"SQLite format 3\x00"


def _integrity(db: sqlite3.Connection) -> None:
    if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ValueError("SQLite integrity check failed")


def _sqlite_snapshot(source: Path, target: Path) -> None:
    deadline = time.monotonic() + _BACKUP_TIMEOUT

    def progress(status: int, remaining: int, total: int) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError("SQLite snapshot timed out; stop all state writers and retry")

    src = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=5)
    dst = sqlite3.connect(target)
    try:
        _integrity(src)
        src.backup(dst, pages=128, progress=progress, sleep=0.05)
        _integrity(dst)
        mode = src.execute("PRAGMA journal_mode").fetchone()[0]
        dst.execute("PRAGMA journal_mode=WAL" if mode == "wal" else "PRAGMA journal_mode=DELETE")
    finally:
        dst.close()
        src.close()


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sync_file(path: Path) -> None:
    os.chmod(path, 0o600)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _destination(destination: Path) -> Iterator[Path]:
    """Serialize cooperating tools and publish a complete directory by rename."""
    target = _path(destination)
    if not target.parent.is_dir():
        raise ValueError("Snapshot destination parent must already exist")
    lock = target.with_name(f".{target.name}.snapshot-lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = None
    try:
        if target.exists():
            raise FileExistsError("Snapshot destination already exists; overwrite is forbidden")
        tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        yield tmp
        if target.exists() or target.is_symlink():
            raise FileExistsError("Snapshot destination appeared during operation")
        _sync_directory(tmp)
        tmp.rename(target)
        tmp = None
        _sync_directory(target.parent)
    finally:
        os.close(fd)
        lock.unlink()
        if tmp is not None:
            shutil.rmtree(tmp)


def backup_state(state_dir: Path, destination: Path, scope: str) -> Path:
    """Create a new local snapshot; never upload or overwrite another snapshot."""
    source, target = _path(state_dir), _path(destination)
    if source == target or source in target.parents:
        raise ValueError("Snapshot destination must be outside the runtime directory")
    with RuntimeStateLease(source, scope):
        source_files, directories = _inventory(source)
        database_paths = {name for name, path in source_files.items() if _sqlite(path)
                          and not name.endswith(_SIDECARS)}
        files = {}
        for name, path in source_files.items():
            if _LOCK_NAME.fullmatch(path.name):
                continue
            if name.endswith(_SIDECARS):
                base = next(name[:-len(suffix)] for suffix in _SIDECARS if name.endswith(suffix))
                if base not in database_paths:
                    raise ValueError("Runtime state has an orphan database sidecar")
                continue
            files[name] = path
        with _destination(target) as stage:
            payload = stage / "state"
            payload.mkdir(mode=0o700)
            for name in directories:
                (payload / name).mkdir(mode=0o700)
            for name, source_file in sorted(files.items()):
                output = payload / name
                kind = "sqlite" if name in database_paths else "file"
                if kind == "sqlite":
                    _sqlite_snapshot(source_file, output)
                else:
                    shutil.copyfile(source_file, output)
                _sync_file(output)
            validate_runtime_state(payload, scope)
            for name in files:
                _sync_file(payload / name)
            entries = [{"path": name, "kind": "sqlite" if name in database_paths else "file",
                        "size": (payload / name).stat().st_size, "sha256": _digest(payload / name)}
                       for name in sorted(files)]
            identity = json.loads((payload / "runtime-state.json").read_text())
            manifest = {"version": 1, "scope": scope, "state_id": identity["state_id"],
                        "files": entries, "directories": directories}
            metadata = stage / "snapshot.json"
            metadata.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            _sync_file(metadata)
            for name in reversed(directories):
                _sync_directory(payload / name)
            _sync_directory(payload)
    return target


def _relative(name: object) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Invalid snapshot member path")
    rel = PurePosixPath(name)
    if rel.is_absolute() or any(part in {".", ".."} for part in rel.parts) or rel.as_posix() != name:
        raise ValueError("Unsafe snapshot member path")
    if any(_sensitive(part) for part in rel.parts):
        raise ValueError("Snapshot contains a credential file")
    return name


def _verify_snapshot(snapshot: Path, scope: str) -> tuple[dict, dict[str, Path]]:
    if not snapshot.is_dir():
        raise ValueError("Snapshot directory is missing")
    outer, outer_dirs = _inventory(snapshot)
    if set(outer) - {"snapshot.json"} != {name for name in outer if name.startswith("state/")}:
        raise ValueError("Unexpected snapshot root contents")
    if "snapshot.json" not in outer or "state" not in outer_dirs:
        raise ValueError("Snapshot metadata or payload is missing")
    meta = json.loads((snapshot / "snapshot.json").read_text())
    if (not isinstance(meta, dict) or type(meta.get("version")) is not int
            or meta.get("version") != 1 or meta.get("scope") != scope
            or not isinstance(meta.get("state_id"), str)
            or not isinstance(meta.get("files"), list) or not isinstance(meta.get("directories"), list)):
        raise ValueError("Snapshot identity or format does not match")
    directories = [_relative(name) for name in meta["directories"]]
    actual_files, actual_dirs = _inventory(snapshot / "state")
    if set(outer_dirs) != {"state", *("state/" + name for name in actual_dirs)}:
        raise ValueError("Unexpected snapshot root directories")
    if len(set(directories)) != len(directories) or set(directories) != set(actual_dirs):
        raise ValueError("Snapshot directory inventory does not match")
    expected = set()
    for entry in meta["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid snapshot file entry")
        name = _relative(entry.get("path"))
        if name in expected or name not in actual_files:
            raise ValueError("Snapshot has duplicate or missing files")
        expected.add(name)
        path = actual_files[name]
        if (entry.get("kind") not in {"file", "sqlite"}
                or (entry.get("kind") == "sqlite") != _sqlite(path)
                or type(entry.get("size")) is not int or entry["size"] != path.stat().st_size
                or entry.get("sha256") != _digest(path)):
            raise ValueError("Snapshot file checksum or format does not match")
        if entry["kind"] == "sqlite":
            db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
            try:
                _integrity(db)
            finally:
                db.close()
    if expected != set(actual_files) or not {"orders.sqlite", "drawdown.sqlite", "runtime-state.json"} <= expected:
        raise ValueError("Snapshot file inventory is incomplete")
    identity = json.loads(actual_files["runtime-state.json"].read_text())
    if (not isinstance(identity, dict) or identity.get("state_id") != meta["state_id"]
            or identity.get("scope") != scope):
        raise ValueError("Snapshot runtime identity does not match")
    return meta, actual_files


def restore_state(snapshot_dir: Path, destination: Path, scope: str) -> Path:
    """Validate and restore to a new directory, retaining the original identity."""
    source, target = _path(snapshot_dir), _path(destination)
    if source == target or source in target.parents:
        raise ValueError("Restore destination must be outside the snapshot")
    meta, files = _verify_snapshot(source, scope)
    with _destination(target) as stage:
        for name in meta["directories"]:
            (stage / name).mkdir(mode=0o700)
        for name, path in files.items():
            output = stage / name
            shutil.copyfile(path, output)
            _sync_file(output)
        # Recheck copied bytes, not only their source, before publishing state.
        for entry in meta["files"]:
            if _digest(stage / entry["path"]) != entry["sha256"]:
                raise ValueError("Snapshot changed while restoring")
        validate_runtime_state(stage, scope)
        # Snapshot readiness is historical evidence, never startup authority.
        # Preserve the baseline and every cash-flow fact, but force a new read.
        db = sqlite3.connect(stage / "orders.sqlite")
        try:
            has_wallet = db.execute("SELECT 1 FROM sqlite_master WHERE name='wallet_metadata'").fetchone()
        finally:
            db.close()
        if has_wallet:
            from shared.account_ledger import AccountLedger
            identity = json.loads((stage / "runtime-state.json").read_text())
            AccountLedger(stage / "orders.sqlite", scope=scope, state_id=identity["state_id"]).invalidate(
                "restored_account_requires_reconciliation"
            )
            _sync_file(stage / "orders.sqlite")
        for name in reversed(meta["directories"]):
            _sync_directory(stage / name)
    return target
