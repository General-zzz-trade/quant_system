"""Coordinated local snapshots for one complete portfolio risk group.

The risk database and every registered account state are captured while all
cooperating writers are excluded. Restores publish only into a new root. This
module never contacts an exchange, resets a baseline or rewrites a state ID.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Mapping

from shared.account_ledger import AccountLedger
from shared.portfolio_reservations import PortfolioReservationStore
from shared.runtime_backup import (
    _LOCK_NAME,
    _SIDECARS,
    _destination,
    _digest,
    _integrity,
    _inventory,
    _path,
    _relative,
    _sqlite,
    _sqlite_snapshot,
    _sync_directory,
    _sync_file,
)
from shared.runtime_state import MANIFEST_NAME, RuntimeStateLease, _json, validate_runtime_state

MANIFEST_NAME_GROUP = "risk-group.json"
PAYLOAD_NAME = "payload"
RISK_NAME = "risk.sqlite"


def _row_counts(path: Path) -> dict[str, int]:
    db = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        _integrity(db)
        result: dict[str, int] = {}
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        for row in rows:
            name = row[0]
            quoted = '"' + name.replace('"', '""') + '"'
            result[name] = int(db.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
        return result
    finally:
        db.close()


def _account_files(root: Path) -> tuple[dict[str, Path], list[str], set[str]]:
    source_files, directories = _inventory(root)
    databases = {
        name for name, path in source_files.items()
        if _sqlite(path) and not name.endswith(_SIDECARS)
    }
    files: dict[str, Path] = {}
    for name, path in source_files.items():
        if _LOCK_NAME.fullmatch(path.name):
            continue
        if name.endswith(_SIDECARS):
            base = next(name[:-len(suffix)] for suffix in _SIDECARS if name.endswith(suffix))
            if base not in databases:
                raise ValueError("Account state has an orphan SQLite sidecar")
            continue
        files[name] = path
    return files, directories, databases


def _snapshot_files(source: Path, target: Path) -> tuple[list[dict[str, Any]], list[str]]:
    files, directories, databases = _account_files(source)
    target.mkdir(mode=0o700)
    for name in directories:
        (target / name).mkdir(mode=0o700)
    entries = []
    for name, source_file in sorted(files.items()):
        output = target / name
        kind = "sqlite" if name in databases else "file"
        if kind == "sqlite":
            _sqlite_snapshot(source_file, output)
        else:
            shutil.copyfile(source_file, output)
        _sync_file(output)
        entries.append({
            "path": name,
            "kind": kind,
            "size": output.stat().st_size,
            "sha256": _digest(output),
            "rows": _row_counts(output) if kind == "sqlite" else None,
        })
    for name in reversed(directories):
        _sync_directory(target / name)
    _sync_directory(target)
    return entries, directories


def _registration_rows(store: PortfolioReservationStore) -> list[dict[str, str]]:
    return [{
        "scope": row.scope, "venue": row.venue,
        "environment": row.environment, "state_id": row.state_id,
    } for row in sorted(store.registrations, key=lambda item: item.scope)]


def _invalidate_wallet(state: Path, scope: str, state_id: str) -> None:
    """Retain ledger facts but require a fresh account observation after restore."""
    path = state / "orders.sqlite"
    db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        has_wallet = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='wallet_metadata'"
        ).fetchone()
    finally:
        db.close()
    if has_wallet:
        ledger = AccountLedger(path, scope=scope, state_id=state_id)
        ledger.invalidate("restored_group_requires_reconciliation")
        _sync_file(path)


def backup_risk_group(
    risk_path: Path,
    destination: Path,
    *,
    group: str,
    accounts: Mapping[str, Path],
) -> Path:
    """Snapshot the risk DB and every exactly registered account while stopped."""
    risk, target = _path(risk_path), _path(destination)
    if not risk.is_file():
        raise ValueError("Portfolio risk state is missing")
    normalized = {scope: _path(path) for scope, path in accounts.items()}
    if len(normalized) != len(accounts) or len(set(normalized.values())) != len(normalized):
        raise ValueError("Account state registry is duplicate or invalid")
    if any(first in second.parents for first in normalized.values() for second in normalized.values()
           if first != second):
        raise ValueError("Account state directories must not overlap")
    if any(risk == state / RISK_NAME or state in risk.parents for state in normalized.values()):
        raise ValueError("Shared risk state must remain outside account state directories")
    if any(state == target or state in target.parents for state in normalized.values()):
        raise ValueError("Snapshot destination must be outside all account state directories")

    risk_db = sqlite3.connect(f"{risk.as_uri()}?mode=rw", uri=True, timeout=1)
    try:
        risk_db.execute("PRAGMA synchronous=FULL")
        risk_db.execute("BEGIN IMMEDIATE")
        _integrity(risk_db)
        store = PortfolioReservationStore(risk, group=group)
        expected_scopes = {row.scope for row in store.registrations}
        if set(normalized) != expected_scopes:
            raise ValueError("Backup requires every and only registered account scope")
        registrations = _registration_rows(store)
        by_scope = {row["scope"]: row for row in registrations}
        with ExitStack() as stack:
            for scope in sorted(normalized):
                state_dir = normalized[scope]
                stack.enter_context(RuntimeStateLease(state_dir, scope))
                identity = _json((state_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
                if identity.get("state_id") != by_scope[scope]["state_id"]:
                    raise ValueError("Account runtime identity differs from the risk registry")
            # Execution leases exclude cooperating processes. Explicit SQLite
            # writer locks also keep committed DB rows fixed across the bundle,
            # including extra checkpoint databases maintained by other tools.
            for scope in sorted(normalized):
                files, _, databases = _account_files(normalized[scope])
                for name in sorted(databases):
                    connection = sqlite3.connect(f"{files[name].as_uri()}?mode=rw", uri=True, timeout=1)
                    stack.callback(connection.close)
                    connection.execute("BEGIN IMMEDIATE")
            with _destination(target) as stage:
                payload = stage / PAYLOAD_NAME
                payload.mkdir(mode=0o700)
                risk_output = payload / RISK_NAME
                _sqlite_snapshot(risk, risk_output)
                _sync_file(risk_output)
                account_root = payload / "accounts"
                account_root.mkdir(mode=0o700)
                manifest_accounts = []
                for index, scope in enumerate(sorted(normalized)):
                    slot = f"account-{index:03d}"
                    account_target = account_root / slot
                    entries, directories = _snapshot_files(normalized[scope], account_target)
                    validate_runtime_state(account_target, scope, require_writable=False)
                    manifest_accounts.append({
                        **by_scope[scope], "slot": slot,
                        "files": entries, "directories": directories,
                    })
                risk_entry = {
                    "path": RISK_NAME, "kind": "sqlite", "size": risk_output.stat().st_size,
                    "sha256": _digest(risk_output), "rows": _row_counts(risk_output),
                }
                manifest = {
                    "version": 1, "group": group, "complete_group": True,
                    "risk": risk_entry, "registrations": registrations,
                    "accounts": manifest_accounts,
                }
                metadata = stage / MANIFEST_NAME_GROUP
                metadata.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                _sync_file(metadata)
                _sync_directory(account_root)
                _sync_directory(payload)
        risk_db.rollback()
    finally:
        risk_db.close()
    return target


def _verify_entry(root: Path, entry: dict[str, Any]) -> Path:
    if not isinstance(entry, dict):
        raise ValueError("Invalid risk-group file entry")
    name = _relative(entry.get("path"))
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise ValueError("Risk-group file is missing or unsafe")
    kind = entry.get("kind")
    if (kind not in {"file", "sqlite"}
            or (kind == "sqlite") != _sqlite(path)
            or type(entry.get("size")) is not int or entry["size"] != path.stat().st_size
            or entry.get("sha256") != _digest(path)):
        raise ValueError("Risk-group file checksum or type differs")
    if kind == "sqlite" and entry.get("rows") != _row_counts(path):
        raise ValueError("Risk-group database row inventory differs")
    if kind == "file" and entry.get("rows") is not None:
        raise ValueError("Non-database risk-group file has a row inventory")
    return path


def _verify_bundle(snapshot: Path, group: str) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    root = _path(snapshot)
    if not root.is_dir():
        raise ValueError("Risk-group snapshot is missing")
    all_files, all_directories = _inventory(root)
    if MANIFEST_NAME_GROUP not in all_files:
        raise ValueError("Risk-group manifest is missing")
    meta = _json(all_files[MANIFEST_NAME_GROUP].read_text(encoding="utf-8"))
    if (type(meta.get("version")) is not int or meta.get("version") != 1 or meta.get("group") != group
            or meta.get("complete_group") is not True
            or not isinstance(meta.get("registrations"), list)
            or not isinstance(meta.get("accounts"), list)):
        raise ValueError("Risk-group manifest identity or version differs")
    payload = root / PAYLOAD_NAME
    risk_entry = meta.get("risk")
    if not isinstance(risk_entry, dict) or risk_entry.get("path") != RISK_NAME:
        raise ValueError("Risk-group database entry is invalid")
    risk = _verify_entry(payload, risk_entry)
    registrations = meta["registrations"]
    if len(registrations) != len(meta["accounts"]):
        raise ValueError("Risk-group account coverage is incomplete")
    reg_by_scope: dict[str, dict[str, Any]] = {}
    for registration in registrations:
        if (not isinstance(registration, dict)
                or set(registration) != {"scope", "venue", "environment", "state_id"}
                or not all(isinstance(value, str) and value for value in registration.values())
                or registration["scope"] in reg_by_scope):
            raise ValueError("Risk-group registration is invalid or duplicated")
        reg_by_scope[registration["scope"]] = registration
    account_paths: dict[str, Path] = {}
    expected_files = {MANIFEST_NAME_GROUP, f"{PAYLOAD_NAME}/{RISK_NAME}"}
    expected_directories = {PAYLOAD_NAME, f"{PAYLOAD_NAME}/accounts"}
    seen_slots = set()
    for account in meta["accounts"]:
        if not isinstance(account, dict):
            raise ValueError("Risk-group account entry is invalid")
        scope, slot = account.get("scope"), account.get("slot")
        if (not isinstance(scope, str) or scope not in reg_by_scope or scope in account_paths
                or not isinstance(slot, str) or slot in seen_slots
                or slot != f"account-{len(seen_slots):03d}"
                or any(account.get(name) != reg_by_scope[scope][name]
                       for name in ("venue", "environment", "state_id"))
                or not isinstance(account.get("files"), list)
                or not isinstance(account.get("directories"), list)):
            raise ValueError("Risk-group account identity or slot differs")
        seen_slots.add(slot)
        account_root = payload / "accounts" / slot
        files, directories = _inventory(account_root)
        declared_dirs = [_relative(name) for name in account["directories"]]
        if len(set(declared_dirs)) != len(declared_dirs) or set(declared_dirs) != set(directories):
            raise ValueError("Risk-group account directory inventory differs")
        declared_files = set()
        for entry in account["files"]:
            path = _verify_entry(account_root, entry)
            name = entry["path"]
            if name in declared_files or files.get(name) != path:
                raise ValueError("Risk-group account file inventory is duplicated")
            declared_files.add(name)
            expected_files.add(f"{PAYLOAD_NAME}/accounts/{slot}/{name}")
        if declared_files != set(files):
            raise ValueError("Risk-group account file inventory is incomplete")
        validate_runtime_state(account_root, scope, require_writable=False)
        identity = _json((account_root / MANIFEST_NAME).read_text(encoding="utf-8"))
        if identity.get("state_id") != account["state_id"]:
            raise ValueError("Risk-group account runtime identity differs")
        account_paths[scope] = account_root
        expected_directories.add(f"{PAYLOAD_NAME}/accounts/{slot}")
        expected_directories.update(
            f"{PAYLOAD_NAME}/accounts/{slot}/{name}" for name in declared_dirs
        )
    if set(all_files) != expected_files or set(all_directories) != expected_directories:
        raise ValueError("Risk-group snapshot contains undeclared content")
    store = PortfolioReservationStore(risk, group=group)
    actual = _registration_rows(store)
    if actual != registrations or set(account_paths) != set(reg_by_scope):
        raise ValueError("Risk database registrations differ from the bundle")
    return meta, risk, account_paths


def restore_risk_group(snapshot: Path, destination: Path, *, group: str) -> Path:
    """Restore a verified complete group to a new operational directory tree."""
    source, target = _path(snapshot), _path(destination)
    if source == target or source in target.parents:
        raise ValueError("Restore destination must be outside the snapshot")
    meta, risk, account_paths = _verify_bundle(source, group)
    with _destination(target) as stage:
        shutil.copyfile(risk, stage / RISK_NAME)
        _sync_file(stage / RISK_NAME)
        _verify_entry(stage, meta["risk"])
        restored_accounts = stage / "accounts"
        restored_accounts.mkdir(mode=0o700)
        by_scope = {account["scope"]: account for account in meta["accounts"]}
        for scope, source_account in sorted(account_paths.items()):
            account = by_scope[scope]
            output_root = restored_accounts / account["slot"]
            output_root.mkdir(mode=0o700)
            for name in account["directories"]:
                (output_root / name).mkdir(mode=0o700)
            for entry in account["files"]:
                output = output_root / entry["path"]
                shutil.copyfile(source_account / entry["path"], output)
                _sync_file(output)
            validate_runtime_state(output_root, scope, require_writable=False)
            for entry in account["files"]:
                _verify_entry(output_root, entry)
            _invalidate_wallet(output_root, scope, account["state_id"])
            for name in reversed(account["directories"]):
                _sync_directory(output_root / name)
            _sync_directory(output_root)
        restored = PortfolioReservationStore(stage / RISK_NAME, group=group)
        if _registration_rows(restored) != meta["registrations"]:
            raise ValueError("Restored risk registry differs from the snapshot")
        restored.invalidate("restored_group_requires_reconciliation")
        _sync_file(stage / RISK_NAME)
        receipt = stage / "risk-group-restore.json"
        receipt.write_text(json.dumps({
            "version": 1, "group": group,
            "snapshot_manifest_sha256": _digest(source / MANIFEST_NAME_GROUP),
            "registrations": meta["registrations"],
            "fresh_reconciliation_required": True,
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        _sync_file(receipt)
        _sync_directory(restored_accounts)
    return target
