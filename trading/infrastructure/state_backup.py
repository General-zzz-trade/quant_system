"""Hourly snapshots of an execution state directory's database (ADR-0016 point 6, S8).

    python -m trading.infrastructure.state_backup --state-dir <state dir> --backup-dir <backup dir>

Run by ``state-backup@.timer``. The snapshot is taken with SQLite's online backup while the runtime runs, written
under a temporary name, checked (it opens, has the right version and every table) and only then given its
name ``state-<UTC time>.sqlite``. Snapshots older than 14 days are deleted after a good backup only. The kill
file is never backed up: only the kill command writes it (K4). A failure exits non-zero and changes nothing that
exists; the unit's ``OnFailure=`` sends the alert. The runtime never depends on this.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TextIO

from trading.adapters.sqlite import schema
from trading.adapters.sqlite.backup import copy_database

KEEP = timedelta(days=14)
_NAME = re.compile(r"state-(\d{8}T\d{6}Z)\.sqlite")
_FORMAT = "%Y%m%dT%H%M%SZ"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def backup(state_dir: Path, backup_dir: Path, *, now: datetime) -> Path:
    """Write and check one snapshot; return its path. Raises RuntimeError and leaves nothing behind on failure."""
    state_dir, backup_dir = Path(state_dir).resolve(), Path(backup_dir).resolve()
    if backup_dir == state_dir or state_dir in backup_dir.parents:
        raise RuntimeError(f"the backup directory {backup_dir} must be outside the state directory {state_dir}")
    source = state_dir / schema.STATE_DB
    if problems := schema.problems(source):
        raise RuntimeError(f"not backed up: {'; '.join(problems)}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    final = backup_dir / f"state-{now.astimezone(timezone.utc):{_FORMAT}}.sqlite"
    if final.exists():
        raise RuntimeError(f"{final.name} exists already; not overwritten")
    temporary = backup_dir / f".{final.name}.tmp"
    try:
        temporary.unlink(missing_ok=True)
        copy_database(source, temporary)
        if problems := schema.problems(temporary):
            raise RuntimeError(f"the snapshot failed its check: {'; '.join(problems)}")
        os.replace(temporary, final)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return final


def prune(backup_dir: Path, *, now: datetime) -> list[Path]:
    """Delete snapshots whose name says they are older than 14 days; other files are left alone."""
    removed = []
    for path in sorted(Path(backup_dir).iterdir()):
        match = _NAME.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        taken = datetime.strptime(match.group(1), _FORMAT).replace(tzinfo=timezone.utc)
        if now - taken > KEEP:
            path.unlink()
            removed.append(path)
    return removed


def main(argv: list[str] | None = None, *, stdout: TextIO = sys.stdout,
         now: Callable[[], datetime] = _now) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    when = now()
    try:
        snapshot = backup(args.state_dir, args.backup_dir, now=when)
    except Exception as exc:
        print(f"state backup failed: {exc}", file=stdout)
        return 1
    removed = prune(args.backup_dir, now=when)
    print(f"state backup: {snapshot} written and checked; {len(removed)} older than 14 days deleted", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["KEEP", "backup", "main", "prune"]
