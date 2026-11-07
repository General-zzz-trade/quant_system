"""Local-only runtime snapshot CLI. Use ``python -m scripts.runtime_state_backup``."""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

from shared.runtime_backup import backup_state, restore_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup", help="Stop all writers, then create a new local snapshot")
    backup.add_argument("--state-dir", type=Path, required=True)
    backup.add_argument("--destination", type=Path, required=True)
    backup.add_argument("--scope", required=True)
    restore = commands.add_parser("restore", help="Validate and restore into a new directory")
    restore.add_argument("--snapshot", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--scope", required=True)
    args = parser.parse_args()
    try:
        if args.command == "backup":
            result = backup_state(args.state_dir, args.destination, args.scope)
        else:
            result = restore_state(args.snapshot, args.destination, args.scope)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, f"Runtime state {args.command} blocked: {exc}\n")
    print(f"Runtime state {args.command} complete: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
