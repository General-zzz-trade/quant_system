"""Local-only complete portfolio snapshots; stop every state writer first."""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

from shared.risk_group_backup import backup_risk_group, restore_risk_group
from shared.runtime_state import _json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("--risk-path", type=Path, required=True)
    backup.add_argument("--registry", type=Path, required=True,
                        help="JSON accounts array declaring scope and state_dir; no credentials")
    backup.add_argument("--destination", type=Path, required=True)
    backup.add_argument("--group", required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--snapshot", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--group", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "backup":
            registry = _json(args.registry.read_text(encoding="utf-8")).get("accounts")
            if not isinstance(registry, list) or not registry:
                raise ValueError("Registry requires an explicit accounts list")
            accounts = {}
            for item in registry:
                if (not isinstance(item, dict) or not isinstance(item.get("scope"), str)
                        or not item["scope"] or item["scope"] in accounts
                        or not isinstance(item.get("state_dir"), str) or not item["state_dir"]):
                    raise ValueError("Invalid or duplicated account registry entry")
                accounts[item["scope"]] = Path(item["state_dir"])
            result = backup_risk_group(args.risk_path, args.destination, group=args.group, accounts=accounts)
        else:
            result = restore_risk_group(args.snapshot, args.destination, group=args.group)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, f"Risk group {args.command} blocked: {exc}\n")
    print(f"Risk group {args.command} complete: {result}")
    if args.command == "restore":
        print("Fresh complete account and portfolio reconciliation is required before new exposure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
