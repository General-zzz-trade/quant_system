"""Offline runtime migration; never connects to an exchange or overwrites state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.runtime_migration import inspect_migration, migrate_runtime_state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("migrate", "inspect"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args(argv)
    if args.action == "migrate":
        if args.source_dir is None:
            parser.error("migrate requires --source-dir")
        result = migrate_runtime_state(args.source_dir, args.target_dir, args.scope)
        payload = {"target": result.target, "source_kind": result.source_kind,
                   "scope": result.scope, "state_id": result.state_id,
                   "execution_ready": result.execution_ready}
    else:
        payload = inspect_migration(args.target_dir, args.scope)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
