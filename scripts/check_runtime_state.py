"""Initialize or validate a local persistent execution state; never send orders."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from shared.runtime_state import RuntimeStateLease, initialize_runtime_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "check"))
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    try:
        if args.action == "init":
            initialize_runtime_state(args.state_dir, args.scope)
            print("Initialized empty runtime state; account reconciliation is still required before execution")
        else:
            with RuntimeStateLease(args.state_dir, args.scope):
                print("Runtime state validation passed; no execution or network access performed")
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, f"Runtime state blocked: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
