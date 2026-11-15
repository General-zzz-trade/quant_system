"""The managed state directory of the execution runtime (migration step 13c; ADR-0007 1c, ADR-0005 5b).

    python -m trading.infrastructure.state_dir --state-dir data/runtime/binance_target init
    python -m trading.infrastructure.state_dir --state-dir data/runtime/binance_target check

A person initializes the directory once: a manifest naming the account scope, the one state database
``state.sqlite`` with every table at its version (ADR-0009), and an inactive kill file. Execution mode then
starts only if the manifest exists, names the same scope, the database has the right version and every table,
the old five-file layout is absent and the kill file is readable; no database, kill file or manifest is created
on the way (reading a WAL database may leave SQLite's -shm/-wal files next to it). Changing the directory or
the scope is changing the account identity: initialize a new directory. Paper mode does not use this.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from trading.adapters.files.kill_switch import KillFile
from trading.adapters.sqlite import schema
from trading.adapters.sqlite.schema import STATE_DB
from trading.infrastructure.config import execution_scope, risk_limits, testnet_from_env

MANIFEST = "manifest.json"
KILL_FILE = "kill_switch.json"
STORES = (STATE_DB,)
# The five-file layout before ADR-0009. Its presence refuses a start: initialize a new directory (ADR-0009 5).
OLD_LAYOUT = ("orders.sqlite", "drawdown.sqlite", "order_limits.sqlite", "entry_pauses.sqlite")


def kill_file_path(state_dir: Path) -> Path:
    """The one place the kill file's path comes from, for the runtime and the kill command alike."""
    return Path(state_dir) / KILL_FILE


def is_managed(state_dir: Path) -> bool:
    return (Path(state_dir) / MANIFEST).is_file()


def verify(state_dir: Path, scope: str) -> None:
    """Raise unless ``state_dir`` was initialized for ``scope`` and every store and the kill file are readable."""
    state_dir = Path(state_dir)
    hint = f"; initialize it with: python -m trading.infrastructure.state_dir --state-dir {state_dir} init"
    try:
        manifest = json.loads((state_dir / MANIFEST).read_text())
    except FileNotFoundError:
        raise RuntimeError(f"state directory {state_dir} is not initialized: no manifest{hint}") from None
    except Exception as exc:
        raise RuntimeError(f"state directory {state_dir}: manifest unreadable ({type(exc).__name__})") from None
    if manifest.get("scope") != scope:
        raise RuntimeError(f"state directory {state_dir} belongs to scope {manifest.get('scope')}, not {scope}")
    if old := [name for name in OLD_LAYOUT if (state_dir / name).exists()]:
        raise RuntimeError(f"state directory {state_dir} has the old state layout ({', '.join(old)}); "
                           "initialize a new directory: python -m trading.infrastructure.state_dir "
                           "--state-dir <new directory> init")
    # Emptied (0 bytes), replaced or of another version: its state - a drawdown latch, today's counts, the
    # orders - is gone or unknown (audit after M3; ADR-0009).
    problems = schema.problems(state_dir / STATE_DB)
    kill = kill_file_path(state_dir)
    if not kill.is_file():
        problems.append(f"{KILL_FILE} is missing")
    elif "unreadable" in KillFile(kill, missing_is_active=True).state().reason:
        problems.append(f"{KILL_FILE} is unreadable")
    if problems:
        raise RuntimeError(f"state directory {state_dir} is not usable: {'; '.join(problems)}")


def initialize(state_dir: Path, *, scope: str, testnet: bool, env: Mapping[str, str] | None) -> bool:
    """Create the state database, an inactive kill file and, last, the manifest. False if already initialized."""
    state_dir = Path(state_dir)
    if is_managed(state_dir):
        return False
    risk_limits(env)                                  # the drawdown ceiling is checked before anything exists
    state_dir.mkdir(parents=True, exist_ok=True)
    schema.create(state_dir / STATE_DB)               # refuses a database left by an interrupted init
    if not kill_file_path(state_dir).exists():
        KillFile(kill_file_path(state_dir), missing_is_active=True).write(active=False, reason="initialized")
    manifest = {"schema": 1, "scope": scope, "testnet": testnet, "stores": list(STORES),
                "state_db_version": schema.VERSION, "kill_file": KILL_FILE,
                "created_at": datetime.fromtimestamp(time.time(), timezone.utc).isoformat(timespec="seconds")}
    tmp = state_dir / f"{MANIFEST}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, state_dir / MANIFEST)
    return True


def main(argv: list[str] | None = None, *, stdout: TextIO = sys.stdout, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("command", choices=["init", "check"])
    args = parser.parse_args(argv)
    testnet = testnet_from_env(env)
    scope = execution_scope(testnet=testnet, env=env)          # raises before anything is created
    if args.command == "check":
        verify(args.state_dir, scope)
        print(f"{args.state_dir}: initialized for {scope}", file=stdout)
        return 0
    if not initialize(args.state_dir, scope=scope, testnet=testnet, env=env):
        print(f"{args.state_dir} is already initialized; nothing changed", file=stdout)
        return 1
    print(f"{args.state_dir}: initialized for {scope}", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["KILL_FILE", "MANIFEST", "OLD_LAYOUT", "STORES", "initialize", "is_managed", "kill_file_path", "main",
           "verify"]
