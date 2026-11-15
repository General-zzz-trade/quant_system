"""The managed state directory of the execution runtime (migration step 13c; ADR-0007 1c, ADR-0005 5b).

    python -m trading.infrastructure.state_dir --state-dir data/runtime/binance_target init
    python -m trading.infrastructure.state_dir --state-dir data/runtime/binance_target check

A person initializes the directory once: a manifest naming the account scope, every store, and an inactive
kill file. Execution mode then starts only if the manifest exists, names the same scope, and every store and
the kill file are there and readable; nothing is created on the way. Changing the directory or the scope is
changing the account identity: initialize a new directory. Paper mode does not use this.
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
from trading.adapters.sqlite.check import database_readable
from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.adapters.sqlite.limit_store import SqliteLimitStore
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.infrastructure.entry_pause import PAUSES_FILE
from trading.infrastructure.settings import execution_scope, risk_limits, testnet_from_env
from trading.infrastructure.target import DRAWDOWN_FILE, LIMITS_FILE, ORDERS_FILE

MANIFEST = "manifest.json"
KILL_FILE = "kill_switch.json"
STORES = (ORDERS_FILE, DRAWDOWN_FILE, LIMITS_FILE, PAUSES_FILE)


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
    problems = []
    for name in STORES:
        path = state_dir / name
        if not path.is_file():
            problems.append(f"{name} is missing")
            continue
        if not database_readable(path):
            problems.append(f"{name} is unreadable")
    kill = kill_file_path(state_dir)
    if not kill.is_file():
        problems.append(f"{KILL_FILE} is missing")
    elif "unreadable" in KillFile(kill, missing_is_active=True).state().reason:
        problems.append(f"{KILL_FILE} is unreadable")
    if problems:
        raise RuntimeError(f"state directory {state_dir} is not usable: {'; '.join(problems)}")


def initialize(state_dir: Path, *, scope: str, testnet: bool, env: Mapping[str, str]) -> bool:
    """Create the stores, an inactive kill file and, last, the manifest. False if already initialized."""
    state_dir = Path(state_dir)
    if is_managed(state_dir):
        return False
    state_dir.mkdir(parents=True, exist_ok=True)
    limits = risk_limits(env)
    SqliteOrderJournal(state_dir / ORDERS_FILE, scope)
    SqliteDrawdownGuard(state_dir / DRAWDOWN_FILE, scope, daily_pct=limits.daily_drawdown_pct,
                        weekly_pct=limits.weekly_drawdown_pct)
    SqliteLimitStore(state_dir / LIMITS_FILE, scope).load("initialized")
    SqliteEntryPauses(state_dir / PAUSES_FILE).reason("initialized")
    if not kill_file_path(state_dir).exists():
        KillFile(kill_file_path(state_dir), missing_is_active=True).write(active=False, reason="initialized")
    manifest = {"schema": 1, "scope": scope, "testnet": testnet, "stores": list(STORES), "kill_file": KILL_FILE,
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
    env = os.environ if env is None else env
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


__all__ = ["KILL_FILE", "MANIFEST", "STORES", "initialize", "is_managed", "kill_file_path", "main", "verify"]
