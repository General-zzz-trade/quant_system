"""The managed state directory of the execution runtime (migration step 13c; ADR-0007 1c, ADR-0005 5b).

    python -m trading.infrastructure.state_dir --state-dir data/runtime/binance_target init
    python -m trading.infrastructure.state_dir --state-dir data/runtime/binance_target check
    python -m trading.infrastructure.state_dir --state-dir <dir> restore --snapshot <file> --reason "<why>"

A person initializes the directory once: a manifest naming the account scope, the one state database
``state.sqlite`` with every table at its version (ADR-0009), and an inactive kill file. ``init`` also creates the
host lock directory if it is missing (ADR-0016 point 5), before anything else. Execution mode then
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
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from trading.adapters.files.host_lock import HostLock, prepare_lock_dir
from trading.adapters.files.kill_switch import KillFile
from trading.adapters.files.lease import ExecutionLease
from trading.adapters.sqlite import schema
from trading.adapters.sqlite.backup import copy_database, enable_wal, record_restore, symbols_in
from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.adapters.sqlite.limit_store import SqliteLimitStore
from trading.adapters.sqlite.schema import STATE_DB
from trading.domain.risk.drawdown import periods
from trading.infrastructure.config import (account_identity, binance_config_from_env, execution_scope, risk_limits,
                                           testnet_from_env)

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
    if schema.holds_paper_state(state_dir / STATE_DB):
        problems.append(f"{STATE_DB} is a paper record's database (it holds a paper state), not an account's")
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


RESTORE_PAUSE = "restored from a backup: a person checks the venue position and the ledger, then releases"
RESTORE_SYMBOLS = ("BTCUSDT", "ETHUSDT")


def restore(state_dir: Path, snapshot: Path, *, scope: str, testnet: bool, env: Mapping[str, str] | None,
            reason: str, symbols: Iterable[str], stdin: TextIO, stdout: TextIO, now: float) -> int:
    """ADR-0016 point 6: put ``snapshot`` back, only when the state database is lost or broken, and only tighten.

    Refused unless the directory was initialized for ``scope``, its database fails the start-up check, the
    snapshot passes it, there is a reason, no runtime of the account runs (the host lock and the directory lease)
    and the person types the scope back. The copy is tightened before it takes the database's place: the current
    day and week latched, today's order count used up, every symbol paused. The kill file is never touched; the
    broken database is kept beside the new one; the restore is recorded in ``manual_actions``."""
    state_dir, snapshot = Path(state_dir), Path(snapshot)
    try:
        manifest = json.loads((state_dir / MANIFEST).read_text())
    except Exception:
        print(f"{state_dir} was never initialized: initialize a new directory instead of restoring", file=stdout)
        return 2
    if manifest.get("scope") != scope:
        print(f"{state_dir} belongs to scope {manifest.get('scope')}, not {scope}", file=stdout)
        return 2
    database = state_dir / STATE_DB
    problems = schema.problems(database)
    if not problems:
        print(f"{STATE_DB} is intact; a restore is only for a lost or broken state database", file=stdout)
        return 1
    if bad := schema.problems(snapshot):
        print(f"the snapshot {snapshot} is not usable: {'; '.join(bad)}", file=stdout)
        return 2
    if not reason.strip():
        print("a restore needs a reason", file=stdout)
        return 2
    limits = risk_limits(env)
    try:                       # the API key names the account (ADR-0016 point 5); the secret is not used here
        identity = account_identity(testnet=testnet, api_key=binance_config_from_env(testnet=testnet, env=env).api_key)
    except RuntimeError:
        print("the API key that names the account is not set", file=stdout)
        return 2
    try:
        host = HostLock(identity, holder=f"state_dir restore --state-dir {state_dir}")
    except RuntimeError as exc:
        print(f"a runtime of this account runs; stop the runtime first ({exc})", file=stdout)
        return 3
    try:
        lease = ExecutionLease(database, scope)
    except RuntimeError:
        host.close()
        print("a runtime holds this directory; stop the runtime first", file=stdout)
        return 3
    try:
        print(f"{STATE_DB} is not usable: {'; '.join(problems)}", file=stdout)
        print(f"Restore {snapshot.name}, latch today's and this week's drawdown, count today's orders as used up "
              f"and pause every symbol. Type {scope} to restore: ", end="", file=stdout)
        if stdin.readline().strip() != scope:
            print("not restored", file=stdout)
            return 1
        temporary = state_dir / f".{STATE_DB}.restoring"
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{temporary}{suffix}").unlink(missing_ok=True)
        copy_database(snapshot, temporary)
        paused = sorted(set(symbols) | symbols_in(temporary))
        SqliteDrawdownGuard(temporary, scope, daily_pct=limits.daily_drawdown_pct,
                            weekly_pct=limits.weekly_drawdown_pct).latch_current(now=now)
        day = periods(now)["day"]
        SqliteLimitStore(temporary, scope).use_up(day, orders=limits.max_daily_orders,
                                                  notional=limits.max_daily_notional)
        pauses = SqliteEntryPauses(temporary, clock=lambda: now)
        for symbol in paused:
            pauses.pause(symbol, RESTORE_PAUSE)
        record_restore(temporary, scope=scope, at=now, snapshot=snapshot.name, reason=reason.strip(),
                       after={"latched": periods(now), "orders_used_up": day, "paused": paused})
        enable_wal(temporary)
        if left := schema.problems(temporary):
            raise RuntimeError(f"the restored copy failed its check: {'; '.join(left)}")
        stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for suffix in ("", "-wal", "-shm"):
            old = Path(f"{database}{suffix}")
            if old.exists():
                os.replace(old, Path(f"{old}.broken-{stamp}"))
        os.replace(temporary, database)
        verify(state_dir, scope)
    finally:
        lease.close()
        host.close()
    print(f"restored {snapshot.name}; drawdown latched for {periods(now)}, today's orders used up, paused: "
          f"{', '.join(paused)}. The kill file was not touched. Release each symbol with: python -m "
          f"trading.infrastructure.entry_pause --state-dir {state_dir} release --symbol <symbol> --reason ...",
          file=stdout)
    return 0


def main(argv: list[str] | None = None, *, stdout: TextIO = sys.stdout, env: Mapping[str, str] | None = None,
         stdin: TextIO = sys.stdin, clock: Callable[[], float] = time.time) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="create the state database, the kill file and the manifest")
    commands.add_parser("check", help="the start-up check of the directory")
    restore_cmd = commands.add_parser("restore", help="put a snapshot back when the state database is lost")
    restore_cmd.add_argument("--snapshot", type=Path, required=True)
    restore_cmd.add_argument("--reason", required=True)
    restore_cmd.add_argument("--symbols", default=",".join(RESTORE_SYMBOLS),
                             help="symbols to pause besides those in the snapshot")
    args = parser.parse_args(argv)
    testnet = testnet_from_env(env)
    scope = execution_scope(testnet=testnet, env=env)          # raises before anything is created
    if args.command == "restore":
        return restore(args.state_dir, args.snapshot, scope=scope, testnet=testnet, env=env, reason=args.reason,
                       symbols=[s.strip() for s in args.symbols.split(",") if s.strip()], stdin=stdin,
                       stdout=stdout, now=clock())
    if args.command == "check":
        verify(args.state_dir, scope)
        print(f"{args.state_dir}: initialized for {scope}", file=stdout)
        return 0
    try:                                                       # first, so a failure leaves nothing behind
        locks = prepare_lock_dir()
    except RuntimeError as exc:
        print(f"not initialized: {exc}", file=stdout)
        return 2
    print(f"host lock directory {locks} is ready", file=stdout)
    if not initialize(args.state_dir, scope=scope, testnet=testnet, env=env):
        print(f"{args.state_dir} is already initialized; nothing changed", file=stdout)
        return 1
    print(f"{args.state_dir}: initialized for {scope}", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["KILL_FILE", "MANIFEST", "OLD_LAYOUT", "STORES", "initialize", "is_managed", "kill_file_path", "main",
           "restore", "verify"]
