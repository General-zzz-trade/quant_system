"""Manual tools for a person: settle a stuck order, align a ledger to the venue (migration step 12b).

    python -m trading.infrastructure.manual --state-dir data/runtime/binance_target orders
    python -m trading.infrastructure.manual --state-dir ... settle --intent <id> --reason "<why>"
    python -m trading.infrastructure.manual --state-dir ... position --symbol BTCUSDT --reason "<why>"

Approved by the tech lead (ADR-0004 item 2, 2026-11-14; ledger alignment 2026-11-15). Every command:

- refuses while a runtime of the same account identity runs on this host (the host lock, ADR-0016 point 5)
  or holds this directory's lease (stop it first), and never creates a state file;
- reads the venue first and prints what it found;
- needs a reason, and the order id or symbol typed back at the prompt - there is no flag that skips it;
- leaves a row in the order journal's ``manual_actions`` table.

``settle`` refuses an order that is still working at the venue, and never records a fill: if the venue
filled more than the journal holds, the gap is shown and the ledger is aligned with ``position``.
``position`` sets the strategy's ledger to the venue position; it is not a fill. It refuses while an order
of the symbol is open or a fill is not consumed yet. Credentials and the scope come from the same
environment variables as the runtime (BINANCE_TESTNET, RISK_SCOPE, the API keys of that environment).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

from trading.adapters.binance.adapter import BinanceAdapter
from trading.adapters.files.host_lock import HostLock
from trading.adapters.files.lease import ExecutionLease
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.adapters.sqlite.schema import STATE_DB
from trading.domain.venue import STOP_TRIGGERED
from trading.infrastructure.config import (account_identity, binance_config_from_env, execution_scope,
                                           testnet_from_env)

JOURNAL = STATE_DB      # the one state database (ADR-0009)
WORKING = frozenset({"new", "partially_filled"})
SETTLES_AS = {"filled": "filled", "canceled": "canceled", "expired": "expired", "rejected": "rejected"}


def _connect(testnet: bool, symbols: list[str], env: Mapping[str, str] | None) -> Any:
    account = BinanceAdapter(binance_config_from_env(testnet=testnet, env=env))
    if not account.connect(symbols=tuple(symbols)):
        raise RuntimeError("Binance account connection failed")
    return account


def _recorded(journal: SqliteOrderJournal, intent_id: str) -> Decimal:
    return sum((Decimal(f["qty"]) for f in journal.intent_fills(intent_id)), Decimal(0))


def _venue_order(account: Any, row: dict[str, Any]) -> Any:
    return account.get_order(row["symbol"], order_id=row["order_id"] or "", client_order_id=row["client_id"])


def _confirm(stdin: TextIO, stdout: TextIO, expected: str, prompt: str) -> bool:
    print(f"{prompt} Type {expected} to confirm: ", end="", file=stdout)
    return stdin.readline().strip() == expected


def _orders(journal: SqliteOrderJournal, account: Any, stdout: TextIO) -> int:
    listed = []
    for row in journal.orders(unresolved=True):
        try:
            state = _venue_order(account, row)
            venue = {"state": "not found"} if state is None else {"state": state.state,
                                                                    "filled_qty": str(state.filled_qty)}
        except Exception as exc:
            venue = {"state": f"lookup failed: {type(exc).__name__}"}
        listed.append({"intent_id": row["intent_id"], "owner": row["owner"], "symbol": row["symbol"],
                       "side": row["side"], "qty": row["qty"], "status": row["status"], "order_id": row["order_id"],
                       "created_at": row["created_at"], "recorded_qty": str(_recorded(journal, row["intent_id"])),
                       "venue": venue})
    print(json.dumps(listed, indent=2), file=stdout)
    return 0


def _settle(journal: SqliteOrderJournal, account: Any, intent_id: str, reason: str, stdin: TextIO,
            stdout: TextIO) -> int:
    row = next((r for r in journal.orders(unresolved=True) if r["intent_id"] == intent_id), None)
    if row is None:
        print(f"no open order {intent_id}", file=stdout)
        return 0
    recorded = _recorded(journal, intent_id)
    try:
        state = _venue_order(account, row)
    except Exception as exc:
        print(f"the venue could not be asked ({type(exc).__name__}); nothing changed", file=stdout)
        return 1
    if state is None:
        if recorded:
            print("the venue has no such order but the journal holds fills for it; nothing changed", file=stdout)
            return 1
        status, venue = "not_found", {"state": "not found", "recorded_qty": str(recorded)}
    elif state.state in WORKING:
        print(f"{intent_id} is still working at the venue ({state.state}); cancel it there first", file=stdout)
        return 1
    elif state.state in SETTLES_AS:
        status = SETTLES_AS[state.state]
        venue = {"state": state.state, "filled_qty": str(state.filled_qty), "recorded_qty": str(recorded)}
        if Decimal(str(state.filled_qty)) != recorded:
            print(f"WARNING: the venue filled {state.filled_qty}, the journal recorded {recorded}. The difference "
                  f"is not in the ledger; afterwards align it with: position --symbol {row['symbol']}", file=stdout)
    else:
        print(f"unexpected venue state {state.state!r}; nothing changed", file=stdout)
        return 1
    print(f"{intent_id} ({row['symbol']} {row['side']} {row['qty']}): journal {row['status']}, venue "
          f"{venue['state']}", file=stdout)
    if not _confirm(stdin, stdout, intent_id, f"Settle it as {status}?"):
        print("not settled", file=stdout)
        return 1
    journal.settle_manually(intent_id, status, venue=venue, reason=reason, at=time.time())
    print(f"settled {intent_id} as {status}", file=stdout)
    return 0


def _unseen_trigger(journal: SqliteOrderJournal, account: Any, owner: str, symbol: str) -> str | None:
    """Why a protective stop of the symbol may still change the ledger, or None. A stop that fired while the
    runtime was stopped is adopted at the next start and its fill lands on the ledger: aligning the ledger
    first would count that fill twice (audit after step 12b, finding 3)."""
    for row in journal.stop_rows(owner, symbol, live_only=True):
        if row["status"] == "awaiting_order":
            return f"protective stop {row['client_algo_id']} has fired and its order is not known yet"
        try:
            state = account.get_algo_order(client_algo_id=row["client_algo_id"])
        except Exception as exc:
            return f"protective stop {row['client_algo_id']} could not be checked ({type(exc).__name__})"
        if state is not None and state.state == STOP_TRIGGERED:
            return f"protective stop {row['client_algo_id']} has fired and the runtime has not seen it"
    return None


def _position(journal: SqliteOrderJournal, account: Any, symbol: str, owner: str, reason: str, stdin: TextIO,
              stdout: TextIO) -> int:
    if any(r["symbol"] == symbol for r in journal.orders(unresolved=True)):
        print(f"{symbol} has open orders; settle the open orders first (orders, settle)", file=stdout)
        return 1
    positions = [p for p in account.get_positions(symbol, strict=True)
                 if p.symbol == symbol and Decimal(str(p.qty)) != 0]
    # After the venue read: a stop that fires before it shows in the position, one that fires after it is seen
    # here (audit after M3). Either way the ledger is not set to a position the stop's fill will change.
    unseen = _unseen_trigger(journal, account, owner, symbol)
    if unseen is not None:
        print(f"{unseen}; start the runtime once so it takes the stop's fill into the ledger, stop it, then "
              "check again; nothing changed", file=stdout)
        return 1
    if len(positions) > 1:
        print(f"{symbol} has more than one venue position; nothing changed", file=stdout)
        return 1
    qty = Decimal(str(positions[0].qty)) if positions else Decimal(0)
    entry = Decimal(str(positions[0].entry_price or 0)) if positions else Decimal(0)
    checkpoint = journal.checkpoint(owner)
    before = checkpoint["position"] if checkpoint else {"qty": "0", "entry_price": "0", "realized_pnl": "0",
                                                        "fees": {}}
    print(f"{owner}: ledger {before['qty']} @ {before['entry_price']}, venue {qty} @ {entry}", file=stdout)
    if Decimal(before["qty"]) == qty:
        print("already aligned; nothing changed", file=stdout)
        return 0
    if not _confirm(stdin, stdout, symbol, f"Set the ledger of {owner} to {qty} @ {entry}?"):
        print("not changed", file=stdout)
        return 1
    position = {"qty": str(qty), "entry_price": str(entry), "realized_pnl": before["realized_pnl"],
                "fees": before.get("fees", {})}
    try:
        journal.set_position_manually(owner, position, venue={"qty": str(qty), "entry_price": str(entry)},
                                      reason=reason, at=time.time())
    except ValueError as exc:
        print(f"{exc}; start the runtime once so it consumes them, stop it, then try again", file=stdout)
        return 1
    print(f"ledger of {owner} set to {qty} @ {entry}", file=stdout)
    return 0


def main(argv: list[str] | None = None, *, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout,
         env: Mapping[str, str] | None = None, account: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("orders", help="list open orders with what the venue says about each")
    settle = commands.add_parser("settle", help="settle one stuck order")
    settle.add_argument("--intent", required=True)
    settle.add_argument("--reason", required=True)
    position = commands.add_parser("position", help="set a strategy's ledger to the venue position")
    position.add_argument("--symbol", required=True)
    position.add_argument("--owner", help="default target.<symbol>")
    position.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    path = args.state_dir / JOURNAL
    if not args.state_dir.is_dir() or not path.is_file():
        print(f"no order journal at {path}", file=stdout)
        return 2
    if args.command != "orders" and not args.reason.strip():
        print("a reason is required", file=stdout)
        return 2
    testnet = testnet_from_env(env)
    scope = execution_scope(testnet=testnet, env=env)
    try:                       # the API key names the account (ADR-0016 point 5); the secret is not used here
        identity = account_identity(testnet=testnet, api_key=binance_config_from_env(testnet=testnet, env=env).api_key)
    except RuntimeError:
        print("the API key that names the account is not set (BINANCE_TESTNET_API_KEY or BINANCE_API_KEY)",
              file=stdout)
        return 2
    try:
        host = HostLock(identity, holder=f"manual {args.command} --state-dir {args.state_dir}")
    except RuntimeError as exc:
        print(f"the runtime holds this account's lease; stop the runtime first ({exc})", file=stdout)
        return 3
    try:
        lease = ExecutionLease(path, scope)
    except RuntimeError:
        host.close()
        print("the runtime holds this account's lease; stop the runtime first", file=stdout)
        return 3
    try:
        journal = SqliteOrderJournal(path, scope)
        if args.command == "position":
            symbols = [args.symbol]
        else:
            symbols = sorted({r["symbol"] for r in journal.orders(unresolved=True)}) or ["BTCUSDT"]
        account = account if account is not None else _connect(testnet, symbols, env)
        if args.command == "orders":
            return _orders(journal, account, stdout)
        if args.command == "settle":
            return _settle(journal, account, args.intent, args.reason.strip(), stdin, stdout)
        return _position(journal, account, args.symbol, args.owner or f"target.{args.symbol}", args.reason.strip(),
                         stdin, stdout)
    finally:
        lease.close()
        host.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
