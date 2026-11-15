"""Release a paused symbol's entries - a person's command, never called by the runtime (migration step 12).

    python -m trading.infrastructure.entry_pause --state-dir data/runtime/binance_target status
    python -m trading.infrastructure.entry_pause --state-dir data/runtime/binance_target \\
        release --symbol BTCUSDT --reason "checked the venue position and the ledger"

The runtime pauses a symbol's entries when its protective stop triggers or is breached. Releasing needs a
reason, which is stored with the pause, and the symbol typed back at the prompt. There is no flag that skips
the prompt: a release is always a person's decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from trading.adapters.sqlite.entry_pause import SqliteEntryPauses

PAUSES_FILE = "entry_pauses.sqlite"


def main(argv: list[str] | None = None, *, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", type=Path, required=True, help="the runtime's state directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="list the symbols whose entries are paused")
    release = commands.add_parser("release", help="end a symbol's pause")
    release.add_argument("--symbol", required=True)
    release.add_argument("--reason", required=True, help="why it is safe to trade the symbol again")
    args = parser.parse_args(argv)
    if not args.state_dir.is_dir():
        print(f"no state directory at {args.state_dir}", file=stdout)
        return 2
    pauses = SqliteEntryPauses(args.state_dir / PAUSES_FILE)
    if args.command == "status":
        print(json.dumps(pauses.active(), indent=2), file=stdout)
        return 0
    if not args.reason.strip():
        print("a release needs a reason", file=stdout)
        return 2
    cause = pauses.reason(args.symbol)
    if cause is None:
        print(f"no pause for {args.symbol}", file=stdout)
        return 0
    print(f"Entries for {args.symbol} are paused: {cause}", file=stdout)
    print(f"Type {args.symbol} to release them: ", end="", file=stdout)
    if stdin.readline().strip() != args.symbol:
        print("not released", file=stdout)
        return 1
    pauses.release(args.symbol, args.reason)
    print(f"released {args.symbol} ({cause}); reason recorded: {args.reason.strip()}", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PAUSES_FILE", "main"]
