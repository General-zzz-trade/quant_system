"""The operator kill switch of the target runtime (migration step 13c; ADR-0005 5b and 6b).

    python -m trading.infrastructure.kill --state-dir data/runtime/binance_target activate --reason "..." [--flatten]
    python -m trading.infrastructure.kill --state-dir data/runtime/binance_target clear
    python -m trading.infrastructure.kill --state-dir data/runtime/binance_target status

The state directory is required (there is no default) and must be initialized, so the command writes exactly
the file the runtime reads. The runtime itself never writes it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from trading.adapters.files.kill_switch import KillFile
from trading.infrastructure.state_dir import is_managed, kill_file_path


def main(argv: list[str] | None = None, *, stdout: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("command", choices=["activate", "clear", "status"])
    parser.add_argument("--reason", default="operator kill switch")
    parser.add_argument("--flatten", action="store_true", help="also close open positions")
    args = parser.parse_args(argv)
    if not is_managed(args.state_dir):
        print(f"{args.state_dir} is not initialized; nothing written", file=stdout)
        return 2
    kill = KillFile(kill_file_path(args.state_dir), missing_is_active=True)
    if args.command == "activate":
        kill.write(active=True, reason=args.reason, flatten=args.flatten)
    elif args.command == "clear":
        kill.write(active=False)
    print(json.dumps(kill.state().__dict__, ensure_ascii=False), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
