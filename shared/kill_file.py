"""Operator kill switch backed by a file, usable without restarting the runner.

    python -m shared.kill_file activate --reason "manual stop" [--flatten]
    python -m shared.kill_file clear
    python -m shared.kill_file status

Semantics (fail-closed):
- no file                         -> inactive
- {"active": false}               -> inactive
- {"active": true, ...}           -> new exposure blocked; with "flatten": true
                                     the runner also closes open positions
- unreadable / malformed file     -> ACTIVE with flatten=false (blocks entries)

Writes are atomic (temp file + rename) so readers never see partial JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from trading.domain.risk.kill import KillState

DEFAULT_PATH = Path("data/runtime/binance/kill_switch.json")  # runner.binance_main state dir


class KillFile:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self.path = Path(path)

    def state(self) -> KillState:
        try:
            if not self.path.exists():
                return KillState(False, False, "")
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("active"), bool):
                raise ValueError("kill file must be an object with a boolean 'active'")
            if not data["active"]:
                return KillState(False, False, "")
            return KillState(True, data.get("flatten") is True, str(data.get("reason") or "operator kill switch"))
        except Exception as exc:
            return KillState(True, False, f"kill file unreadable ({type(exc).__name__}); treated as active")

    # CompositeRiskGate kill_switch protocol
    def allow_order(self, *, symbol: str) -> tuple[bool, str]:
        state = self.state()
        return (not state.active), state.reason

    def write(self, *, active: bool, reason: str = "", flatten: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"active": active, "flatten": flatten, "reason": reason,
                                   "updated_at": time.time()}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["activate", "clear", "status"])
    parser.add_argument("--reason", default="operator kill switch")
    parser.add_argument("--flatten", action="store_true", help="Also close open positions")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    kill = KillFile(args.path)
    if args.command == "activate":
        kill.write(active=True, reason=args.reason, flatten=args.flatten)
    elif args.command == "clear":
        kill.write(active=False)
    print(json.dumps(kill.state().__dict__, ensure_ascii=False))


if __name__ == "__main__":
    main()
