"""The operator kill switch as a file (moved from shared/kill_file.py for the target runtime, step 13c).

- ``{"active": false}``  -> inactive
- ``{"active": true, "flatten": ..., "reason": ...}`` -> new exposure blocked; with ``flatten`` the runtime
  also closes positions
- unreadable or malformed -> active without flatten (blocks entries)
- missing -> active without flatten when ``missing_is_active`` (a managed state directory, ADR-0005 5b);
  inactive otherwise

The runtime only reads it. ``write`` is for the kill command and for initialising a state directory.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from trading.domain.risk.kill import KillState


class KillFile:
    def __init__(self, path: Path, *, missing_is_active: bool) -> None:
        self.path, self.missing_is_active = Path(path), missing_is_active

    def state(self) -> KillState:
        try:
            if not self.path.exists():
                if self.missing_is_active:
                    return KillState(True, False, "kill file missing; treated as active")
                return KillState(False, False, "")
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("active"), bool):
                raise ValueError("kill file must be an object with a boolean 'active'")
            if not data["active"]:
                return KillState(False, False, "")
            return KillState(True, data.get("flatten") is True, str(data.get("reason") or "operator kill switch"))
        except Exception as exc:
            return KillState(True, False, f"kill file unreadable ({type(exc).__name__}); treated as active")

    def allow_order(self, *, symbol: str) -> tuple[bool, str]:
        state = self.state()
        return (not state.active), state.reason

    def write(self, *, active: bool, reason: str = "", flatten: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"active": active, "flatten": flatten, "reason": reason,
                                   "updated_at": time.time()}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


__all__ = ["KillFile"]
