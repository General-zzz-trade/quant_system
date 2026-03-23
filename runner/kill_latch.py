"""Persistent kill latch for active host services.

Prevents operator restarts from silently bypassing a previously-triggered
drawdown/risk kill on the same Bybit account scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from execution.adapters.bybit.config import BybitConfig
from runner.ownership import _account_scope, _safe_token

DEFAULT_ENV_FILE = ".env"
DEFAULT_KILL_DIR = "data/runtime/kills"
STARTUP_GUARD_EXIT_CODE = 73

log = logging.getLogger(__name__)


class RuntimeKillLatch:
    def __init__(self, path: Path) -> None:
        self.path = path

    def is_armed(self) -> bool:
        return self.path.exists()

    def read(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        raw = self.path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            return {"corrupt": True, "raw": ""}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"corrupt": True, "raw": raw}
        if not isinstance(payload, dict):
            return {"corrupt": True, "raw": raw}
        return payload

    def arm(self, *, reason: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "reason": reason,
            "pid": os.getpid(),
            "armed_at": datetime.now(timezone.utc).isoformat(),
        }
        if payload:
            record.update(payload)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(_json_safe(record), handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return record

    def clear(self) -> bool:
        if not self.path.exists():
            return False
        self.path.unlink()
        return True


def build_bybit_kill_latch(
    *,
    adapter: Any,
    service_name: str,
    scope_name: str,
    latch_dir: str = DEFAULT_KILL_DIR,
) -> RuntimeKillLatch:
    account_scope = _account_scope(adapter)
    account_hash = hashlib.sha256(account_scope.encode("utf-8")).hexdigest()[:16]
    filename = f"{_safe_token(service_name)}_{account_hash}_{_safe_token(scope_name)}.json"
    return RuntimeKillLatch(Path(latch_dir) / filename)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def require_kill_latch_clear(latch: RuntimeKillLatch, *, runtime_name: str) -> None:
    if not latch.is_armed():
        return
    record = latch.read() or {}
    reason = record.get("reason", "unknown")
    armed_at = record.get("armed_at")
    detail = f"reason={reason}"
    if armed_at:
        detail = f"{detail} armed_at={armed_at}"
    raise RuntimeError(
        f"persistent kill latch armed for {runtime_name}: {detail} path={latch.path}"
    )


class PersistentKillSwitch:
    """Wrap a runtime kill switch and persist every arm() to disk."""

    def __init__(
        self,
        kill_switch: Any,
        *,
        latch: RuntimeKillLatch | None,
        service_name: str,
        scope_name: str,
    ) -> None:
        self._kill_switch = kill_switch
        self._latch = latch
        self._service_name = service_name
        self._scope_name = scope_name

    def is_armed(self) -> bool:
        return bool(self._kill_switch.is_armed()) or bool(self._latch and self._latch.is_armed())

    def arm(self, scope: str, key: str, mode: str, reason: str, **kwargs: Any) -> Any:
        result = self._kill_switch.arm(scope, key, mode, reason, **kwargs)
        if self._latch is not None:
            try:
                payload = {
                    "service": self._service_name,
                    "scope_name": self._scope_name,
                    "scope": scope,
                    "key": key,
                    "mode": mode,
                }
                if kwargs:
                    payload["context"] = kwargs
                self._latch.arm(reason=reason, payload=payload)
            except Exception:
                log.exception("Failed to persist runtime kill latch for %s", self._service_name)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kill_switch, name)


def _read_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip("'").strip('"')
    return data


def _merged_env(env_file: str) -> dict[str, str]:
    merged = _read_env_file(Path(env_file))
    merged.update({k: v for k, v in os.environ.items() if v})
    return merged


def _config_holder(env_file: str) -> Any:
    env = _merged_env(env_file)
    api_key = env.get("BYBIT_API_KEY", "")
    if not api_key:
        raise RuntimeError("BYBIT_API_KEY is required to resolve the kill latch scope")
    cfg = BybitConfig(
        api_key=api_key,
        api_secret=env.get("BYBIT_API_SECRET", ""),
        base_url=env.get("BYBIT_BASE_URL", "https://api-demo.bybit.com"),
        account_type=env.get("BYBIT_ACCOUNT_TYPE", "UNIFIED"),
        category=env.get("BYBIT_CATEGORY", "linear"),
    )
    return SimpleNamespace(_config=cfg)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or clear persistent active-runtime kill latches")
    parser.add_argument("--service", choices=("alpha", "mm"), required=True)
    parser.add_argument("--symbol", default="ETHUSDT", help="Market-maker symbol when --service mm")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--clear", action="store_true", help="Clear the latch after manual review")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _service_config(args: argparse.Namespace) -> tuple[str, str]:
    if args.service == "alpha":
        return "bybit-alpha.service", "portfolio"
    return "bybit-mm.service", str(args.symbol).upper()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    service_name, scope_name = _service_config(args)
    try:
        holder = _config_holder(args.env_file)
        latch = build_bybit_kill_latch(
            adapter=holder,
            service_name=service_name,
            scope_name=scope_name,
        )
    except Exception as exc:
        payload = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"runtime_kill_latch: error: {exc}")
        return 1

    cleared = False
    if args.clear:
        cleared = latch.clear()
    record = latch.read()
    payload = {
        "ok": True,
        "service": service_name,
        "scope": scope_name,
        "path": str(latch.path),
        "armed": latch.is_armed(),
        "cleared": cleared,
        "record": record,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
