"""Check rendered execution service storage without starting any service.

Input is Docker Compose canonical JSON on stdin. A successful result validates
local state and mount bindings only; fresh account reconciliation is still
required inside the runner before new exposure is allowed.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path, PurePosixPath
import sqlite3
import sys
from typing import Any

from runner.portfolio_risk import bybit_environment
from shared.portfolio_reservations import PortfolioReservationStore, ScopeRegistration
from shared.runtime_state import MANIFEST_NAME, _json, _path, validate_execution_state
from strategy.config import PORTFOLIO_NOTIONAL_CAPS_USD


def _container_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value:
        raise ValueError("Execution state paths must be absolute container paths")
    path = PurePosixPath(value)
    if ".." in path.parts or str(path) != value:
        raise ValueError("Execution state paths must be canonical")
    return path


def _persistent_directory(service: dict, target: PurePosixPath, project_root: Path) -> Path:
    volumes = service.get("volumes", [])
    if not isinstance(volumes, list) or any(not isinstance(row, dict) for row in volumes):
        raise ValueError("Expected rendered Compose mount objects")
    for row in volumes:
        path = _container_path(row.get("target"))
        if path != target and target in path.parents:
            raise ValueError("Nested mounts cannot shadow execution state")
    matching = [row for row in volumes if row.get("target") == str(target)]
    if len(matching) != 1:
        raise ValueError("Each execution state directory needs exactly one persistent bind mount")
    mount = matching[0]
    if (mount.get("type") != "bind" or mount.get("read_only", False) is not False
            or not isinstance(mount.get("bind", {}), dict)
            or mount.get("bind", {}).get("create_host_path", False) is not False):
        raise ValueError("Execution state needs a writable bind with create_host_path=false")
    source = mount.get("source")
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise ValueError("Execution state requires an absolute host source")
    path = _path(Path(source))
    if not path.is_dir() or not path.stat().st_mode & 0o222:
        raise ValueError("Execution state source must be an existing writable directory")
    root = project_root.resolve()
    if path == root or root in path.parents:
        raise ValueError("Execution state must live outside the checkout")
    return path


def check_execution_runtime(config: dict, project_root: Path, service_name: str = "quant-live") -> dict:
    service = config["services"][service_name]
    if not isinstance(service, dict) or service.get("entrypoint"):
        raise ValueError("Execution preflight requires the standard Alpha entrypoint")
    command = service.get("command")
    if (not isinstance(command, list) or len(command) < 4
            or command[0] not in {"python", "python3"} or command[1] != "-m"
            or command[2] not in {"runner.alpha_main", "scripts.run_bybit_alpha"}
            or command.count("--enable-execution") != 1
            or any(flag in command for flag in ("--dry-run", "--check-startup", "--check-paper-contract"))):
        raise ValueError("Execution service must explicitly enable the standard Alpha runner")
    if command.count("--venue") > 1:
        raise ValueError("Execution service declares multiple venues")
    if "--venue" in command:
        index = command.index("--venue")
        if index + 1 >= len(command) or command[index + 1] != "bybit":
            raise ValueError("Execution state preflight currently supports Bybit only")
    if any(str(value).startswith("--venue=") and value != "--venue=bybit" for value in command):
        raise ValueError("Execution state preflight currently supports Bybit only")
    environment = service.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("Execution service requires resolved environment bindings")
    target = PurePosixPath("/app/data/runtime")
    for name, expected in {
        "RUNTIME_STATE_DIR": target,
        "ORDER_JOURNAL_PATH": target / "orders.sqlite",
        "DRAWDOWN_STATE_PATH": target / "drawdown.sqlite",
    }.items():
        if _container_path(environment.get(name)) != expected:
            raise ValueError("Execution state paths must use the managed runtime directory")
    state_dir = _persistent_directory(service, target, project_root)
    scope = environment.get("RISK_SCOPE")
    if not isinstance(scope, str) or not scope or scope.strip() != scope:
        raise ValueError("Execution service needs an explicit account scope")
    validate_execution_state(state_dir, scope, require_writable=False)
    manifest = _json((state_dir / MANIFEST_NAME).read_text(encoding="utf-8"))

    risk_path = _container_path(environment.get("PORTFOLIO_RISK_PATH"))
    risk_target = risk_path.parent
    if target == risk_target or target in risk_target.parents or risk_target in target.parents:
        raise ValueError("Shared risk state requires a separate persistent directory")
    risk_dir = _persistent_directory(service, risk_target, project_root)
    if state_dir == risk_dir or state_dir in risk_dir.parents or risk_dir in state_dir.parents:
        raise ValueError("Account and shared risk host directories must not overlap")
    group = environment.get("PORTFOLIO_RISK_GROUP")
    risk_file = _path(risk_dir / risk_path.name)
    store = PortfolioReservationStore(risk_file, group=group)
    if store.caps != {symbol: Decimal(str(value)) for symbol, value in PORTFOLIO_NOTIONAL_CAPS_USD.items()}:
        raise ValueError("Shared portfolio caps differ from configured strategy caps")
    registration = ScopeRegistration(
        scope, "bybit", bybit_environment(environment.get("BYBIT_BASE_URL", "https://api-demo.bybit.com")),
        manifest["state_id"],
    )
    if registration not in store.registrations:
        raise ValueError("Runtime identity is not declared in the shared risk group")
    with sqlite3.connect(risk_file.as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("Shared risk database failed integrity validation")
    return {"state_validated": True, "execution_eligible": False,
            "reason": "fresh_account_reconciliation_required", "service": service_name,
            "scope": scope, "state_id": manifest["state_id"], "risk_group": group}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--service", default="quant-live")
    args = parser.parse_args()
    try:
        check_execution_runtime(_json(sys.stdin.read()), args.project_root, args.service)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, sqlite3.Error):
        # Rendered environment may contain credentials. Never echo inputs or
        # exception messages that can contain a malformed caller-supplied value.
        print("Execution state preflight FAILED; state, identity or persistent mounts are invalid", file=sys.stderr)
        return 1
    print("Execution state preflight passed; account reconciliation still required; no services changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
