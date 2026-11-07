"""Validate rendered paper Compose state paths without starting services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from shared.runtime_state import validate_runtime_state


def check_paper_runtime(config: dict, project_root: Path) -> Path:
    service = config["services"]["quant-paper"]
    environment = service["environment"]
    target = "/app/data/runtime"
    if environment.get("RUNTIME_STATE_DIR") != target:
        raise ValueError("Paper runtime must use the persistent runtime directory")
    if environment.get("ORDER_JOURNAL_PATH") != target + "/orders.sqlite":
        raise ValueError("Paper journal must be inside the persistent runtime directory")
    if environment.get("DRAWDOWN_STATE_PATH") != target + "/drawdown.sqlite":
        raise ValueError("Paper drawdown state must be inside the persistent runtime directory")
    if "--dry-run" not in service.get("command", []):
        raise ValueError("Paper deployment requires --dry-run")
    volumes = service.get("volumes", [])
    if any(str(v.get("target", "")).startswith(target + "/") for v in volumes):
        raise ValueError("Nested mounts must not shadow persistent runtime files")
    mounts = [v for v in volumes if v.get("target") == target]
    if len(mounts) != 1:
        raise ValueError("Paper runtime requires exactly one persistent state mount")
    mount = mounts[0]
    if (mount.get("type") != "bind" or mount.get("read_only", False)
            # Compose omits false values in its canonical JSON output.
            or mount.get("bind", {}).get("create_host_path", False)):
        raise ValueError("Paper state must be a writable bind mount with create_host_path=false")
    source = Path(mount["source"])
    if not source.is_absolute():
        raise ValueError("Paper state source must be an absolute host path")
    if source.resolve().is_relative_to(project_root.resolve()):
        raise ValueError("Paper state must live outside the checkout so checkout cleanup cannot erase it")
    validate_runtime_state(source, environment["RISK_SCOPE"])
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        check_paper_runtime(json.load(sys.stdin), args.project_root)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, sqlite3.Error) as exc:
        # Rendered Compose may contain credentials; never echo its contents.
        print(f"Paper state preflight failed: {exc}", file=sys.stderr)
        return 1
    print("Paper persistent state preflight passed; no services changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
