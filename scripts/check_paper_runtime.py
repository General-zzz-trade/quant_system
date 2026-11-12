"""Validate rendered paper Compose state paths without starting services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from shared.runtime_state import validate_runtime_state


SHADOW_COMMAND = ["python3", "-m", "runner.shadow_main", "--manifest", "config/shadow.json", "--dry-run"]


def shadow_run_manifest(service: dict, project_root: Path, *, symbol_config: dict | None = None):
    """Validate the account-free Compose entry and its image-owned run list."""
    from runner.run_manifest import load_run_manifest

    if service.get("command") != SHADOW_COMMAND:
        raise ValueError("Shadow deployment requires the exact manifest-based entry without overrides")
    if service.get("env_file") or service.get("secrets"):
        raise ValueError("Shadow deployment cannot inject account environment files or secrets")
    allowed_environment = {
        "RUNTIME_STATE_DIR", "ORDER_JOURNAL_PATH", "DRAWDOWN_STATE_PATH", "RISK_SCOPE",
        "QUANT_RELEASE_REVISION", "QUANT_RELEASE_IMAGE_ID", "QUANT_RELEASE_INPUT_DIGEST",
    }
    environment = service.get("environment")
    if not isinstance(environment, dict) or set(environment) - allowed_environment:
        raise ValueError("Shadow deployment requires an explicit account-free environment")
    root = project_root.absolute()
    entry = root / "runner/shadow_main.py"
    if entry.is_symlink() or not entry.is_file():
        raise ValueError("Shadow runtime entry is unavailable in the checked-out revision")
    manifest = load_run_manifest(
        root / "config/shadow.json", project_root=root, validate_files=False, symbol_config=symbol_config,
    )
    if manifest.models_root != root / "models_v8" or manifest.data_root != root / "data_files":
        raise ValueError("Shadow deployment inputs must use the standard mounted model and data roots")
    return manifest


def check_paper_runtime(config: dict, project_root: Path, *, symbol_config: dict | None = None) -> Path:
    service = config["services"]["quant-paper"]
    command = service.get("command", [])
    if isinstance(command, list) and "runner.shadow_main" in command:
        shadow_run_manifest(service, project_root, symbol_config=symbol_config)
    environment = service["environment"]
    target = "/app/data/runtime"
    if environment.get("RUNTIME_STATE_DIR") != target:
        raise ValueError("Paper runtime must use the persistent runtime directory")
    if environment.get("ORDER_JOURNAL_PATH") != target + "/orders.sqlite":
        raise ValueError("Paper journal must be inside the persistent runtime directory")
    if environment.get("DRAWDOWN_STATE_PATH") != target + "/drawdown.sqlite":
        raise ValueError("Paper drawdown state must be inside the persistent runtime directory")
    if "--dry-run" not in command:
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
