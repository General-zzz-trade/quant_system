"""Read-only release identity and paper acceptance evidence.

File hashes identify bytes; they do not certify model quality, a live feed or an
account. This gate supports the existing manual paper deployment only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any
import urllib.request

from scripts.check_paper_runtime import check_paper_runtime, shadow_run_manifest
from shared.runtime_state import _json

_REVISION = re.compile(r"[0-9a-f]{40}")
_IMAGE = re.compile(r"sha256:[0-9a-f]{64}")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
SCHEMA_VERSIONS_SUPPORTED = {"runtime": 1, "portfolio": 1, "wallet": 1}


def _identity(repository: str, revision: str) -> None:
    if not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository):
        raise ValueError("Release repository must be owner/name")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError("Release revision must be a full lowercase Git SHA")


def fetch_ci_runs(repository: str, revision: str, *, token: str = "") -> list[dict]:
    """Fetch actual GitHub evidence; never send credentials to another host."""
    _identity(repository, revision)
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "quant-release-check"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/actions/runs?head_sha={revision}&per_page=100",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = _json(response.read().decode("utf-8"))
    if not isinstance(result.get("workflow_runs"), list):
        raise ValueError("GitHub response has no workflow runs")
    return result["workflow_runs"]


def _ci_evidence(runs: list[dict], repository: str, revision: str) -> dict:
    if not isinstance(runs, list):
        raise ValueError("CI evidence must contain actual workflow runs")
    matching = [run for run in runs if isinstance(run, dict)
                and run.get("head_sha") == revision and run.get("head_branch") == "master"
                and run.get("path") == ".github/workflows/ci.yml" and run.get("event") == "push"
                and isinstance(run.get("repository"), dict)
                and run.get("repository", {}).get("full_name") == repository
                and type(run.get("id")) is int and run["id"] > 0]
    if not matching:
        raise ValueError("Exact master revision has no matching CI run")
    latest = max(matching, key=lambda run: run["id"])
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise ValueError("Latest CI run for the exact master revision has not passed")
    return {"id": latest["id"], "head_sha": revision, "head_branch": "master",
            "event": "push", "path": ".github/workflows/ci.yml", "conclusion": "success",
            "url": f"https://github.com/{repository}/actions/runs/{latest['id']}"}


def _safe_path(path: Path) -> Path:
    result = path.absolute()
    if ".." in result.parts or any(p.is_symlink() for p in (result, *result.parents)):
        raise ValueError("Release input paths cannot contain traversal or symbolic links")
    return result


def _file_entry(path: Path, name: str) -> dict:
    path = _safe_path(path)
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
            raise ValueError("Release inputs must be nonempty regular files: " + name)
        while block := stream.read(1024 * 1024):
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = path.stat()
    def signature(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if signature(before) != signature(after) or signature(before) != signature(current):
        raise ValueError("Release input changed while being hashed: " + name)
    return {"path": name, "size": before.st_size, "sha256": digest.hexdigest()}


def _tree(root: Path, prefix: str) -> list[dict]:
    root = _safe_path(root)
    if not root.is_dir():
        raise ValueError("Release input directory is missing: " + prefix)
    result = []
    for path in sorted(root.rglob("*")):
        _safe_path(path)
        if not path.is_dir():
            result.append(_file_entry(path, prefix + "/" + path.relative_to(root).as_posix()))
    return result


def _mount(service: dict, target: str) -> Path:
    mounts = service.get("volumes", [])
    if not isinstance(mounts, list) or any(not isinstance(mount, dict) for mount in mounts):
        raise ValueError("Release input mounts must be rendered Compose objects")
    if any(str(mount.get("target", "")).startswith(target + "/") for mount in mounts):
        raise ValueError("Release input mounts cannot be shadowed")
    found = [mount for mount in mounts if mount.get("target") == target]
    if len(found) != 1 or found[0].get("type") != "bind" or found[0].get("read_only") is not True:
        raise ValueError("Release inputs require one read-only bind mount: " + target)
    source = Path(found[0]["source"])
    if not source.is_absolute():
        raise ValueError("Rendered release input paths must be absolute")
    return _safe_path(source)


def _model_references(config: dict) -> set[str]:
    """Include every declared horizon/regime model, including optional branches."""
    result = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"lgbm", "xgb", "ridge", "meta"} and isinstance(item, str) and item:
                    result.add(item)
                elif key == "models" and isinstance(item, list):
                    result.update(name for name in item if isinstance(name, str) and name)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(config)
    if not result:
        raise ValueError("Model config has no declared model artifacts")
    for name in result:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Model reference escapes its model directory")
    return result


def _runtime_contract(config: dict) -> dict:
    service = config["services"]["quant-paper"]
    if not isinstance(service, dict) or service.get("entrypoint") or service.get("working_dir", "/app") != "/app":
        raise ValueError("Release requires the image runtime entry and working directory")
    mounts = service.get("volumes", [])
    targets = {"/app/data/runtime", "/app/models_v8", "/app/data_files", "/app/logs"}
    if (not isinstance(mounts, list) or any(not isinstance(mount, dict) for mount in mounts)
            or any(mount.get("target") not in targets for mount in mounts)):
        raise ValueError("Release mounts cannot replace image source code; shadowed runtime inputs are forbidden")
    if service.get("volumes_from") or service.get("tmpfs") or service.get("configs") or service.get("secrets"):
        raise ValueError("Release requires explicitly validated runtime mounts")
    return service


def release_input_manifest(config: dict, *, symbol_config: dict | None = None,
                           project_root: Path | None = None) -> dict:
    """Hash mounted bytes actually selectable by the configured paper runners."""
    if symbol_config is None:
        from strategy.config import SYMBOL_CONFIG
        symbol_config = SYMBOL_CONFIG
    service = _runtime_contract(config)
    command = service.get("command")
    manifest = None
    if isinstance(command, list) and "runner.shadow_main" in command:
        manifest = shadow_run_manifest(service, project_root or Path.cwd(), symbol_config=symbol_config)
        keys = list(manifest.runner_keys)
    else:
        if (not isinstance(command, list) or command[:3] != ["python3", "-m", "scripts.run_bybit_alpha"]
                or command.count("--symbols") != 1 or command.count("--dry-run") != 1
                or any(arg in command for arg in ("--enable-execution", "--check-startup", "--check-paper-contract"))):
            raise ValueError("Release requires the explicit paper runtime entry and symbols")
        start = command.index("--symbols") + 1
        keys = []
        for value in command[start:]:
            if not isinstance(value, str):
                raise ValueError("Invalid paper command argument")
            if value.startswith("--"):
                break
            keys.append(value)
    if not keys or len(set(keys)) != len(keys) or any(key not in symbol_config for key in keys):
        raise ValueError("Paper command contains missing, duplicate or unsupported runner keys")
    models = _mount(service, "/app/models_v8")
    data = _mount(service, "/app/data_files")
    if manifest is not None and (models != manifest.models_root or data != manifest.data_root):
        raise ValueError("Shadow mounts must match the validated manifest input roots")
    model_dirs, symbols = set(), set()
    entries = []
    manifest_binding = None
    if manifest is not None:
        entries.append(_file_entry(manifest.manifest_path, "config/shadow.json"))
        manifest_binding = {"path": "config/shadow.json", "digest": manifest.digest, "mode": manifest.mode,
                            "shadow_balance": str(manifest.shadow_balance)}
    for key in keys:
        runner = symbol_config[key]
        name = runner["model_dir"]
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid configured model directory")
        symbols.add(runner.get("symbol", key))
        if name in model_dirs:
            continue
        model_dirs.add(name)
        directory = _safe_path(models / name)
        config_file = _safe_path(directory / "config.json")
        config_entry = _file_entry(config_file, "models/" + name + "/config.json")
        model_config = _json(config_file.read_text(encoding="utf-8"))
        required_entries = [config_entry]
        for reference in _model_references(model_config):
            required_entries.append(_file_entry(directory / reference, "models/" + name + "/" + reference))
        model_entries = _tree(directory, "models/" + name)
        if any(entry not in model_entries for entry in required_entries):
            raise ValueError("Required model input changed during release validation")
        entries.extend(model_entries)
    # All auxiliary feature CSVs are included as well as the three mandatory
    # causal inputs. Missing optional inputs are not invented or certified.
    required_data = []
    for symbol in symbols:
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]+", symbol):
            raise ValueError("Invalid configured market symbol")
        for suffix in ("_1h.csv", "_open_interest.csv", "_ls_ratio.csv"):
            required_data.append(_file_entry(data / (symbol + suffix), "data/" + symbol + suffix))
    data_entries = _tree(data, "data")
    if any(entry not in data_entries for entry in required_data):
        raise ValueError("Required data input changed during release validation")
    entries.extend(data_entries)
    entries.sort(key=lambda entry: entry["path"])
    payload = entries if manifest_binding is None else {"files": entries, "run_manifest": manifest_binding}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = {"sha256": digest, "files": entries, "runners": keys, "symbols": sorted(symbols),
              "mode": "shadow" if manifest is not None else "paper"}
    if manifest_binding is not None:
        result["run_manifest"] = manifest_binding
    return result


def validate_release(*, repository: str, revision: str, config: dict, project_root: Path,
                     image_inspect: dict, ci_runs: list[dict], previous_report: dict | None = None,
                     health: dict | None = None, require_execution: bool = False,
                     symbol_config: dict | None = None, now: float | None = None) -> dict:
    """Validate fresh evidence. A successful result only accepts a paper release."""
    _identity(repository, revision)
    checked_now = time.time() if now is None else now
    if type(checked_now) not in (int, float) or not math.isfinite(checked_now) or checked_now <= 0:
        raise ValueError("Release validation requires a finite positive clock")
    if require_execution:
        raise ValueError("Execution acceptance is unavailable: this release path is paper only")
    ci = _ci_evidence(ci_runs, repository, revision)
    if (not isinstance(image_inspect, dict) or not isinstance(image_inspect.get("Id"), str)
            or not _IMAGE.fullmatch(image_inspect["Id"])
            or not isinstance(image_inspect.get("Config"), dict)
            or not isinstance(image_inspect["Config"].get("Labels"), dict)
            or image_inspect.get("Config", {}).get("Labels", {}).get("org.opencontainers.image.revision") != revision):
        raise ValueError("Image must have an immutable local image ID and exact revision label")
    _runtime_contract(config)
    state = check_paper_runtime(config, project_root, symbol_config=symbol_config)
    identity = _json((state / "runtime-state.json").read_text(encoding="utf-8"))
    inputs = release_input_manifest(config, symbol_config=symbol_config, project_root=project_root)
    release = {"revision": revision, "image_id": image_inspect["Id"], "input_digest": inputs["sha256"]}
    result = {"version": 1, "stage": "preflight", "repository": repository, "branch": "master",
              "checked_at": checked_now, "release": release, "ci": ci,
              "inputs": inputs, "runtime": {"scope": identity["scope"], "state_id": identity["state_id"],
                                            "schema_version": identity["version"]},
              "schema_versions_supported": dict(SCHEMA_VERSIONS_SUPPORTED),
              "paper_accepted": False, "shadow_accepted": False, "runtime_mode": inputs["mode"],
              "execution_eligible": False, "live_acceptance": False,
              "artifact_loading_verified": False, "reason": "runtime_health_not_yet_checked"}
    if (previous_report is None) != (health is None):
        raise ValueError("Post-release acceptance requires both preflight and runtime status")
    if previous_report is not None:
        if not isinstance(previous_report, dict) or not isinstance(health, dict):
            raise ValueError("Post-release evidence must be JSON objects")
        for key in ("version", "repository", "branch", "release", "inputs", "runtime", "schema_versions_supported"):
            if previous_report.get(key) != result[key]:
                raise ValueError("Release evidence changed after preflight: " + key)
        if previous_report.get("stage") != "preflight":
            raise ValueError("Post-release acceptance requires a preflight receipt")
        from shared.runtime_status import evaluate_status
        evaluation = evaluate_status(health, now=checked_now, expected_release=release)
        checked_at = previous_report.get("checked_at")
        if (type(checked_at) not in (int, float) or not math.isfinite(checked_at)
                or not 0 < checked_at <= checked_now
                or health["started_at"] < checked_at or health["heartbeat_at"] < checked_at):
            raise ValueError("Runtime observations must originate after this release preflight")
        if (health.get("mode") != inputs["mode"] or health.get("scope") != identity["scope"]
                or health.get("state_id") != identity["state_id"]
                or health.get("runner_keys") != inputs["runners"] or not evaluation.get("ready")):
            raise ValueError("Runtime has no fresh ready paper evidence for this release")
        if inputs["mode"] == "shadow" and health.get("manifest_digest") != inputs["run_manifest"]["digest"]:
            raise ValueError("Shadow runtime loaded a different manifest than the release preflight")
        result.update(stage="postflight", paper_accepted=True, shadow_accepted=inputs["mode"] == "shadow",
                      reason=inputs["mode"] + "_runtime_ready",
                      runtime_boot_id=health["boot_id"])
    return result
