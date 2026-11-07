"""Verify state survives disposable container replacement without exchange access.

Uses a temporary bind directory and serial containers on the local Docker daemon.
Example: python scripts/check_runtime_recovery.py --image quant-ci:startup-check
Pass --source-root . to test current source with an already-built runtime image.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid

WORKER = "tests/fixtures/runtime_recovery_worker.py"


def docker_environment():
    env = dict(os.environ)
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        env.pop(name, None)
    return env


def run_worker(image, phase, state_dir, *, source_root=None, mount_state=True, extra=()):
    name = "quant-offline-recovery-" + uuid.uuid4().hex
    command = ["docker", "--host=unix:///var/run/docker.sock", "run", "--rm", "--name", name,
               "--user", f"{os.getuid()}:{os.getgid()}",
               "--network", "none", "--workdir", "/app", "-e", "PYTHONPATH=/app",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python"]
    if source_root is not None:
        command += ["--mount", f"type=bind,src={source_root},dst=/app,readonly"]
    if mount_state:
        command += ["--mount", f"type=bind,src={state_dir},dst=/state"]
    command += [image, "-B", WORKER, phase, *extra]
    try:
        result = subprocess.run(command, env=docker_environment(), capture_output=True, text=True, timeout=120)
    except BaseException:
        # Clean only the uniquely named container created by this invocation.
        subprocess.run(["docker", "--host=unix:///var/run/docker.sock", "rm", "-f", name],
                       env=docker_environment(), capture_output=True, timeout=20)
        raise
    print(f"CONTAINER {phase}: exit={result.returncode}; state_mount={mount_state}", flush=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print(result.stderr.rstrip(), flush=True)
    return result


def result_object(result):
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("Recovery container did not return its verification record") from exc


def check_compose(project_root, state_dir, scope):
    """Render actual Compose without loading account env files or starting services."""
    from scripts.check_paper_runtime import check_paper_runtime

    # Compose still requires env_file to exist with --no-env-resolution. Render
    # an exact copy of the checked-in YAML next to an empty synthetic .env.
    # Never create, inspect or resolve the checkout's account configuration.
    with tempfile.TemporaryDirectory(prefix="quant-compose-validation-") as scratch:
        fixture = Path(scratch)
        (fixture / "docker-compose.yml").write_bytes((project_root / "docker-compose.yml").read_bytes())
        (fixture / ".env").write_text("")
        command = ["docker", "--host=unix:///var/run/docker.sock", "compose",
                   "--project-directory", str(fixture), "--file", str(fixture / "docker-compose.yml"),
                   "--env-file", "/dev/null", "--profile", "live", "--profile", "framework",
                   "config", "--no-env-resolution", "--format", "json"]
        env = docker_environment()
        env.update(QUANT_PAPER_STATE_DIR=str(state_dir), QUANT_PAPER_RISK_SCOPE=scope)
        configured = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        if configured.returncode:
            raise RuntimeError("Actual Compose configuration failed to render")
        check_paper_runtime(json.loads(configured.stdout), project_root)
        # Optional profiles must still parse when paper has not been configured.
        env.pop("QUANT_PAPER_STATE_DIR")
        env.pop("QUANT_PAPER_RISK_SCOPE")
        unconfigured = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        if unconfigured.returncode:
            raise RuntimeError("Unconfigured paper variables broke other Compose profiles")
        try:
            check_paper_runtime(json.loads(unconfigured.stdout), project_root)
        except (ValueError, OSError, RuntimeError):
            pass
        else:
            raise RuntimeError("Unconfigured paper state unexpectedly passed deployment preflight")
    print("COMPOSE: persistent paper mount verified; other profiles parse; unconfigured paper rejected.", flush=True)


def check(image, *, source_root=None, inject_missing_state_mount=False):
    with tempfile.TemporaryDirectory(prefix="quant-container-recovery-") as scratch:
        directory = Path(scratch)
        prepared = run_worker(image, "prepare", directory, source_root=source_root)
        if prepared.returncode != 0:
            raise RuntimeError("First container failed to prepare durable fixture state")
        initial = result_object(prepared)
        if initial.get("phase") != "prepared":
            raise RuntimeError("First container did not finish preparing fixture state")
        project_root = source_root or Path(__file__).resolve().parents[1]
        check_compose(project_root, directory, initial["scope"])
        extra = ("--state-id", initial["state_id"], "--client-id", initial["unknown_client_id"])
        recovered = run_worker(image, "recover", directory, source_root=source_root,
                               mount_state=not inject_missing_state_mount, extra=extra)
        if recovered.returncode != 0 or result_object(recovered).get("phase") != "recovered":
            raise RuntimeError("Replacement container failed the durable recovery contract")
        # Independent negative case: removing the state mount must fail closed.
        missing = run_worker(image, "recover", directory, source_root=source_root,
                             mount_state=False, extra=extra)
        blocked = result_object(missing)
        if missing.returncode != 2 or blocked.get("phase") != "blocked" or "missing" not in blocked.get("reason", ""):
            raise RuntimeError("Missing persistent state was not rejected at startup")
    print("Container replacement recovery passed; missing state was rejected; no exchange access.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--inject-missing-state-mount", action="store_true",
                        help="Test-only mutation: remove the replacement container mount; this command must fail")
    args = parser.parse_args()
    source_root = args.source_root.resolve() if args.source_root is not None else None
    if source_root is not None and not (source_root / WORKER).is_file():
        parser.error("source root does not contain the recovery fixture")
    try:
        check(args.image, source_root=source_root, inject_missing_state_mount=args.inject_missing_state_mount)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print("Container recovery check FAILED:", exc, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
