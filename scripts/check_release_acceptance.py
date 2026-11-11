"""Collect and validate exact-revision paper release evidence without deploying."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import urllib.error

from shared.release_acceptance import fetch_ci_runs, validate_release
from shared.runtime_state import _json


def _write(path: Path, report: dict) -> None:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.absolute().parents):
        raise ValueError("Release report cannot be written through symbolic links")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".release-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--runtime-status", type=Path)
    parser.add_argument("--require-execution", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = _json(sys.stdin.read())
        if args.image != config["services"]["quant-paper"]["image"]:
            raise ValueError("Inspected image must be the configured paper image")
        ci_runs = fetch_ci_runs(args.repository, args.revision, token=os.environ.get("GH_TOKEN", ""))
        image = subprocess.run(["docker", "image", "inspect", "--format", "{{json .}}", args.image],
                               check=True, capture_output=True, text=True, timeout=30)
        report = validate_release(
            repository=args.repository, revision=args.revision, config=config, project_root=args.project_root,
            image_inspect=_json(image.stdout), ci_runs=ci_runs, require_execution=args.require_execution,
            previous_report=_json(args.preflight.read_text()) if args.preflight else None,
            health=_json(args.runtime_status.read_text()) if args.runtime_status else None,
        )
        _write(args.output, report)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, sqlite3.Error,
            subprocess.SubprocessError, urllib.error.URLError) as exc:
        # Neither Docker stderr nor the rendered environment may be printed.
        print("Paper release acceptance failed: " + type(exc).__name__, file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ("stage", "release", "paper_accepted", "execution_eligible")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
