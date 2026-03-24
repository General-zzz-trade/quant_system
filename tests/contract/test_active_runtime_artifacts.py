from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


ACTIVE_RUNTIME_ARTIFACTS = (
    "runner/alpha_main.py",
    "infra/systemd/bybit-alpha.service",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_active_runtime_artifacts_exist() -> None:
    root = _repo_root()
    for rel_path in ACTIVE_RUNTIME_ARTIFACTS:
        assert (root / rel_path).exists(), rel_path


def test_active_runtime_artifacts_are_tracked_when_git_metadata_present() -> None:
    root = _repo_root()
    if not (root / ".git").exists():
        pytest.skip("git metadata not available")

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *ACTIVE_RUNTIME_ARTIFACTS],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "Active runtime artifacts must be tracked by git.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
