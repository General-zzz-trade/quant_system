"""The Claude Code guard hook (.claude/hooks/guard.py) blocks what the project rules forbid and nothing else.

Rules it enforces: never read or write the env file; never read docs/_archive/; never force-push and never
push to master (master changes only through a merged PR).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
GUARD = ROOT / ".claude" / "hooks" / "guard.py"
_spec = importlib.util.spec_from_file_location("claude_guard", GUARD)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _bash(command, branch="feature"):
    return guard.check_bash(command, lambda: branch)


@pytest.mark.parametrize("command", [
    "cat .env", "grep TOKEN /quant_system/.env", "source .env.local && echo x", "cp .env.polymarket /tmp/x",
    "echo $(cat .env)", "wc -l <.env", "ls docs/_archive/",
    "git push --force origin x", "git push -f origin x", "git push -uf origin x",
    "git push --force-with-lease origin x", "git push origin +x", "git push --mirror origin",
    "git push origin master", "git push origin HEAD:master", "git push origin :master", "git push --all origin",
])
def test_blocked_commands(command):
    assert _bash(command) is not None


@pytest.mark.parametrize("command", [
    "git grep -n polymarket -- .env.example",
    "git commit -q -F - <<'EOF'\nThe .env.polymarket ignore rule stays; see docs/_archive/x\nEOF",
    'git grep -n "EnvironmentFile=/quant_system/.env" -- infra',
    "git grep -n foo -- . ':!docs/_archive'",
    'git grep -n "moved to docs/_archive in March" -- docs/arch',
    "git push -q -u origin cleanup-batch-e-venues",
    "git push -q origin --delete cleanup-batch-e-venues",
    "for i in 1 2; do git push -q -u origin fix-master-docs && break; done",
    "ruff check --select E,W,F . && git status",
])
def test_allowed_commands(command):
    assert _bash(command) is None


@pytest.mark.parametrize("command", ["git push", "git push origin", "git push origin HEAD"])
def test_a_push_of_the_current_branch_is_blocked_only_on_master(command):
    assert _bash(command, branch="master") is not None
    assert _bash(command, branch="feature") is None


@pytest.mark.parametrize("tool_input,blocked", [
    ({"file_path": "/root/quant_system/.env"}, True),
    ({"file_path": "/root/quant_system/.env.example"}, False),
    ({"file_path": "/root/quant_system/docs/_archive/old.md"}, True),
    ({"file_path": "/root/quant_system/docs/arch/target.md"}, False),
    ({"pattern": "x", "path": "docs/_archive"}, True),
    ({"pattern": "docs/_archive/**"}, True),
    ({"pattern": "TELEGRAM", "path": "trading"}, False),
])
def test_file_tools(tool_input, blocked):
    assert (guard.check_paths(tool_input) is not None) is blocked


def _run(mode, event):
    return subprocess.run([sys.executable, "-I", str(GUARD), mode], input=event, capture_output=True,
                          text=True, timeout=60)


def test_the_hook_exits_2_with_a_reason_when_it_blocks_and_0_otherwise():
    blocked = _run("pre", json.dumps({"tool_name": "Read", "tool_input": {"file_path": ".env"}}))
    assert blocked.returncode == 2 and "env file" in blocked.stderr
    assert _run("pre", json.dumps({"tool_name": "Read", "tool_input": {"file_path": "README.md"}})).returncode == 0


def test_a_broken_input_is_let_through_with_a_message_not_a_block():
    out = _run("pre", "{not json")
    assert out.returncode == 1 and "was not checked" in out.stderr


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed here")
def test_the_post_hook_reports_ruff_failures(tmp_path):
    bad, ok = tmp_path / "bad.py", tmp_path / "ok.py"
    bad.write_text("import os\n")
    ok.write_text("x = 1\n")
    failed = _run("post", json.dumps({"tool_name": "Edit", "tool_input": {"file_path": str(bad)}}))
    assert failed.returncode == 2 and "F401" in failed.stdout + failed.stderr
    assert _run("post", json.dumps({"tool_name": "Edit", "tool_input": {"file_path": str(ok)}})).returncode == 0
