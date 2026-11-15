#!/usr/bin/env python3
"""Claude Code hooks for this repository (standard library only).

``pre``  (PreToolUse): refuse a tool call that
  - reads or writes the env file (``.env``, ``.env.local``, ...; ``.env.example`` is allowed),
  - touches ``docs/_archive/`` (a git exclusion pathspec such as ``':!docs/_archive'`` is allowed),
  - force-pushes, or pushes to ``master`` (changes reach master only through a PR merge).
``post`` (PostToolUse): run ``ruff check --select E,W,F`` on an edited Python file and report failures.

Exit code 2 blocks the call (pre) or shows the message to Claude (post). These are guards against
mistakes, not a sandbox: text inside a heredoc is not inspected, so a commit message may mention
the env file. If this script itself fails it lets the call through and says so, so a bug here
cannot lock every tool.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys

ENV_FILE = re.compile(r"(?:^|/)\.env(?:\.(?!example$)[A-Za-z0-9_.-]+)?$")
ARCHIVE = "docs/_archive"
EXCLUDE_PREFIXES = (":!", ":^", ":(exclude)")
HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
SEGMENT = re.compile(r"&&|\|\||[;|\n]")
PROTECTED = "master"


def _env_path(text: str) -> bool:
    return bool(ENV_FILE.search(text.strip().rstrip("/")))


def _archive_path(text: str) -> bool:
    return ARCHIVE in text and not text.startswith(EXCLUDE_PREFIXES)


def strip_heredocs(command: str) -> str:
    """Drop heredoc bodies (commit messages, inline scripts): only the command lines are checked."""
    out, lines, i = [], command.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for match in HEREDOC.finditer(line):
            while i < len(lines) and lines[i].strip() != match.group(2):
                i += 1
            i += 1
    return "\n".join(out)


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment, comments=True)
    except ValueError:
        return segment.split()


def _push_problem(tokens: list[str], current_branch) -> str | None:
    words = [t for t in tokens if "=" not in t or t.startswith("-")]
    if "git" not in words or "push" not in words or words.index("push") < words.index("git"):
        return None
    args = words[words.index("push") + 1:]
    options = [a for a in args if a.startswith("-")]
    refs = [a for a in args if not a.startswith("-")]
    for opt in options:
        if opt.startswith("--force") or opt == "--mirror" or (not opt.startswith("--") and "f" in opt[1:]):
            return f"force push ({opt}) is not allowed; history is never rewritten"
    if "--all" in options:
        return "git push --all would push master; push the branch by name"
    for ref in refs[1:]:
        if ref.startswith("+"):
            return f"force push ({ref}) is not allowed; history is never rewritten"
        target = ref.rsplit(":", 1)[-1]
        if target in (PROTECTED, f"refs/heads/{PROTECTED}") or (target == "HEAD" and current_branch() == PROTECTED):
            return "pushing to master is not allowed; master changes only through a merged PR"
    if len(refs) <= 1 and "--delete" not in options and "-d" not in options and current_branch() == PROTECTED:
        return "this push would update master (current branch); push a branch and open a PR"
    return None


def check_bash(command: str, current_branch) -> str | None:
    visible = strip_heredocs(command)
    for segment in SEGMENT.split(visible):
        tokens = _tokens(segment)
        for token in (t.strip("()<>`") for t in tokens):
            if " " in token or ("=" in token and not token.startswith("-")):
                continue                               # a search pattern or an assignment, not a path
            if _env_path(token):
                return f"the env file ({token}) must not be read or written; it holds credentials"
            if _archive_path(token):
                return "docs/_archive/ must not be read (old docs are not evidence)"
        problem = _push_problem(tokens, current_branch)
        if problem:
            return problem
    return None


def check_paths(tool_input: dict) -> str | None:
    for key in ("file_path", "path", "notebook_path", "pattern", "glob"):
        value = tool_input.get(key)
        if not isinstance(value, str):
            continue
        if key in ("file_path", "path", "notebook_path") and _env_path(value):
            return f"the env file ({value}) must not be read or written; it holds credentials"
        if _archive_path(value):
            return "docs/_archive/ must not be read (old docs are not evidence)"
    return None


def _branch_reader(cwd: str):
    def current() -> str:
        try:
            out = subprocess.run(["git", "-C", cwd or ".", "rev-parse", "--abbrev-ref", "HEAD"],
                                 capture_output=True, text=True, timeout=5)
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return current


def pre(event: dict) -> int:
    tool, tool_input = event.get("tool_name", ""), event.get("tool_input") or {}
    if tool == "Bash":
        problem = check_bash(str(tool_input.get("command", "")), _branch_reader(event.get("cwd", "")))
    else:
        problem = check_paths(tool_input)
    if problem:
        print(f"Blocked by .claude/hooks/guard.py: {problem}.", file=sys.stderr)
        return 2
    return 0


def post(event: dict) -> int:
    tool_input = event.get("tool_input") or {}
    path = tool_input.get("file_path") or (event.get("tool_response") or {}).get("filePath") or ""
    if not path.endswith(".py") or not os.path.isfile(path):
        return 0
    try:
        out = subprocess.run(["ruff", "check", "--select", "E,W,F", "--quiet", path],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return 0                                       # ruff missing or hung: CI still lints
    if out.returncode != 0:
        print(f"ruff check --select E,W,F failed for {path}:\n{out.stdout}{out.stderr}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str]) -> int:
    try:
        event = json.load(sys.stdin)
        return pre(event) if argv[1:] == ["pre"] else post(event)
    except Exception as exc:                           # noqa: BLE001 - never lock every tool on a bug here
        print(f"guard.py failed ({type(exc).__name__}: {exc}); the call was not checked.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
