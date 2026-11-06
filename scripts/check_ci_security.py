"""Fail CI on tracked Python credential literals or private-key file names.

Reports locations only, never credential values. This is a narrow assignment
check, not a general secret detector or a scan of Git history.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path


CREDENTIAL = re.compile(r"(?:.*_)?(?:API_KEY|API_SECRET|API_PASSPHRASE|BOT_TOKEN)$", re.IGNORECASE)


def check(root: Path) -> int:
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True, check=True,
    )
    if Path(top.stdout.strip()).resolve() != root.resolve():
        print("Security check FAILED: run from the repository root", file=sys.stderr)
        return 1
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached"], cwd=root, capture_output=True, check=True,
    )
    failures = []
    for name in result.stdout.decode().split("\0"):
        if not name:
            continue
        path = root / name
        if path.suffix.lower() in {".pem", ".key", ".p12"}:
            failures.append(f"{name}: private-key file name is forbidden")
        if path.suffix != ".py" or path.is_symlink():
            continue  # Do not read external symlink targets or local .env files.
        try:
            tree = ast.parse(path.read_bytes(), filename=name)
        except (OSError, SyntaxError, ValueError):
            failures.append(f"{name}: unable to scan Python source")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigned_value = node.value
            if assigned_value is None:
                continue
            for target in targets:
                # Pair tuple assignments as well as ordinary/chained assignments.
                pairs = [(target, assigned_value)]
                while pairs:
                    target, value = pairs.pop()
                    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
                        pairs.extend(zip(target.elts, value.elts))
                        continue
                    name_part = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
                    if not CREDENTIAL.fullmatch(name_part):
                        continue
                    try:
                        literal = ast.literal_eval(value)
                    except (ValueError, TypeError, SyntaxError):
                        continue  # Environment/config lookups are not literal credentials.
                    if isinstance(literal, (str, bytes)) and literal:
                        failures.append(f"{name}:{node.lineno}: hardcoded credential assignment")
    if failures:
        print("Security check FAILED:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("Security check passed (tracked Python assignments and private-key file names)")
    return 0


def main() -> int:
    try:
        return check(Path.cwd())
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        print("Security check FAILED: unable to enumerate tracked files", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
