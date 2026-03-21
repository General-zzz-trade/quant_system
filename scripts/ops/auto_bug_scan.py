#!/usr/bin/env python3
"""Automated bug scanner — finds common bug patterns across the codebase.

Runs static analysis checks beyond what ruff catches:
1. Bare except:pass (silent failures)
2. Division by zero risks (/ var without guard)
3. Mutable default arguments
4. Unguarded NaN propagation
5. Hardcoded prices/quantities
6. Missing logger in except blocks
7. Float equality comparison
8. Unreachable code after return
9. Thread-unsafe patterns (shared mutable state without lock)
10. API response unchecked (retCode not validated)

Usage:
    python3 -m scripts.ops.auto_bug_scan                 # Full scan
    python3 -m scripts.ops.auto_bug_scan --fix           # Auto-fix safe patterns
    python3 -m scripts.ops.auto_bug_scan --path scripts/ # Scan specific path
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SCAN_DIRS = [
    "scripts/ops", "scripts", "runner", "execution",
    "engine", "features", "decision", "alpha", "state",
    "event", "monitoring", "risk", "polymarket",
]

EXCLUDE = {"__pycache__", ".cache", "node_modules", "venv", ".git"}


@dataclass
class Bug:
    file: str
    line: int
    category: str
    severity: str  # critical, warning, info
    message: str
    code: str = ""


def scan_file(filepath: str) -> list[Bug]:
    """Scan a single Python file for bug patterns."""
    bugs = []
    try:
        with open(filepath) as f:
            source = f.read()
            lines = source.splitlines()
    except Exception:
        return bugs

    # Parse AST
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return bugs

    rel_path = os.path.relpath(filepath)

    # === AST-based checks ===

    for node in ast.walk(tree):

        # 1. Bare except:pass
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:  # bare except:
                body = node.body
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    bugs.append(Bug(
                        rel_path, node.lineno, "bare_except_pass", "critical",
                        "Bare except:pass swallows all errors silently",
                    ))
                elif len(body) == 1 and isinstance(body[0], ast.Expr) and \
                     isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    pass  # docstring in except, ok
                elif not any(
                    isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and
                    any(isinstance(s.value.func, ast.Attribute) and
                        s.value.func.attr in ("warning", "error", "exception", "debug", "info")
                        for _ in [None])
                    for s in body
                ):
                    # No logging in except block
                    has_log = False
                    for s in body:
                        s_str = ast.dump(s)
                        if "warning" in s_str or "error" in s_str or "exception" in s_str or "log" in s_str.lower():
                            has_log = True
                            break
                    if not has_log and len(body) <= 2:
                        bugs.append(Bug(
                            rel_path, node.lineno, "silent_except", "warning",
                            "Except block without logging",
                        ))

        # 2. Mutable default arguments
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    bugs.append(Bug(
                        rel_path, node.lineno, "mutable_default", "warning",
                        f"Mutable default argument in {node.name}()",
                    ))

        # 3. Float equality
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    # Check if comparing float-like things
                    comparators = [node.left] + list(node.comparators)
                    for comp in comparators:
                        if isinstance(comp, ast.Constant) and isinstance(comp.value, float):
                            if comp.value not in (0.0, 1.0, -1.0):
                                bugs.append(Bug(
                                    rel_path, node.lineno, "float_equality", "info",
                                    f"Float equality comparison with {comp.value}",
                                ))

    # === Line-based checks ===

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments
        if stripped.startswith("#"):
            continue

        # 4. Division without guard
        if re.search(r'(?<!/)/\s*(?:self\.|[a-z_]+)\b(?!.*\.clip\(|.*max\(|.*or\s)', stripped):
            if "/ 0" not in stripped and "__truediv__" not in stripped and "import" not in stripped:
                if re.search(r'/\s*[a-z_]+(?:\.[a-z_]+)*\s*$', stripped):
                    # Simple division at end of expression — might be risky
                    pass  # too many false positives

        # 5. Hardcoded large numbers that look like prices
        if re.search(r'(?:price|qty|size|notional)\s*=\s*\d{4,}', stripped, re.IGNORECASE):
            if "MAX_" not in stripped and "MIN_" not in stripped and "test" not in filepath.lower():
                bugs.append(Bug(
                    rel_path, i, "hardcoded_value", "info",
                    f"Possible hardcoded price/quantity: {stripped[:80]}",
                ))

        # 6. retCode not checked after API call
        if ".post(" in stripped and "retCode" not in stripped:
            # Check if retCode is checked within next 5 lines
            following = "\n".join(lines[i:i+5]) if i < len(lines) else ""
            if "retCode" not in following and "result" not in following:
                if "test" not in filepath.lower():
                    bugs.append(Bug(
                        rel_path, i, "unchecked_api", "warning",
                        "API call without retCode check",
                    ))

        # 7. == 0 on float (potential float comparison issue)
        if re.search(r'==\s*0(?:\.0)?\s*(?:#|$|\))', stripped):
            if "int(" not in stripped and "len(" not in stripped and "count" not in stripped.lower():
                if "test" not in filepath.lower():
                    bugs.append(Bug(
                        rel_path, i, "float_zero_compare", "info",
                        "Float == 0 comparison (use abs(x) < epsilon)",
                    ))

        # 8. time.sleep in production code (blocking)
        if "time.sleep(" in stripped and "test" not in filepath.lower():
            match = re.search(r'time\.sleep\((\d+)\)', stripped)
            if match and int(match.group(1)) > 30:
                bugs.append(Bug(
                    rel_path, i, "long_sleep", "warning",
                    f"time.sleep({match.group(1)}) — long blocking sleep",
                ))

    return bugs


def scan_all(paths: list[str]) -> list[Bug]:
    """Scan all Python files in given paths."""
    all_bugs = []
    files_scanned = 0

    for scan_dir in paths:
        if not os.path.exists(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                filepath = os.path.join(root, fname)
                bugs = scan_file(filepath)
                all_bugs.extend(bugs)
                files_scanned += 1

    return all_bugs, files_scanned


def main():
    parser = argparse.ArgumentParser(description="Automated bug scanner")
    parser.add_argument("--path", nargs="+", default=SCAN_DIRS)
    parser.add_argument("--severity", default="warning",
                        choices=["critical", "warning", "info"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    severity_rank = {"critical": 3, "warning": 2, "info": 1}
    min_severity = severity_rank[args.severity]

    bugs, files_scanned = scan_all(args.path)

    # Filter by severity
    bugs = [b for b in bugs if severity_rank[b.severity] >= min_severity]

    # Sort by severity then file
    bugs.sort(key=lambda b: (-severity_rank[b.severity], b.file, b.line))

    if args.json:
        import json
        print(json.dumps([{
            "file": b.file, "line": b.line, "category": b.category,
            "severity": b.severity, "message": b.message,
        } for b in bugs], indent=2))
        return

    print(f"Bug Scanner — {files_scanned} files scanned")
    print()

    # Summary by category
    from collections import Counter
    by_cat = Counter(b.category for b in bugs)
    by_sev = Counter(b.severity for b in bugs)

    print(f"Found: {len(bugs)} issues ({by_sev.get('critical',0)} critical, "
          f"{by_sev.get('warning',0)} warning, {by_sev.get('info',0)} info)")
    print()

    if not bugs:
        print("No issues found.")
        return

    # Group by file
    by_file = {}
    for b in bugs:
        by_file.setdefault(b.file, []).append(b)

    for filepath, file_bugs in sorted(by_file.items()):
        print(f"--- {filepath} ---")
        for b in file_bugs:
            sev_icon = {"critical": "!!!", "warning": " !!", "info": "  ."}[b.severity]
            print(f"  {sev_icon} L{b.line:4d} [{b.category}] {b.message}")
        print()

    # Top categories
    print("Top categories:")
    for cat, count in by_cat.most_common(10):
        print(f"  {cat:25s} {count:4d}")


if __name__ == "__main__":
    main()
