"""ADR-0013 points 2 and 3 (S10 part 3): paper and execution take one path; what differs is ``ModePolicy``.

- The session has no paper branch: no condition reads a sleeve, a mode or anything named paper, and the session
  never reads the read-only broker view the composition root may hang on it for people and tests.
- No module of the application layer compares anything with the mode names "paper" or "execution": they read
  the policy's fields instead.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPLICATION = ROOT / "trading" / "application"
SESSION = APPLICATION / "session.py"
MODE_WORDS = {"sleeve", "mode", "paper"}
MODE_NAMES = {"paper", "execution"}


def _names(node: ast.AST) -> set[str]:
    found = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
    return {name for name in found if name in MODE_WORDS or "paper" in name.lower()}


def mode_branches(tree: ast.AST) -> list[str]:
    """Conditions (if, conditional expressions, while, comprehension filters, assert) that read a sleeve, a mode or
    anything named paper; and every read of an attribute ``sleeve``."""
    found = []
    for node in ast.walk(tree):
        tests: list[ast.AST] = []
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
            tests = [node.test]
        elif isinstance(node, ast.comprehension):
            tests = list(node.ifs)
        found += [f"{test.lineno}: {ast.unparse(test)}" for test in tests if _names(test)]
        if isinstance(node, ast.Attribute) and node.attr == "sleeve" and isinstance(node.ctx, ast.Load):
            found.append(f"{node.lineno}: reads .sleeve")
    return found


def mode_comparisons(tree: ast.AST) -> list[str]:
    """Comparisons with the literal mode names, and matches on them."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            parts = [node.left, *node.comparators]
            if any(isinstance(p, ast.Constant) and p.value in MODE_NAMES for p in parts):
                found.append(f"{node.lineno}: {ast.unparse(node)}")
        elif isinstance(node, ast.MatchValue) and isinstance(node.value, ast.Constant) \
                and node.value.value in MODE_NAMES:
            found.append(f"{node.lineno}: case {node.value.value!r}")
    return found


def test_the_session_has_no_paper_branch():
    assert mode_branches(ast.parse(SESSION.read_text(encoding="utf-8"))) == []


def test_the_application_never_compares_with_a_mode_name():
    offenders = [f"{path.relative_to(ROOT)}:{line}" for path in sorted(APPLICATION.rglob("*.py"))
                 for line in mode_comparisons(ast.parse(path.read_text(encoding="utf-8")))]
    assert offenders == []


def test_the_rules_catch_what_they_forbid():
    branches = mode_branches(ast.parse(
        "if self.sleeve is not None:\n    pass\n"
        "x = 1 if self.mode == 'paper' else 2\n"
        "while paper_store:\n    break\n"
        "y = [s for s in sessions if s.paper_view]\n"
        "assert mode\n"
        "z = self.sleeve.broker\n"
        "if self.executor is None:\n    pass\n"))
    assert sorted({int(line.split(":")[0]) for line in branches}) == [1, 3, 4, 6, 7, 8]
    comparisons = mode_comparisons(ast.parse(
        "a = mode == 'paper'\nb = 'execution' != policy.name\nc = name in ('x',)\n"
        "match m:\n    case 'paper':\n        pass\n"))
    assert sorted(int(line.split(":")[0]) for line in comparisons) == [1, 2, 5]


def test_the_entry_point_loads_neither_research_nor_the_old_research_broker():
    """Completion criterion 1 (S10): the runtime's entry, with paper on the simulated exchange, loads no module of lab/
    and not runner.offline (the old path of the research broker)."""
    import subprocess
    import sys
    code = ("import sys, runner.binance_target; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'lab' or m.startswith('runner.offline')))")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
