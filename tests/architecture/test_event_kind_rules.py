"""ADR-0014 point 2: an event is written with an ``EventKind``, never a string. A call ``x.write("word", key=...)``
(a string first argument and keyword fields, the shape of an event write) is forbidden in production code."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ("trading", "runner", "lab")


def _string_kinds(tree: ast.AST, path: str) -> list[str]:
    return [f"{path}:{node.lineno}: {ast.unparse(node)[:80]}" for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write"
            and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
            and node.keywords]


def test_no_event_is_written_with_a_string_kind():
    found = []
    for package in PRODUCTION:
        for path in sorted((ROOT / package).rglob("*.py")):
            rel = str(path.relative_to(ROOT))
            found += _string_kinds(ast.parse(path.read_text(encoding="utf-8")), rel)
    assert found == []


def test_the_rule_catches_what_it_forbids():
    code = ('journal.write("intent", symbol=s)\n'
            'self.events.write("alert", reason=r)\n'
            'journal.write(EventKind.INTENT, symbol=s)\n'      # allowed
            'stdout.write("done\\n")\n')                         # not an event: no fields
    assert sorted(int(f.split(":")[1]) for f in _string_kinds(ast.parse(code), "x.py")) == [1, 2]
