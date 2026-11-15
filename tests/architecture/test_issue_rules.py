"""ADR-0014: trading behaviour never depends on the wording of a problem. In trading/application no decision
searches text: no ``"constant" in x`` membership test, no ``startswith``/``endswith``/``find``/``index``/
``count`` call and no ``re`` module, except the listed uses that do not read problem text."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_SEARCH = {"startswith", "endswith", "find", "index", "count", "rfind", "rindex"}
ALLOWED = {
    # The n-th flatten attempt of an intent: a prefix of our own intent identifiers, not problem text.
    ("trading/application/execution/flatten.py", "str(r['intent_id']).startswith(prefix)"),
}


def _text_searches(tree: ast.AST, path: str) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) \
                and isinstance(node.left.value, str) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            found.append(f"{path}:{node.lineno}: {ast.unparse(node)}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            text = ast.unparse(node)
            if node.func.attr in TEXT_SEARCH and (path, text) not in ALLOWED:
                found.append(f"{path}:{node.lineno}: {text}")
            elif isinstance(node.func.value, ast.Name) and node.func.value.id == "re":
                found.append(f"{path}:{node.lineno}: {text}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if "re" in names:
                found.append(f"{path}:{node.lineno}: import re")
    return found


def test_no_trading_decision_in_the_application_layer_reads_problem_text():
    found = []
    for path in sorted((ROOT / "trading" / "application").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        found += _text_searches(ast.parse(path.read_text(encoding="utf-8")), rel)
    assert found == []


def test_the_rule_catches_what_it_forbids():
    code = ('if any("stop_breached" in i for i in issues): pass\n'
            'if issue.endswith(UNRESOLVED): pass\n'
            'if "breached" not in cause: pass\n'
            'import re\n'
            'm = re.search("x", issue)\n'
            'if status in SETTLED: pass\n')                          # a name, not text: allowed
    found = _text_searches(ast.parse(code), "x.py")
    assert sorted(int(line.split(":")[1]) for line in found) == [1, 2, 3, 4, 5]
