"""Architecture rules that import-linter cannot express (see .importlinter and docs/arch/target.md 2.3)."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRADING = ROOT / "trading"

# Test directories that guard the money path; CI and `make test` must run all of them under pytest.
REQUIRED_GATED = ("tests/architecture/", "tests/characterization/", "tests/offline/", "tests/contract/",
                  "tests/unit/lab/", "tests/unit/binance/")


def _modules(base: Path):
    for path in sorted(base.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _env_reads(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv", "putenv", "environb"} \
                and isinstance(node.value, ast.Name) and node.value.id == "os":
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "os" \
                and any(alias.name in {"environ", "getenv", "environb"} for alias in node.names):
            lines.append(node.lineno)
    return sorted(lines)


def test_only_infrastructure_reads_the_environment():
    offenders = [f"{path.relative_to(ROOT)}:{line}"
                 for path, tree in _modules(TRADING) if "infrastructure" not in path.relative_to(TRADING).parts
                 for line in _env_reads(tree)]
    assert offenders == []


def test_env_rule_detects_reads():
    tree = ast.parse("import os\nfrom os import environ\nx = os.environ.get('A')\ny = os.getenv('B')\n")
    assert _env_reads(tree) == [2, 3, 4]


def _ci_core_test_paths() -> str:
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    block = text.split("name: Run core unit tests", 1)[1]
    return block.split("- name:", 1)[0]


def _make_core_tests() -> str:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    return re.search(r"^CORE_TESTS =(.*?)^\S", text, re.S | re.M).group(1)


def test_money_path_tests_are_gated_in_ci_and_make():
    ci, make = _ci_core_test_paths(), _make_core_tests()
    assert [d for d in REQUIRED_GATED if d not in ci] == []
    assert [d for d in REQUIRED_GATED if d not in make] == []


def test_ci_runs_the_import_contracts():
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "lint-imports" in text and (ROOT / ".importlinter").is_file()


def _any_parameters(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            found += [f"{node.name}({a.arg})" for a in params
                      if a.annotation is not None and ast.unparse(a.annotation) in {"Any", "typing.Any"}]
    return found


def test_application_parameters_are_typed_by_ports_not_any():
    offenders = [f"{path.relative_to(ROOT)}: {name}"
                 for path, tree in _modules(TRADING / "application") for name in _any_parameters(tree)]
    assert offenders == []


def test_any_rule_detects_untyped_collaborators():
    tree = ast.parse("from typing import Any\nclass G:\n    def __init__(self, limiter: Any, n: int, **kw: Any): ...\n")
    assert _any_parameters(tree) == ["__init__(limiter)"]


def test_the_hard_leverage_cap_is_defined_once():
    pattern = re.compile(r"^\s*MAX_TOTAL_LEVERAGE\s*=", re.M)
    sources = [p for p in ROOT.rglob("*.py") if not {".git", ".claude", "target"} & set(p.relative_to(ROOT).parts)]
    defined = [str(p.relative_to(ROOT)) for p in sources
               if pattern.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert defined == ["trading/domain/risk/exposure.py"]
