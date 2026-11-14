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
