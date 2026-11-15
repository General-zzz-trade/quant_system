"""ADR-0017: in trading/ and runner/ only trading/infrastructure/config.py reads the environment (os.environ,
os.getenv, os.putenv or an ``environ`` imported from os). Everything else receives the RuntimeConfig."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READER = "trading/infrastructure/config.py"


def _environment_reads(tree: ast.AST, path: str) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os" \
                and node.attr in ("environ", "getenv", "putenv", "environb"):
            found.append(f"{path}:{node.lineno}: os.{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module == "os" \
                and any(a.name in ("environ", "getenv", "environb") for a in node.names):
            found.append(f"{path}:{node.lineno}: from os import ...")
    return found


def test_only_the_config_module_reads_the_environment():
    found = []
    for package in ("trading", "runner"):
        for path in sorted((ROOT / package).rglob("*.py")):
            rel = str(path.relative_to(ROOT))
            if rel != READER:
                found += _environment_reads(ast.parse(path.read_text(encoding="utf-8")), rel)
    assert found == []


def test_the_rule_catches_what_it_forbids():
    code = ("import os\nx = os.environ.get('A')\ny = os.getenv('B')\nfrom os import environ\n"
            "z = os.path.join('a', 'b')\n")                                      # os.path is fine
    assert sorted(int(f.split(":")[1]) for f in _environment_reads(ast.parse(code), "x.py")) == [2, 3, 4]
