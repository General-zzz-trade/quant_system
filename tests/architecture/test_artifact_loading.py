"""Model and spec artifacts load only through infra/model_signing.py (CLAUDE.md "Signed artifacts"; design 2026-11
M5, invariant T2).

- No production module deserializes with pickle or a pickle-based library: a pickle is code, and the signed
  loader for it was deleted with the last pickle artifact (2026-11-15).
- The runtime's strategy spec is read only by ``load_spec``, and ``load_spec`` reads only through
  ``load_verified_json``; the ``spec_path`` argument flows nowhere else.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ("trading", "runner", "lab", "data", "infra", "monitoring", "scripts")
DESERIALIZERS = {"pickle", "cPickle", "_pickle", "dill", "cloudpickle", "joblib", "marshal", "shelve"}
PICKLE_CALLS = {"read_pickle", "load_pickle"}
RAW_READS = {"open", "read_text", "read_bytes", "load", "loads"}
SPEC_SINKS = {"load_spec", "preflight", "TargetRuntime"}


def _deserializers(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] in DESERIALIZERS for a in node.names):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in DESERIALIZERS:
            lines.append(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if name in PICKLE_CALLS or (name == "load" and any(
                    k.arg == "allow_pickle" and not (isinstance(k.value, ast.Constant) and k.value.value is False)
                    for k in node.keywords)):
                lines.append(node.lineno)
    return sorted(lines)


def _call_name(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def _spec_path_leaks(tree: ast.AST) -> list[int]:
    """Uses of a ``spec_path`` name (or ``args.spec``) other than as an argument to a spec sink."""
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in SPEC_SINKS:
            for value in [*node.args, *(k.value for k in node.keywords)]:
                allowed.add(id(value))
    leaks = []
    for node in ast.walk(tree):
        is_spec = (isinstance(node, ast.Name) and node.id == "spec_path" and isinstance(node.ctx, ast.Load)) or \
                  (isinstance(node, ast.Attribute) and node.attr == "spec" and isinstance(node.value, ast.Name)
                   and node.value.id == "args")
        if is_spec and id(node) not in allowed:
            leaks.append(node.lineno)
    return sorted(leaks)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_no_production_module_deserializes_with_pickle():
    found = []
    for package in PRODUCTION:
        for path in sorted((ROOT / package).rglob("*.py")):
            lines = _deserializers(ast.parse(path.read_text(encoding="utf-8")))
            found += [f"{path.relative_to(ROOT)}:{line}" for line in lines]
    assert found == []


def test_the_runtime_reads_its_spec_only_through_the_signed_loader():
    tree = ast.parse((ROOT / "runner" / "binance_target.py").read_text(encoding="utf-8"))
    calls = {_call_name(n) for n in ast.walk(_function(tree, "load_spec")) if isinstance(n, ast.Call)}
    assert "load_verified_json" in calls
    assert not calls & RAW_READS, f"load_spec reads the spec itself: {sorted(calls & RAW_READS)}"
    assert _spec_path_leaks(tree) == []


def test_the_rules_detect_what_they_forbid():
    assert _deserializers(ast.parse(
        "import pickle\nfrom joblib import load\nimport dill as d\nx = pd.read_pickle(p)\n"
        "y = np.load(p, allow_pickle=True)\nz = np.load(p, allow_pickle=False)\n")) == [1, 2, 3, 4, 5]
    assert _spec_path_leaks(ast.parse(
        "def f(spec_path):\n    load_spec(spec_path)\n    json.loads(spec_path.read_text())\n"
        "def g(args):\n    preflight(args.spec)\n    open(args.spec)\n")) == [3, 6]
