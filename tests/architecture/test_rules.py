"""Architecture rules that import-linter cannot express (see .importlinter and docs/arch/target.md 2.3)."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRADING = ROOT / "trading"

# Test directories that guard the money path; CI and `make test` must run all of them under pytest.
REQUIRED_GATED = ("tests/architecture/", "tests/characterization/", "tests/offline/", "tests/contract/",
                  "tests/unit/lab/", "tests/unit/binance/", "tests/unit/trading/")


def _modules(base: Path):
    for path in sorted(base.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


ENV_NAMES = {"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"}


def _env_reads(tree: ast.AST) -> list[int]:
    """Any use of an environment accessor, whatever the module alias (``import os as o; o.environ``)."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ENV_NAMES:
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "os" \
                and any(alias.name in ENV_NAMES for alias in node.names):
            lines.append(node.lineno)
    return sorted(lines)


def _in_infrastructure(path: Path) -> bool:
    return path.relative_to(TRADING).parts[0] == "infrastructure"


def test_only_infrastructure_reads_the_environment():
    offenders = [f"{path.relative_to(ROOT)}:{line}"
                 for path, tree in _modules(TRADING) if not _in_infrastructure(path)
                 for line in _env_reads(tree)]
    assert offenders == []


def test_env_rule_detects_reads():
    tree = ast.parse("import os\nfrom os import environ\nx = os.environ.get('A')\ny = os.getenv('B')\n"
                     "import os as o\nz = o.environ\nw = o.getenvb(b'C')\n")
    assert _env_reads(tree) == [2, 3, 4, 6, 7]


IO_CALLS = {"open", "__import__", "exec", "eval", "compile", "input", "breakpoint"}
IO_METHODS = {"read_text", "write_text", "read_bytes", "write_bytes", "open", "now", "utcnow", "today",
              "touch", "unlink", "mkdir", "rename", "replace"}


def _io_uses(tree: ast.AST) -> list[str]:
    """I/O, clock reads and dynamic imports that need no forbidden import (``open``, ``Path.write_text``)."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in IO_CALLS:
            found.append(f"{node.func.id}()@{node.lineno}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in IO_METHODS:
            found.append(f".{node.func.attr}()@{node.lineno}")
    return found


def test_domain_and_application_do_no_io_without_imports_either():
    offenders = [f"{path.relative_to(ROOT)}: {use}"
                 for layer in ("domain", "application") for path, tree in _modules(TRADING / layer)
                 for use in _io_uses(tree)]
    assert offenders == []


def test_io_rule_detects_import_free_io():
    tree = ast.parse("open('f')\n__import__('sqlite3')\np.write_text('x')\ndatetime.now()\n")
    assert _io_uses(tree) == ["open()@1", "__import__()@2", ".write_text()@3", ".now()@4"]


def _tokens(block: str) -> list[str]:
    return [t for line in block.splitlines() for t in line.replace("\\", " ").split()]


def _gated(tokens: list[str], directory: str) -> bool:
    ignored = any(t.startswith("--ignore") or t.startswith("--deselect") for t in tokens)
    return directory in tokens and not ignored


def _ci_core_test_paths() -> str:
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    block = text.split("name: Run core unit tests", 1)[1]
    return block.split("- name:", 1)[0]


def _make_core_tests() -> str:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    return re.search(r"^CORE_TESTS =(.*?)^\S", text, re.S | re.M).group(1)


def test_money_path_tests_are_gated_in_ci_and_make():
    ci, make = _tokens(_ci_core_test_paths()), _tokens(_make_core_tests())
    assert [d for d in REQUIRED_GATED if not _gated(ci, d)] == []
    assert [d for d in REQUIRED_GATED if not _gated(make, d)] == []


def test_gate_check_rejects_ignored_directories():
    assert not _gated(["pytest", "--ignore=tests/offline/", "tests/offline/"], "tests/offline/")
    assert not _gated(["pytest", "tests/offline/x.py"], "tests/offline/")


def test_ci_runs_the_import_contracts_and_the_type_check():
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert re.search(r"^\s*run:\s*lint-imports\s*$", text, re.M) and (ROOT / ".importlinter").is_file()
    assert re.search(r"^\s*run:\s*mypy trading/\s*$", text, re.M)


def _is_any(node: ast.AST) -> bool:
    """Any at the top of an annotation: Any, t.Any, "Any", Optional[Any], Union[..., Any], Any | None.

    Any nested inside a container (``dict[str, Any]`` for a JSON payload) is not a collaborator type.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return False
    if isinstance(node, ast.Name):
        return node.id == "Any"
    if isinstance(node, ast.Attribute):
        return node.attr == "Any"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_any(node.left) or _is_any(node.right)
    if isinstance(node, ast.Subscript) and ast.unparse(node.value).split(".")[-1] in {"Optional", "Union"}:
        members = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return any(_is_any(m) for m in members)
    return False


def _any_parameters(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            found += [f"{node.name}({a.arg})" for a in params if a.annotation is not None and _is_any(a.annotation)]
    return found


def test_application_parameters_are_typed_by_ports_not_any():
    offenders = [f"{path.relative_to(ROOT)}: {name}"
                 for path, tree in _modules(TRADING / "application") for name in _any_parameters(tree)]
    assert offenders == []


def test_any_rule_detects_untyped_collaborators():
    tree = ast.parse("class G:\n    def __init__(self, a: Any, b: 'Any', c: Optional[Any], d: t.Any, e: Any | None,"
                     " n: int, p: dict[str, Any], **kw: Any): ...\n")
    assert _any_parameters(tree) == ["__init__(a)", "__init__(b)", "__init__(c)", "__init__(d)", "__init__(e)"]


def test_the_hard_leverage_cap_is_defined_once():
    pattern = re.compile(r"^\s*MAX_TOTAL_LEVERAGE\s*=", re.M)
    sources = [p for p in ROOT.rglob("*.py") if not {".git", ".claude", "target"} & set(p.relative_to(ROOT).parts)]
    defined = [str(p.relative_to(ROOT)) for p in sources
               if pattern.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert defined == ["trading/domain/risk/exposure.py"]


def test_binance_order_paths_use_the_status_table_not_literal_sets():
    paths = ("shared/confirmed_execution.py", "runner/binance_target.py", "runner/binance_main.py")
    assert [p for p in paths if '"mmp_canceled"' in (ROOT / p).read_text(encoding="utf-8")] == []


def test_only_the_sqlite_adapter_writes_order_tables_on_the_binance_path():
    paths = [ROOT / "shared/confirmed_execution.py", *sorted((ROOT / "execution/adapters/binance").glob("*.py"))]
    assert [str(p.relative_to(ROOT)) for p in paths if ".execute(" in p.read_text(encoding="utf-8")] == []
