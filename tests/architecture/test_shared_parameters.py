"""What research, paper and execution share is defined once (ADR-0013 item 5, design 2026-11 M5 step S10).

The maker limit offset - how far inside its reference price a maker entry rests - was written out four times
(the research broker's default, lab/simulate.py, lab/breakout_study.py and the execution wiring), equal only by
convention. It is defined once, as ``trading.domain.orders.LIMIT_OFFSET_BPS``; every other place refers to it.

Rule: in production code nothing whose name contains ``limit_offset`` (any case) gets a literal value, except that
one definition. A name gets a value in an assignment, a dataclass field, a keyword argument or a parameter
default. A literal is a number or string constant, possibly wrapped in ``Decimal``/``D``/``str``/``float``/``int``
and arithmetic, or a module-level name bound to one.

The research broker and its research settings (``BarBroker``, ``ReplayConfig``, ``BrokerFill``, ``SymbolRules``,
``SYMBOL_RULES``) are defined once too, in trading/adapters/simulated/bar_broker.py; research and paper import
them from there. ``broker_config`` is not in that list: the closed breakout study keeps its own costs.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ("trading", "runner", "lab", "data", "infra", "monitoring", "scripts")
OFFSET = re.compile(r"limit_?offset", re.IGNORECASE)
WRAPPERS = {"Decimal", "D", "str", "float", "int"}
THE_OFFSET = ("trading/domain/orders.py", "LIMIT_OFFSET_BPS")
BROKER_HOME = "trading/adapters/simulated/bar_broker.py"
BROKER_NAMES = ("BarBroker", "ReplayConfig", "BrokerFill", "SymbolRules", "SYMBOL_RULES")


def _production():
    for top in PRODUCTION:
        for path in sorted((ROOT / top).rglob("*.py")):
            yield path.relative_to(ROOT).as_posix(), ast.parse(path.read_text(encoding="utf-8"))


def _literal(node: ast.AST, constants: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str)) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id in constants
    if isinstance(node, ast.UnaryOp):
        return _literal(node.operand, constants)
    if isinstance(node, ast.BinOp):
        return _literal(node.left, constants) and _literal(node.right, constants)
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        return name in WRAPPERS and len(node.args) == 1 and not node.keywords and _literal(node.args[0], constants)
    return False


def _name(target: object) -> str:
    if isinstance(target, str):
        return target
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.arg):
        return target.arg
    return ""


def _bindings(node: ast.AST) -> list[tuple[object, ast.AST]]:
    """(target, value) of every name the node gives a value."""
    if isinstance(node, ast.Assign):
        pairs: list[tuple[object, ast.AST]] = []
        for target in node.targets:
            if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple) \
                    and len(target.elts) == len(node.value.elts):
                pairs += list(zip(target.elts, node.value.elts))
            else:
                pairs.append((target, node.value))
        return pairs
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return [(node.target, node.value)]
    if isinstance(node, ast.keyword) and node.arg:
        return [(node.arg, node.value)]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = node.args
        positional = args.posonlyargs + args.args
        pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
        return pairs + [(arg, value) for arg, value in zip(args.kwonlyargs, args.kw_defaults) if value is not None]
    return []


def _offset_literals(tree: ast.Module) -> list[tuple[str, int]]:
    """(name, line) of every limit offset given a literal value."""
    constants = {target.id for stmt in tree.body if isinstance(stmt, ast.Assign) and _literal(stmt.value, set())
                 for target in stmt.targets if isinstance(target, ast.Name)}
    found = set()
    for node in ast.walk(tree):
        for target, value in _bindings(node):
            name = _name(target)
            if OFFSET.search(name) and _literal(value, constants):
                found.add((name, value.lineno))
    return sorted(found, key=lambda site: (site[1], site[0]))


def _defined(tree: ast.Module) -> set[str]:
    """The broker names this module defines at its top level (a class or an assignment; an import is not)."""
    names = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            names |= {target.id for target in stmt.targets if isinstance(target, ast.Name)}
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
    return names & set(BROKER_NAMES)


def test_the_limit_offset_is_defined_once():
    sites = {(rel, name): line for rel, tree in _production() for name, line in _offset_literals(tree)}
    assert set(sites) == {THE_OFFSET}, sites


def test_the_research_broker_and_its_research_settings_are_defined_once():
    homes: dict[str, list[str]] = {}
    for rel, tree in _production():
        for name in _defined(tree):
            homes.setdefault(name, []).append(rel)
    assert homes == {name: [BROKER_HOME] for name in BROKER_NAMES}


def test_the_offset_rule_finds_every_way_of_writing_a_literal():
    source = ("import decimal\n"
              "from decimal import Decimal as D\n"
              "OFFSET = D('0.0002')\n"
              "LIMIT_OFFSET = D('0.0002')\n"
              "class Config:\n"
              "    limit_offset_bps: str = '2'\n"
              "def wire(limit_offset=decimal.Decimal(2) / 10000, *, maker_limit_offset=-0.0002):\n"
              "    service(limit_offset=OFFSET)\n"
              "    config(limit_offset_bps=str(-D(2)))\n"
              "    self.limit_offset = 0.0002\n"
              "    entry, LimitOffset = 1, 2\n"
              "check = lambda limitoffset='2': limitoffset\n")
    assert _offset_literals(ast.parse(source)) == [
        ("LIMIT_OFFSET", 4), ("limit_offset_bps", 6), ("limit_offset", 7), ("maker_limit_offset", 7),
        ("limit_offset", 8), ("limit_offset_bps", 9), ("limit_offset", 10), ("LimitOffset", 11),
        ("limitoffset", 12)]


def test_the_offset_rule_allows_references_to_the_definition():
    source = ("from trading.domain.orders import LIMIT_OFFSET_BPS\n"
              "LIMIT_OFFSET = LIMIT_OFFSET_BPS / 10000\n"
              "config(limit_offset_bps=str(LIMIT_OFFSET_BPS), maker_bps='2')\n"
              "service(limit_offset=LIMIT_OFFSET, other=spec.limit_offset)\n"
              "def f(limit_offset=None, offset_enabled=True): ...\n"
              "limit = price * (1 - sign * D(config.limit_offset_bps) / 10000)\n")
    assert _offset_literals(ast.parse(source)) == []


def test_the_definition_rule_counts_classes_and_assignments_but_not_imports():
    source = ("from trading.adapters.simulated.bar_broker import BarBroker, SYMBOL_RULES\n"
              "class ReplayConfig: ...\n"
              "SymbolRules: type = dict\n"
              "BrokerFill = SYMBOL_RULES = None\n"
              "def BarBroker_factory(): ...\n")
    assert _defined(ast.parse(source)) == {"ReplayConfig", "SymbolRules", "BrokerFill", "SYMBOL_RULES"}
