"""Research code does not write its own fill model (design 2026-11 M5, step S12).

Rule: no module in lab/ multiplies by a trading-cost rate, that is, no ``*`` or ``*=`` either side of which
mentions a name or attribute containing fee, taker, maker, slip, commission, bps, cost, spread or impact (any
case). Fees, slippage and fills are charged only by the shared conservative broker (runner/offline/broker.py),
reached through lab.simulate (``simulate``, ``Sleeve``) or ``BarBroker`` itself; lab/simulate.py sets the
broker's rates. The rule catches a bar loop and a vectorized cost model alike.

The exceptions are the two fill models written before the shared broker could serve them. Each stays exactly as
it is; the list must match the offenders, so a new one fails and a removed one must leave the list.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COST = re.compile(r"fee|taker|maker|slip|commission|bps|cost|spread|impact", re.IGNORECASE)
SHARED = {"lab/simulate.py"}
EXCEPTIONS = {
    "lab/pair_study.py::simulate_pair": (
        "two-leg pair model of the closed ETH/BTC line; lab/candidates_forward.py replays the frozen forward "
        "candidate pair_ridge_pd_7d with it at the looks registered in research/forward_looks.jsonl "
        "(2027-02-13, 2027-05-13), so it must not change"),
    "lab/midfreq.py::simulate": (
        "cross-sectional 15-symbol 5-minute model of the closed mid-frequency line (2026-11-14); the shared "
        "broker simulates one symbol on hourly bars; the code only reproduces the recorded result"),
}


def _names(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            yield n.id
        elif isinstance(n, ast.Attribute):
            yield n.attr


def _units(tree: ast.Module):
    """(top-level function, ``Class.method``, class or ``<module>``, statement) for every top-level statement."""
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield stmt.name, stmt
        elif isinstance(stmt, ast.ClassDef):
            for item in stmt.body:
                is_method = isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                yield (f"{stmt.name}.{item.name}" if is_method else stmt.name), item
        else:
            yield "<module>", stmt


def _cost_products(tree: ast.Module) -> list[tuple[str, int]]:
    """(unit, line) of every multiplication by a cost rate."""
    found = set()
    for unit, stmt in _units(tree):
        for node in ast.walk(stmt):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                sides = (node.left, node.right)
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Mult):
                sides = (node.target, node.value)
            else:
                continue
            if any(COST.search(name) for side in sides for name in _names(side)):
                found.add((unit, node.lineno))
    return sorted(found)


def test_research_code_charges_costs_only_through_the_shared_broker():
    offenders: dict[str, list[int]] = {}
    for path in sorted((ROOT / "lab").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in SHARED:
            continue
        for unit, line in _cost_products(ast.parse(path.read_text(encoding="utf-8"))):
            offenders.setdefault(f"{rel}::{unit}", []).append(line)
    assert sorted(offenders) == sorted(EXCEPTIONS), offenders


def test_the_rule_detects_what_it_forbids():
    loop = ("def backtest(bars, sides, fee_rate, slippage):\n"
            "    cash, qty = 1000.0, 0.0\n"
            "    for bar, side in zip(bars, sides):\n"
            "        px = bar.open * (1 + slippage * side)\n"
            "        cash -= (side - qty) * px + abs(side - qty) * px * fee_rate\n"
            "        qty = side\n"
            "    return cash\n")
    vectorized = ("def equity(weights, returns, taker_bps=5):\n"
                  "    turnover = weights.diff().abs()\n"
                  "    return (1 + weights.shift() * returns - turnover * taker_bps / 1e4).cumprod()\n")
    method = ("class Book:\n"
              "    def fill(self, qty, price):\n"
              "        self.cash -= qty * price * self.config.fee\n"
              "        self.cash *= 1 - self.config.FUNDING_COST\n")
    module = "net = gross * (1 - 2 * SLIPPAGE_BPS / 1e4)\n"
    assert _cost_products(ast.parse(loop)) == [("backtest", 4), ("backtest", 5)]
    assert _cost_products(ast.parse(vectorized)) == [("equity", 3)]
    assert _cost_products(ast.parse(method)) == [("Book.fill", 3), ("Book.fill", 4)]
    assert _cost_products(ast.parse(module)) == [("<module>", 1)]


def test_the_rule_allows_the_shared_broker_and_its_results():
    shared = ("from lab.simulate import simulate\n"
              "def run(bars, targets, funding, spec):\n"
              "    result = simulate('BTCUSDT', bars, targets, funding, spec, start_ms=0, cost_multiplier=2)\n"
              "    return result.curve['equity'].iloc[-1] - result.curve['fees'].iloc[-1]\n")
    assert _cost_products(ast.parse(shared)) == []
