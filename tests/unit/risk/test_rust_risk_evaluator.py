"""Tests for Phase 3: RustRiskEvaluator parity with Python risk rules.

Verifies that the Rust evaluator produces the same decisions as the
Python MaxPositionRule, LeverageCapRule, MaxDrawdownRule, and portfolio rules.
"""
from __future__ import annotations

import pytest

pytest.importorskip("_quant_hotpath")

from _quant_hotpath import RustRiskEvaluator


@pytest.fixture
def evaluator():
    return RustRiskEvaluator(
        max_position_qty=10.0,
        max_leverage=3.0,
        max_drawdown_pct=0.20,
        max_gross_leverage=3.0,
        max_net_leverage=1.0,
        max_concentration=0.4,
        allow_auto_reduce=True,
        dd_action="kill",
    )


def _rules(results):
    return {r.rule for r in results}


def _actions(results):
    return {r.rule: r.action for r in results}


# ── Max Position Rule ──

def test_position_within_limit(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=2.0, delta_qty=3.0, is_reducing=False,
        equity=100000.0, gross_notional=200.0, net_notional=200.0,
        peak_equity=100000.0, price=100.0, sym_notional=200.0,
    )
    assert "max_position" not in _rules(r)


def test_position_exceeds_limit_reduce(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=8.0, delta_qty=5.0, is_reducing=False,
        equity=100000.0, gross_notional=800.0, net_notional=800.0,
        peak_equity=100000.0, price=100.0, sym_notional=800.0,
    )
    pos_r = [v for v in r if v.rule == "max_position"]
    assert len(pos_r) == 1
    assert pos_r[0].action == "reduce"
    assert pos_r[0].max_qty == pytest.approx(2.0)  # headroom = 10 - 8 = 2


def test_position_exceeds_no_headroom(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=11.0, delta_qty=1.0, is_reducing=False,
        equity=100000.0, gross_notional=1100.0, net_notional=1100.0,
        peak_equity=100000.0, price=100.0, sym_notional=1100.0,
    )
    pos_r = [v for v in r if v.rule == "max_position"]
    assert len(pos_r) == 1
    assert pos_r[0].action == "reject"  # no headroom


def test_position_reducing_skips(evaluator):
    """Reducing exposure bypasses max_position check."""
    r = evaluator.evaluate_order(
        cur_qty=15.0, delta_qty=-5.0, is_reducing=True,
        equity=100000.0, gross_notional=1500.0, net_notional=1500.0,
        peak_equity=100000.0, price=100.0, sym_notional=1500.0,
    )
    assert "max_position" not in _rules(r)


# ── Leverage Cap Rule ──

def test_leverage_within_limit(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=1.0, is_reducing=False,
        equity=10000.0, gross_notional=20000.0, net_notional=20000.0,
        peak_equity=10000.0, price=100.0, sym_notional=20000.0,
    )
    assert "leverage_cap" not in _rules(r)


def test_leverage_exceeds_cap_reduce(evaluator):
    """Leverage exceeds 3x → REDUCE with max_qty."""
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=100.0, is_reducing=False,
        equity=10000.0, gross_notional=25000.0, net_notional=25000.0,
        peak_equity=10000.0, price=100.0, sym_notional=25000.0,
    )
    lev_r = [v for v in r if v.rule == "leverage_cap"]
    assert len(lev_r) == 1
    assert lev_r[0].action == "reduce"
    # headroom = 3*10000 - 25000 = 5000, max_qty = 5000/100 = 50
    assert lev_r[0].max_qty == pytest.approx(50.0)


def test_leverage_reducing_skips(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=500.0, delta_qty=-100.0, is_reducing=True,
        equity=10000.0, gross_notional=50000.0, net_notional=50000.0,
        peak_equity=10000.0, price=100.0, sym_notional=50000.0,
    )
    assert "leverage_cap" not in _rules(r)


# ── Max Drawdown Rule ──

def test_drawdown_within_limit(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=1.0, is_reducing=False,
        equity=9000.0, gross_notional=0.0, net_notional=0.0,
        peak_equity=10000.0, price=100.0, sym_notional=0.0,
    )
    assert "max_drawdown" not in _rules(r)


def test_drawdown_exceeds_kill(evaluator):
    """Drawdown 30% > 20% limit → KILL."""
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=1.0, is_reducing=False,
        equity=7000.0, gross_notional=0.0, net_notional=0.0,
        peak_equity=10000.0, price=100.0, sym_notional=0.0,
    )
    dd_r = [v for v in r if v.rule == "max_drawdown"]
    assert len(dd_r) == 1
    assert dd_r[0].action == "kill"


def test_drawdown_reducing_skips(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=10.0, delta_qty=-5.0, is_reducing=True,
        equity=7000.0, gross_notional=1000.0, net_notional=1000.0,
        peak_equity=10000.0, price=100.0, sym_notional=1000.0,
    )
    assert "max_drawdown" not in _rules(r)


# ── Gross Exposure Rule ──

def test_gross_exposure_within_limit(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=10.0, is_reducing=False,
        equity=10000.0, gross_notional=20000.0, net_notional=20000.0,
        peak_equity=10000.0, price=100.0, sym_notional=20000.0,
    )
    assert "gross_exposure" not in _rules(r)


def test_gross_exposure_exceeds_reduce(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=100.0, is_reducing=False,
        equity=10000.0, gross_notional=25000.0, net_notional=25000.0,
        peak_equity=10000.0, price=100.0, sym_notional=25000.0,
    )
    ge_r = [v for v in r if v.rule == "gross_exposure"]
    assert len(ge_r) == 1
    assert ge_r[0].action == "reduce"
    assert ge_r[0].max_qty == pytest.approx(50.0)


# ── Net Exposure Rule ──

def test_net_exposure_within_limit(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=50.0, is_reducing=False,
        equity=10000.0, gross_notional=0.0, net_notional=0.0,
        peak_equity=10000.0, price=100.0, sym_notional=0.0,
    )
    assert "net_exposure" not in _rules(r)


def test_net_exposure_exceeds_reject(evaluator):
    """Net exposure > 1x → REJECT (no auto-reduce for net)."""
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=200.0, is_reducing=False,
        equity=10000.0, gross_notional=0.0, net_notional=0.0,
        peak_equity=10000.0, price=100.0, sym_notional=0.0,
    )
    ne_r = [v for v in r if v.rule == "net_exposure"]
    assert len(ne_r) == 1
    assert ne_r[0].action == "reject"


# ── Concentration Rule ──

def test_concentration_within_limit(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=1.0, is_reducing=False,
        equity=100000.0, gross_notional=10000.0, net_notional=5000.0,
        peak_equity=100000.0, price=100.0, sym_notional=2000.0,
    )
    assert "concentration" not in _rules(r)


def test_concentration_exceeds_reject(evaluator):
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=1.0, is_reducing=False,
        equity=100000.0, gross_notional=1000.0, net_notional=1000.0,
        peak_equity=100000.0, price=100.0, sym_notional=800.0,
    )
    conc_r = [v for v in r if v.rule == "concentration"]
    assert len(conc_r) == 1
    assert conc_r[0].action == "reject"


# ── Quick Check Methods ──

def test_check_drawdown(evaluator):
    assert evaluator.check_drawdown(7000.0, 10000.0) is True   # 30% > 20%
    assert evaluator.check_drawdown(9000.0, 10000.0) is False  # 10% < 20%
    assert evaluator.check_drawdown(10000.0, 10000.0) is False  # 0% < 20%
    assert evaluator.check_drawdown(5000.0, 0.0) is False       # no peak


def test_check_leverage(evaluator):
    assert evaluator.check_leverage(40000.0, 10000.0) is True   # 4x > 3x
    assert evaluator.check_leverage(20000.0, 10000.0) is False  # 2x < 3x
    assert evaluator.check_leverage(30000.0, 10000.0) is False  # 3x = 3x (not >)


# ── Config Variations ──

def test_no_auto_reduce():
    ev = RustRiskEvaluator(
        max_position_qty=10.0,
        allow_auto_reduce=False,
    )
    r = ev.evaluate_order(
        cur_qty=8.0, delta_qty=5.0, is_reducing=False,
        equity=100000.0, gross_notional=0.0, net_notional=0.0,
        peak_equity=100000.0, price=100.0, sym_notional=0.0,
    )
    pos_r = [v for v in r if v.rule == "max_position"]
    assert pos_r[0].action == "reject"  # no auto-reduce → reject


def test_dd_action_reject():
    ev = RustRiskEvaluator(max_drawdown_pct=0.10, dd_action="reject")
    r = ev.evaluate_order(
        cur_qty=0.0, delta_qty=1.0, is_reducing=False,
        equity=8000.0, gross_notional=0.0, net_notional=0.0,
        peak_equity=10000.0, price=100.0, sym_notional=0.0,
    )
    dd_r = [v for v in r if v.rule == "max_drawdown"]
    assert dd_r[0].action == "reject"


# ── All ALLOW scenario ──

def test_all_rules_allow(evaluator):
    """Small order, healthy account → no violations."""
    r = evaluator.evaluate_order(
        cur_qty=0.0, delta_qty=1.0, is_reducing=False,
        equity=100000.0, gross_notional=10000.0, net_notional=5000.0,
        peak_equity=100000.0, price=100.0, sym_notional=2000.0,
    )
    assert len(r) == 0
