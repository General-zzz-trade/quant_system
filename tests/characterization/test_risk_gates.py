"""Entry risk gates as runner.binance_target composes them (written before migration step 4 moves them).

Each test kills a defect that survived the 2026-11-14 mutation run (docs/arch/current-state.md 5.5).
"""
from __future__ import annotations

import sqlite3
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from execution.safety.limits import OrderLimiter, OrderLimitsConfig
from lab.orders import make_order
from runner.binance_main import EntryReadinessGate
from shared.drawdown_guard import DrawdownGuard
from shared.entry_risk import CompositeRiskGate
from shared.execution_journal import ExecutionJournal
from shared.exposure_gate import AccountExposureGate
from shared.kill_file import KillFile
from tests.unit.binance.fake_venue import FakeBinance, make_bars

T0 = 1_794_700_000.0
SCOPE = "binance-testnet-ut"


def _order(qty="0.01", *, reduce_only=False, side="buy"):
    return make_order(order_id=f"BTCUSDT:{qty}:{side}:{reduce_only}", source="target.BTCUSDT", symbol="BTCUSDT",
                      side=side, qty=D(qty), price=D("100"), reduce_only=reduce_only, reason="rebalance")


def _gates(tmp_path, *, limits=None):
    fake = FakeBinance(make_bars(10), clock=lambda: T0)
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    journal = ExecutionJournal(tmp_path / "orders.sqlite", SCOPE)
    kill = KillFile(tmp_path / "kill_switch.json")
    drawdown = DrawdownGuard(tmp_path / "drawdown.sqlite", SCOPE, daily_pct=5, weekly_pct=12)
    drawdown.observe(10_000.0)
    limiter = OrderLimiter(limits or OrderLimitsConfig(max_order_notional=D("5000"), max_orders_per_second=1000.0))
    inner = CompositeRiskGate(limiter, kill_switch=kill, drawdown_guard=drawdown,
                              portfolio_reader=AccountExposureGate(adapter, journal), venue="binance")
    session = SimpleNamespace(entry_ready=True, entry_reason="")
    return SimpleNamespace(fake=fake, journal=journal, kill=kill, drawdown=drawdown, session=session,
                           gate=EntryReadinessGate(session, inner))


def test_composed_gate_allows_a_small_entry_and_any_reduction(tmp_path):
    g = _gates(tmp_path)
    assert g.gate.check(_order()).allowed
    g.kill.write(active=True, reason="test")
    assert g.gate.check(_order(reduce_only=True, side="sell")).allowed


@pytest.mark.parametrize("block,reason", [
    ("inputs", "inputs: bar processed too late"),
    ("kill", "kill_switch: test"),
    ("drawdown", "drawdown: equity refresh failed"),
    ("exposure", "portfolio_gate: gross exposure"),
    ("limiter", "max_order_notional"),
])
def test_composed_gate_names_the_first_failing_check(tmp_path, block, reason):
    g = _gates(tmp_path)
    qty = "0.01"
    if block == "inputs":
        g.session.entry_ready, g.session.entry_reason = False, "bar processed too late"
    elif block == "kill":
        g.kill.write(active=True, reason="test")
    elif block == "drawdown":
        g.drawdown.observe(float("nan"))
    elif block == "exposure":
        qty = "301"          # 30,100 USDT > 3 x 10,000 equity
    else:
        qty = "60"           # 6,000 USDT > 5,000 per-order cap, within 3x
    result = g.gate.check(_order(qty))
    assert not result.allowed and result.reason.startswith(reason)


def test_exposure_counts_unresolved_entry_intents(tmp_path):
    g = _gates(tmp_path)
    with g.journal.connect() as db:
        db.execute("INSERT INTO submissions (scope,intent_id,symbol,client_id,order_id,settled,side,qty,reducing,"
                   "owner,status) VALUES (?, 'pending-1', 'BTCUSDT', 'c', '', 0, 'buy', '250', 0, 'o', 'unknown')",
                   (SCOPE,))
    gate = AccountExposureGate(BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=g.fake), g.journal)
    assert gate.exposure() == (D("10000"), D("25000"))
    allowed, reason = gate.check_would_exceed(symbol="BTCUSDT", venue="binance", side="buy",
                                              proposed_notional_usd=6000.0)
    assert not allowed and "31000.00" in reason


def test_drawdown_latch_reads_as_latched_when_the_store_is_unreadable(tmp_path):
    guard = DrawdownGuard(tmp_path / "drawdown.sqlite", SCOPE)
    guard.observe(1000.0, now=T0)
    assert guard.latched(now=T0) is False
    (tmp_path / "drawdown.sqlite").write_bytes(b"not a database" * 100)
    assert guard.latched(now=T0) is True
    assert guard.check(now=T0)[0] is False


def test_a_wider_limit_after_restart_cannot_relax_the_current_period(tmp_path):
    path = tmp_path / "drawdown.sqlite"
    DrawdownGuard(path, SCOPE, daily_pct=3, weekly_pct=10).observe(1000.0, now=T0)
    wider = DrawdownGuard(path, SCOPE, daily_pct=20, weekly_pct=30)
    allowed, reason = wider.observe(965.0, now=T0 + 60)        # 3.5 % below the period peak
    assert not allowed and "day drawdown latched" in reason
    with sqlite3.connect(path) as db:
        assert {row[0] for row in db.execute("SELECT limit_pct FROM drawdown_periods WHERE kind='day'")} == {3.0}


@pytest.mark.parametrize("content", ['{"active": "yes"}', "[]", '{"active": 1}', "{broken"])
def test_kill_file_with_a_non_boolean_or_broken_active_flag_blocks_entries(tmp_path, content):
    path = tmp_path / "kill_switch.json"
    path.write_text(content)
    state = KillFile(path).state()
    assert state.active is True and state.flatten is False
    assert KillFile(path).allow_order(symbol="BTCUSDT")[0] is False
