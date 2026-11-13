"""Protective stops, 3x exposure cap, kill file and peak drawdown."""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
from execution.adapters.binance.protective_stops import ProtectiveStopManager
from shared.drawdown_guard import DrawdownGuard
from shared.exposure_gate import MAX_TOTAL_LEVERAGE, AccountExposureGate
from shared.kill_file import KillFile
from tests.unit.binance.fake_venue import FakeBinance, make_bars

OWNER = "alpha.BTCUSDT"


@pytest.fixture
def env(tmp_path):
    clock = {"now": 1_800_000_000.0}
    fake = FakeBinance(make_bars(10), clock=lambda: clock["now"])
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    executor = BinanceExecutionAdapter(adapter, scope="binance-testnet-ut", owner=OWNER,
                                       journal_path=tmp_path / "orders.sqlite", wait_seconds=0,
                                       clock=lambda: clock["now"])
    stops = ProtectiveStopManager(adapter, executor, "BTCUSDT", stop_pct=D("0.04"), clock=lambda: clock["now"])
    return fake, adapter, executor, stops, clock


def _checkpoint(executor, qty, entry="100", revision="r1"):
    with executor.journal.connect() as db:
        db.execute("INSERT OR REPLACE INTO execution_consumers VALUES (?,?,?)",
                   (executor.scope, OWNER, json.dumps({"position": {"qty": str(qty), "entry_price": entry,
                                                                    "realized_pnl": "0", "fees": {}},
                                                       "revision": revision, "engine": {}, "strategy": {}})))


def _live_stops(fake):
    return {k: v for k, v in fake.algo.items() if v["algoStatus"] == "NEW"}


def test_stop_follows_position_lifecycle(env):
    fake, _, executor, stops, _ = env
    assert stops.sync() == [] and not fake.algo  # flat: nothing placed
    _checkpoint(executor, "0.1")
    stops.sync()
    (stop,) = _live_stops(fake).values()
    assert stop["side"] == "SELL" and D(stop["triggerPrice"]) == D("96")
    stops.sync()
    assert len(_live_stops(fake)) == 1  # idempotent
    _checkpoint(executor, "0.2", entry="101", revision="r2")  # added to the position
    stops.sync()
    (stop2,) = _live_stops(fake).values()
    assert stop2["clientAlgoId"] != stop["clientAlgoId"] and D(stop2["triggerPrice"]) == D("96.9")
    _checkpoint(executor, "0", entry="0", revision="r3")
    stops.sync()
    assert not _live_stops(fake)  # flat again: stale stop canceled


def test_short_stop_and_breach_reporting(env):
    fake, _, executor, stops, _ = env
    _checkpoint(executor, "-0.1")
    stops.sync()
    (stop,) = _live_stops(fake).values()
    assert stop["side"] == "BUY" and D(stop["triggerPrice"]) == D("104")
    _checkpoint(executor, "-0.1", entry="90", revision="r2")  # price already above 93.6
    issues = stops.sync()
    assert any("stop_breached" in issue for issue in issues)


def test_triggered_stop_fills_reach_the_journal_once(env):
    fake, _, executor, stops, _ = env
    fake.positions["BTCUSDT"], fake.entry["BTCUSDT"] = D("0.1"), D("100")
    _checkpoint(executor, "0.1")
    stops.sync()
    (client_id,) = _live_stops(fake)
    fake.trigger_stop(client_id, D("95.9"))
    issues = stops.sync()
    assert any("triggered" in issue for issue in issues)
    fills = executor.reconcile_pending()
    assert [(f.side, f.qty) for f in fills] == [("sell", D("0.1"))]
    stops.sync()
    assert len([r for r in executor.journal.orders(OWNER) if r["intent_id"].startswith("stop-")]) == 1


def test_exposure_gate_caps_gross_notional(env):
    fake, adapter, executor, _, _ = env
    gate = AccountExposureGate(adapter, executor.journal)
    fake.equity = D("1000")
    assert gate.check_would_exceed(symbol="BTCUSDT", venue="binance", side="buy", proposed_notional_usd=2900)[0]
    fake.positions["BTCUSDT"] = D("10")  # 1000 notional already open
    allowed, reason = gate.check_would_exceed(symbol="BTCUSDT", venue="binance", side="buy",
                                              proposed_notional_usd=2100)
    assert not allowed and "3" in reason
    fake.position_error = True
    assert gate.check_would_exceed(symbol="BTCUSDT", venue="binance", side="buy", proposed_notional_usd=1)[0] is False
    with pytest.raises(ValueError):
        AccountExposureGate(adapter, max_leverage=MAX_TOTAL_LEVERAGE + 1)


def test_kill_file_fails_closed(tmp_path):
    kill = KillFile(tmp_path / "kill.json")
    assert kill.allow_order(symbol="BTCUSDT")[0] is True
    kill.write(active=True, reason="manual", flatten=True)
    state = kill.state()
    assert state.active and state.flatten and kill.allow_order(symbol="BTCUSDT")[0] is False
    kill.write(active=False)
    assert kill.allow_order(symbol="BTCUSDT")[0] is True
    (tmp_path / "kill.json").write_text("{broken")
    state = kill.state()
    assert state.active and not state.flatten


def test_drawdown_measured_from_period_peak(tmp_path):
    guard = DrawdownGuard(tmp_path / "dd.sqlite", "binance-testnet-ut", daily_pct=3, weekly_pct=10)
    t = 1_800_000_000.0
    assert guard.observe(1000, now=t)[0]
    assert guard.observe(1100, now=t + 60)[0]           # new high-water mark
    allowed, reason = guard.observe(1060, now=t + 120)  # -3.6% from peak, +6% from first equity
    assert not allowed and "latched" in reason and guard.latched(now=t + 120)
    assert not guard.observe(1200, now=t + 180)[0]      # latched until the period ends
