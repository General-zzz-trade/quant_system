"""The execution path decides what the research simulation decides (backtest/live parity, migration step 10).

Both are fed the same bars and funding. Before each bar the account equity is set to the research
sleeve's equity, and the exchange fills every market order in full at once. Then, bar by bar, the orders
the execution path sends must equal the orders research sends, and the confirmed position must equal the
research position. A difference can then only come from the decision path: bar handling, the target,
the rebalance plan, the lot rules, the gates, the executor and the fill consumer.

Not compared, because research and the exchange differ there by design: fill price, fees and funding
cash (they enter through equity, which is forced equal), and the risk limits, which must not bind here.

Scope (audit after step 10b): one symbol, market entries, and a target that is ready on every bar. With
two symbols the account-wide limit of one order per second can refuse the second symbol's entry on the same
bar (known defect R23, pinned in tests/characterization/test_runtime_decisions.py); maker entries are compared
with research only through the paper tests. The funding cut-off cannot change the target in this spec
(no funding filter); it is tested in test_runtime_decisions.py.

Since 2026-11-14 execution uses the exchange's own minimum order size while research keeps its constant
(100 USDT for BTC). Every order in this run is above both, so the comparison is exact; the difference
below the research constant is pinned in test_target_composition.py.
"""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from lab.data import frame_from_bars, funding_series
from lab.simulate import SYMBOL_RULES, Sleeve, broker_config
from lab.strategy import StrategySpec, compute_targets
from runner.binance_target import TargetRuntime
from tests.unit.lab.test_target_runtime import LIVE, _advance, _world
from trading.domain.bars import bar_from_kline
from tests.support.state_db import state_dir_db

BALANCE = "2000"        # small enough that the default order limits never bind


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _events(tmp_path):
    path = tmp_path / "exec/target_events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_the_execution_path_sends_the_orders_research_sends_bar_by_bar(tmp_path):
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
    # Bars around 1,000 move about 1% per swing: enough for an order sized at the wrong price to differ by
    # a lot step, and small enough that the default drawdown limits never latch.
    world = _world(tmp_path, spec, bases=(("BTCUSDT", 1000.0), ("ETHUSDT", 2000.0)))
    world["spec_path"].write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target",
                                              "symbols": ["BTCUSDT"], "spec": spec.record(), "passed": False}))
    venue = world["fakes"]["BTCUSDT"]
    venue.equity = D(BALANCE)
    state_dir_db(tmp_path / "exec")   # a state database as init creates it (ADR-0009)
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"],
                            clock=lambda: world["clock"]["now"], testnet=True)
    runtime.warmup()
    session = runtime.sessions["BTCUSDT"]

    bars = [bar_from_kline(row) for row in world["rows"]["BTCUSDT"]]
    funding = {at: (D(str(rate)), D("1")) for at, rate in world["funding"]}
    targets = compute_targets(frame_from_bars(bars), funding_series(funding), spec)
    target_at = dict(zip(targets["available_at"].tolist(), zip(targets["target"].tolist(), targets["ready"].tolist())))
    rules = SYMBOL_RULES["BTCUSDT"]
    research = Sleeve("BTCUSDT", spec, rules, broker_config(spec, rules, starting_balance=BALANCE))

    entries = reductions = 0
    for i in range(world["warm"], world["warm"] + LIVE):
        bar = bars[i]
        research.on_bar(bar, funding[bar.open_time][0] if bar.open_time in funding else None)
        assert session.position_qty() == research.broker.position.qty, f"position differs before bar {i}"
        venue.equity = research.broker.equity
        target, ready = target_at.get(bar.available_at, (0.0, False))
        expected = [(o.side, o.qty, o.reduce_only) for o in research.decide(bar, target, ready)]
        _advance(world, runtime, i)
        # Live computes the target on a trailing window, research on the whole history: the two can differ
        # in the last bits of a float (seen at 5e-17 on more volatile bars), never by more.
        assert session.last_target == (pytest.approx(target, rel=1e-12, abs=1e-15), ready), f"target at bar {i}"
        sent = [(r["side"], D(r["qty"]), r["reduce_only"]) for r in _events(tmp_path)
                if r["kind"] == "intent" and r["order_id"].startswith(f"BTCUSDT:{bar.available_at}:")]
        assert sent == expected, f"orders differ at bar {i}"
        entries += sum(1 for o in expected if not o[2])
        reductions += sum(1 for o in expected if o[2])

    assert entries >= 3 and reductions >= 3                       # the comparison saw real trading both ways
    assert len(research.broker.fills) == entries + reductions      # research filled every order it sent
    kinds = {r["kind"] for r in _events(tmp_path)}
    assert "blocked" not in kinds and "issue" not in kinds        # no gate or reconciliation interfered
    assert session.position_qty() == venue.positions["BTCUSDT"]   # and the ledger equals the exchange
