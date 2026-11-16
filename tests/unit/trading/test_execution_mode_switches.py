"""S10 part 3 audit, finding M2: execution never gets a paper-only switch, shown by execution's behaviour.

- the order lifecycle cancels a resting maker entry once its lifetime has passed (paper leaves that to the
  simulated venue: ``ModePolicy.entry_ttl``);
- the order journal refuses an entry while a reduction of the symbol is still working (paper allows the entry of a
  flip beside its acknowledged reduction: ``ModePolicy.entry_beside_working_reduction``);
- after a restart a session counts as pending only its own symbol's working orders, and a reduction as a reduction.
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.support.state_db import state_dir_db
from tests.unit.lab.test_target_runtime import _advance, _world
from trading.application.session import ExecutionParts, TargetSession
from trading.domain.orders import ENTRY_UNRESOLVED, OrderIntent
from trading.domain.rebalance import Lot

HOUR_S = 3600.0


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _execution(tmp_path, execution="market", ttl=1):
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution=execution,
                        ttl_bars=ttl)
    world = _world(tmp_path, spec)
    state_dir_db(tmp_path / "exec")
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"],
                            clock=lambda: world["clock"]["now"], testnet=True)
    runtime.warmup()
    return world, runtime


def test_execution_cancels_a_resting_maker_entry_once_its_lifetime_has_passed(tmp_path):
    world, runtime = _execution(tmp_path, "maker", 1)
    fakes, i = world["fakes"], world["warm"]

    def resting():
        return [(s, o) for s, f in fakes.items() for o in f.orders.values()
                if o["type"] == "LIMIT" and o["status"] == "NEW"]
    while not resting():
        assert i < world["warm"] + 200, "no maker entry rested"
        _advance(world, runtime, i)
        i += 1
    (symbol, order), = resting()[:1]
    lifecycle = runtime.sessions[symbol].executor._service._orders           # wiring check: private on purpose
    assert lifecycle.venue_expires_entries is False and lifecycle.entry_ttl == HOUR_S
    world["clock"]["now"] += HOUR_S                                           # one bar's lifetime later
    runtime.reconcile()
    assert order["status"] == "CANCELED"


def test_execution_refuses_an_entry_while_a_reduction_of_the_symbol_is_working(tmp_path):
    world, runtime = _execution(tmp_path)
    journal = runtime.sessions["BTCUSDT"].executor.journal

    def reserve(intent_id, reducing):
        return journal.reserve(intent_id=intent_id, symbol="BTCUSDT", client_id=f"c-{intent_id}", side="sell",
                               qty="0.1", reducing=reducing, owner="target.BTCUSDT", parent=None,
                               order_type="market", clock=lambda: world["clock"]["now"])[0]
    assert reserve("reduce", True) == ""
    journal.update_status("reduce", "submitted", settle=False)              # acknowledged, still working
    assert reserve("entry", False) == ENTRY_UNRESOLVED


class _Journal:
    def __init__(self, rows):
        self.rows = rows

    def orders(self, owner=None, *, unresolved=False):
        return [r for r in self.rows if owner is None or r["owner"] == owner]


def test_a_restart_counts_only_the_symbol_s_own_working_orders_and_a_reduction_as_a_reduction():
    rows = [{"intent_id": "b-entry", "symbol": "BTCUSDT", "reducing": 0, "owner": "target.BTCUSDT"},
            {"intent_id": "b-reduce", "symbol": "BTCUSDT", "reducing": 1, "owner": "target.BTCUSDT"},
            {"intent_id": "e-entry", "symbol": "ETHUSDT", "reducing": 0, "owner": "target.BTCUSDT"}]
    executor = SimpleNamespace(journal=_Journal(rows), owner="target.BTCUSDT")
    session = TargetSession("BTCUSDT", StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10),
                            journal=SimpleNamespace(write=lambda *a, **k: None),
                            rules=Lot(D("0.001"), D("0.001"), D("5")), make_order=OrderIntent,
                            execution=ExecutionParts(executor, None, None, None, None))
    session.restore_pending()
    assert session.pending == {"b-entry": False, "b-reduce": True}
