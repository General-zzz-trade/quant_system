"""Paper runs the execution path on the simulated exchange (ADR-0013 plan B, S10 part 3).

The acceptance tests (paper equals research fill by fill, restarts, outages) are in test_target_runtime.py and
test_target_runtime_resilience.py. Here: what paper now has that research has not - the venue every order, stop and
exit goes to, sizing from each symbol's own simulated account, the working orders a restart keeps, the simulated
stop (decision L) and the ``paper_divergence`` events, written once for each entry refused, stop gone off and pause.
"""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.unit.lab.test_target_runtime import _advance, _world
from trading.adapters.simulated.exchange import SimulatedExchange
from trading.domain.symbol_state import SymbolState
from trading.infrastructure.modes import PAPER


@pytest.fixture(autouse=True)
def _dev_signing(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "RISK_SCOPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")


def _spec(execution="market", ttl=1):
    return StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution=execution,
                        ttl_bars=ttl)


def _paper(world, state_dir, **kw):
    return TargetRuntime(spec_path=world["spec_path"], state_dir=state_dir, mode="paper", market=world["adapter"],
                         clock=lambda: world["clock"]["now"], **kw)


def _events(state_dir):
    return [json.loads(line) for line in (state_dir / "target_events.jsonl").read_text().splitlines()]


def _divergences(state_dir, cause=None):
    return [e for e in _events(state_dir) if e["kind"] == "paper_divergence" and cause in (None, e["cause"])]


def test_every_order_stop_and_exit_of_paper_goes_to_the_simulated_exchange(tmp_path):
    world = _world(tmp_path, _spec())
    runtime = _paper(world, tmp_path / "paper")
    venue = runtime.account
    assert isinstance(venue, SimulatedExchange) and runtime.policy is PAPER
    for symbol, session in runtime.sessions.items():
        service = session.executor._service                           # wiring check: private on purpose
        assert service._venue is venue and service._orders.adapter is venue
        assert session.stops.adapter is venue and session.flattener.account is venue
        assert service._orders.venue_expires_entries is True
        assert service._orders.journal.inner.entry_beside_working_reduction is True
        assert session.allocation == 1 and session.equity.current() == venue.get_symbol_equity(symbol)
        assert session.sleeve.broker is venue.broker(symbol)          # the read-only view the tests compare


def test_paper_sizes_each_symbol_from_its_own_simulated_account(tmp_path):
    world = _world(tmp_path, _spec())
    runtime = _paper(world, tmp_path / "paper")
    runtime.warmup()
    for i in range(world["warm"], world["warm"] + 40):
        _advance(world, runtime, i)
    btc, eth = (runtime.sessions[s] for s in ("BTCUSDT", "ETHUSDT"))
    assert btc.sleeve.broker.equity != eth.sleeve.broker.equity        # the accounts have drifted apart
    assert btc.sleeve_equity() == btc.sleeve.broker.equity and eth.sleeve_equity() == eth.sleeve.broker.equity


def test_a_paper_restart_still_knows_the_entry_that_is_working(tmp_path):
    world = _world(tmp_path, _spec("maker", 4))
    state = tmp_path / "paper"
    runtime = _paper(world, state)
    runtime.warmup()
    i = world["warm"]
    while not any(s.sleeve.broker.pending for s in runtime.sessions.values()):
        assert i < world["warm"] + 200, "no maker entry ever rested"
        _advance(world, runtime, i)
        i += 1
    working = {symbol: dict(s.pending) for symbol, s in runtime.sessions.items()}
    assert any(not reducing for p in working.values() for reducing in p.values())
    restarted = _paper(world, state)
    assert {symbol: dict(s.pending) for symbol, s in restarted.sessions.items()} == working


def test_each_entry_the_gates_refuse_is_one_divergence_event(tmp_path):
    world = _world(tmp_path, _spec())
    state = tmp_path / "paper"
    runtime = _paper(world, state)
    runtime.warmup()
    runtime.kill.write(active=True, reason="test", flatten=False)
    for i in range(world["warm"], world["warm"] + 60):
        _advance(world, runtime, i)
    refused = [e for e in _events(state) if e["kind"] == "blocked" and not e.get("retry")]
    assert refused, "the kill switch refused no entry"
    divergences = _divergences(state, "entry_refused")
    assert sorted(e["order_id"] for e in divergences) == sorted(e["order_id"] for e in refused)
    assert all(e["symbol"] and e["reason"] for e in divergences)
    assert not [e for e in _events(state) if e["kind"] == "fill"]     # nothing traded, so nothing else diverged
    assert _divergences(state) == divergences


def test_a_crash_through_the_stop_fills_it_pauses_the_symbol_and_writes_one_event_per_cause(tmp_path):
    world = _world(tmp_path, _spec())
    state = tmp_path / "paper"
    runtime = _paper(world, state)
    runtime.warmup()
    eth = runtime.sessions["ETHUSDT"]
    i = world["warm"]
    while eth.position_qty() == 0 or not eth.stops.has_active_stop():
        assert i < world["warm"] + 300, "no ETH position with a stop"
        _advance(world, runtime, i)
        i += 1
    assert _divergences(state) == []                                  # until now paper is research
    qty = eth.position_qty()
    (stop,) = [s for s in runtime.account.get_open_algo_orders() if s.symbol == "ETHUSDT"]
    row = world["rows"]["ETHUSDT"][i]                                 # the next bar crashes through the stop
    if qty > 0:
        row["low"] = float(stop.trigger_price * D("0.97"))
    else:
        row["high"] = float(stop.trigger_price * D("1.03"))
    for j in range(i, i + 3):
        _advance(world, runtime, j)

    broker = eth.sleeve.broker
    stop_fills = [f for f in broker.fills if str(f["order"]).startswith("stop:")]
    expected = stop.trigger_price * (D("0.999") if qty > 0 else D("1.001"))
    assert [(D(f["qty"]), D(f["price"])) for f in stop_fills] == [(abs(qty), expected)]
    assert eth.position_qty() == 0 and broker.position.qty == 0       # the ledger consumed the stop fill
    assert eth.state is SymbolState.PAUSED and runtime.pauses.reason("ETHUSDT") is not None
    assert [e["symbol"] for e in _divergences(state, "stop_triggered")] == ["ETHUSDT"]
    assert [e["symbol"] for e in _divergences(state, "pause")] == ["ETHUSDT"]


def test_the_paper_status_shows_the_symbol_states_and_the_risk_state(tmp_path):
    world = _world(tmp_path, _spec())
    runtime = _paper(world, tmp_path / "paper")
    runtime.warmup()
    _advance(world, runtime, world["warm"])
    status = json.loads((tmp_path / "paper/target_status.json").read_text())
    assert status["mode"] == "paper"
    assert {s["state"] for s in status["symbols"].values()} == {"ready"}
    assert status["kill"]["active"] is False and status["drawdown_latched"] is False
