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


# -- added after the mutation self-check ------------------------------------------------------------------------
def test_settling_a_bar_sets_the_symbol_states_before_the_decision(tmp_path):
    world = _world(tmp_path, _spec())
    runtime = _paper(world, tmp_path / "paper")
    runtime.warmup()
    assert runtime.sessions["BTCUSDT"].state is SymbolState.READY
    runtime.pauses.pause("BTCUSDT", "protective stop triggered")
    runtime.exits.settle()
    assert runtime.sessions["BTCUSDT"].state is SymbolState.PAUSED
    assert runtime.sessions["ETHUSDT"].state is SymbolState.READY


def test_a_bar_that_is_not_live_only_extends_the_window(tmp_path, monkeypatch):
    from trading.domain.bars import bar_from_kline
    world = _world(tmp_path, _spec())
    runtime = _paper(world, tmp_path / "paper")
    runtime.warmup()
    session, decided = runtime.sessions["BTCUSDT"], []
    monkeypatch.setattr(session, "decide", lambda bar, *, now_ms: decided.append(bar.open_time))
    rows = world["rows"]["BTCUSDT"]
    first, second = bar_from_kline(rows[world["warm"]]), bar_from_kline(rows[world["warm"] + 1])
    session.process_bar(first, live=False, now_ms=0)
    session.process_bar(second, live=True, now_ms=0)
    assert decided == [second.open_time] and session.cursor == second.open_time


def test_a_restart_decides_the_missed_bars_with_a_fresh_equity_reading(tmp_path):
    spec = _spec()
    for name in ("probe", "run"):
        (tmp_path / name).mkdir()
    probe = _world(tmp_path / "probe", spec)               # where does BTC first enter after a few bars?
    runtime = _paper(probe, tmp_path / "probe/paper")
    runtime.warmup()
    entry, i = None, probe["warm"]
    while entry is None:
        assert i < probe["warm"] + 200, "BTC never entered"
        _advance(probe, runtime, i)
        if i >= probe["warm"] + 3 and any(e["kind"] == "intent" and e["symbol"] == "BTCUSDT" and not e["reduce_only"]
                                          for e in _events(tmp_path / "probe/paper")
                                          if e["kind"] == "intent" and e["order_id"].startswith(
                                              f"BTCUSDT:{probe['rows']['BTCUSDT'][i]['start'] + 3_600_000}:")):
            entry = i
        i += 1
    world = _world(tmp_path / "run", spec)
    state = tmp_path / "run/paper"
    runtime = _paper(world, state)
    runtime.warmup()
    for j in range(world["warm"], entry - 2):
        _advance(world, runtime, j)
    world["clock"]["now"] = (world["rows"]["BTCUSDT"][entry]["start"] + 3_600_000 + 5_000) / 1000
    _paper(world, state).warmup()                            # down for three bars, the last one the entry's
    events = _events(state)
    close = world["rows"]["BTCUSDT"][entry]["start"] + 3_600_000
    entries = [e["order_id"] for e in events if e["kind"] == "intent" and not e["reduce_only"]
               and e["order_id"].startswith(f"BTCUSDT:{close}:")]
    assert entries
    assert not [e for e in events if e["kind"] in ("blocked", "paper_divergence") and e.get("order_id") in entries]
    assert [e["status"] for e in events if e["kind"] == "order" and e.get("order_id") in entries][:1] == ["reserved"]


# -- S10 part 3 audit: M1 (risk exits are divergences) and L3 (no single-order notional cap in paper) -------------
def _risk_divergences(state):
    return [(e["cause"], e["symbol"], e["reason"]) for e in _divergences(state)
            if e["cause"] in ("flatten", "cancel_entries")]


def _holding_any(world, runtime):
    i = world["warm"]
    while all(s.position_qty() == 0 for s in runtime.sessions.values()):
        assert i < world["warm"] + 300, "no position"
        _advance(world, runtime, i)
        i += 1
    return i


def test_a_drawdown_latch_flatten_is_one_divergence_per_symbol(tmp_path):
    world = _world(tmp_path, _spec())
    state = tmp_path / "paper"
    runtime = _paper(world, state)
    runtime.warmup()
    i = _holding_any(world, runtime)
    assert _risk_divergences(state) == []
    runtime.drawdown.observe(float(runtime.account.get_account_equity()) * 0.9)      # a 10 % fall today
    for j in range(i, i + 3):
        _advance(world, runtime, j)
    assert _risk_divergences(state) == [("flatten", "BTCUSDT", "drawdown_latched"),
                                        ("flatten", "ETHUSDT", "drawdown_latched")]
    assert all(runtime.account.broker(s).position.qty == 0 for s in runtime.sessions)


def test_the_kill_switch_is_one_divergence_per_symbol_and_action_each_time_it_starts(tmp_path):
    world = _world(tmp_path, _spec())
    state = tmp_path / "paper"
    runtime = _paper(world, state)
    runtime.warmup()
    i = _holding_any(world, runtime)
    runtime.kill.write(active=True, reason="test", flatten=False)
    for j in range(i, i + 2):
        _advance(world, runtime, j)
    runtime.kill.write(active=True, reason="test", flatten=True)
    for j in range(i + 2, i + 4):
        _advance(world, runtime, j)
    runtime.kill.write(active=False, reason="test")
    _advance(world, runtime, i + 4)
    runtime.kill.write(active=True, reason="again", flatten=True)
    _advance(world, runtime, i + 5)
    symbols = ("BTCUSDT", "ETHUSDT")
    assert _risk_divergences(state) == [*[("cancel_entries", s, "kill_switch") for s in symbols],
                                        *[("flatten", s, "kill_switch") for s in symbols],
                                        *[("flatten", s, "kill_switch") for s in symbols]]


def test_cancelling_resting_entries_for_an_account_problem_is_a_divergence_once(tmp_path):
    from trading.application.issues import Issue
    from trading.domain.issues import IssueKind
    world = _world(tmp_path, _spec())
    state = tmp_path / "paper"
    runtime = _paper(world, state)
    runtime.warmup()
    foreign = Issue(IssueKind.FOREIGN_HOLDING, "BTCUSDT", "BTCUSDT: unowned open order x")
    for _ in range(2):
        runtime.exits.last_issues["BTCUSDT"] = [foreign]
        runtime.exits.risk_actions()
    assert _risk_divergences(state) == [("cancel_entries", "BTCUSDT", "foreign_holding")]


def test_paper_has_no_single_order_notional_cap_and_keeps_the_other_limits(tmp_path):
    world = _world(tmp_path, _spec())
    runtime = _paper(world, tmp_path / "paper")
    config = runtime.sessions["BTCUSDT"].gate.inner._limiter._cfg     # wiring check: private on purpose
    assert config.max_order_notional is None
    assert (config.max_daily_notional, config.max_daily_orders, config.max_orders_per_second) == \
        (D("50000"), 50, 1.0)
