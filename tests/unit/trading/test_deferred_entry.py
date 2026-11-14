"""An entry refused only by the orders-per-second limit is tried again within the same bar (fix of R23).

Decision of the tech lead, 2026-11-14: the limit itself is unchanged; the refused entry is kept and sent
on a later cycle of the same bar, and never after the bar is too old to open a position.
"""
from __future__ import annotations

import time
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from tests.characterization.test_exit_paths import _tick
from tests.characterization.test_runtime_decisions import _events, _start, _until_position
from tests.unit.binance.fake_venue import HOUR
from tests.unit.lab.test_target_runtime import _advance
from trading.application.risk.entry_gate import RATE_LIMIT_RULE, CompositeRiskGate, GateResult
from trading.application.session import MAX_ENTRY_DELAY_MS
from trading.domain.orders import OrderIntent
from trading.domain.rebalance import Intent


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


RATE = GateResult(False, "max_orders_per_second: rate=1/s", retry=True)


class ScriptedGate:
    """Answers entries from a script, then allows; reductions always pass (as the real gate does)."""

    def __init__(self, *script):
        self.script, self.entries = list(script), []

    def check(self, order):
        if order.reduce_only or order.qty == 0:
            return GateResult(True)
        self.entries.append(order.order_id)
        return self.script.pop(0) if self.script else GateResult(True)


def _at_first_entry(tmp_path, *script):
    """A runtime stopped just before the bar whose close opens the first BTC position; BTC's gate is scripted."""
    reference, runtime = _start(tmp_path / "ref")
    entry_bar = _until_position(reference, runtime) - 1
    world, runtime = _start(tmp_path / "run")
    for i in range(world["warm"], entry_bar):
        _advance(world, runtime, i)
    session = runtime.sessions["BTCUSDT"]
    session.gate = ScriptedGate(*script)
    return world, runtime, session, entry_bar


def _btc(tmp_path, kind):
    return [r for r in _events(tmp_path / "run", kind) if r.get("symbol") == "BTCUSDT"]


def test_a_rate_limited_entry_is_sent_on_the_next_cycle_of_the_same_bar(tmp_path):
    world, runtime, session, bar = _at_first_entry(tmp_path, RATE)
    btc = world["fakes"]["BTCUSDT"]
    _advance(world, runtime, bar)
    assert btc.posts == [] and session.deferred is not None
    (blocked,) = _btc(tmp_path, "blocked")
    assert blocked["reason"].startswith("max_orders_per_second") and blocked["retry"] is True
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] != 0 and session.deferred is None
    assert session.gate.entries == [blocked["order_id"]] * 2                 # the same intent, asked twice
    assert [r["order_id"] for r in _btc(tmp_path, "intent")] == [blocked["order_id"]]   # decided once
    assert [r["order_id"] for r in _btc(tmp_path, "fill")] == [blocked["order_id"]]
    close = world["rows"]["BTCUSDT"][bar]["start"] + HOUR
    assert blocked["order_id"] == f"BTCUSDT:{close}:rebalance:0"


def test_an_entry_still_rate_limited_stays_deferred_until_it_passes(tmp_path):
    world, runtime, session, bar = _at_first_entry(tmp_path, RATE, RATE, RATE)
    btc = world["fakes"]["BTCUSDT"]
    _advance(world, runtime, bar)
    for _ in range(2):
        _tick(world, runtime)
        assert btc.posts == [] and session.deferred is not None
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] != 0 and len(session.gate.entries) == 4
    assert len(_btc(tmp_path, "blocked")) == 1                               # logged when deferred, not per retry


def test_a_deferred_entry_refused_for_another_reason_is_dropped(tmp_path):
    world, runtime, session, bar = _at_first_entry(tmp_path, RATE, GateResult(False, "kill_switch: test"))
    btc = world["fakes"]["BTCUSDT"]
    _advance(world, runtime, bar)
    _tick(world, runtime)
    assert btc.posts == [] and session.deferred is None
    assert [r["reason"] for r in _btc(tmp_path, "blocked")][-1] == "kill_switch: test"
    _tick(world, runtime)
    assert btc.posts == [] and len(session.gate.entries) == 2                # not asked again


def test_a_deferred_entry_is_dropped_once_the_bar_is_too_old_to_open_a_position(tmp_path):
    world, runtime, session, bar = _at_first_entry(tmp_path, RATE)
    btc = world["fakes"]["BTCUSDT"]
    _advance(world, runtime, bar)
    close = world["rows"]["BTCUSDT"][bar]["start"] + HOUR
    world["clock"]["now"] = (close + MAX_ENTRY_DELAY_MS + 1_000) / 1000
    runtime.cycle()
    assert btc.posts == [] and session.deferred is None and len(session.gate.entries) == 1
    assert _btc(tmp_path, "blocked")[-1]["reason"] == "inputs: bar processed too late"


def test_the_next_bar_replaces_a_deferred_entry_with_its_own_decision(tmp_path):
    world, runtime, session, bar = _at_first_entry(tmp_path, RATE, RATE)
    btc = world["fakes"]["BTCUSDT"]
    _advance(world, runtime, bar)
    old_close = world["rows"]["BTCUSDT"][bar]["start"] + HOUR
    assert session.deferred.bar.available_at == old_close
    _advance(world, runtime, bar + 1)          # an hour later: the old entry is too late and the new bar decides
    assert btc.posts == [] and session.deferred.bar.available_at == old_close + HOUR
    assert session.gate.entries == [f"BTCUSDT:{old_close}:rebalance:0", f"BTCUSDT:{old_close + HOUR}:rebalance:0"]
    assert [r["reason"] for r in _btc(tmp_path, "blocked")][1] == "inputs: bar processed too late"


def test_a_flatten_is_not_held_back_by_a_deferred_entry_and_cancels_it(tmp_path):
    world, runtime, session, bar = _at_first_entry(tmp_path)
    btc = world["fakes"]["BTCUSDT"]
    _advance(world, runtime, bar)              # the position is open
    assert btc.positions["BTCUSDT"] != 0
    _advance(world, runtime, bar + 1)
    session.gate.script.append(RATE)
    session.submit([Intent("buy" if btc.positions["BTCUSDT"] > 0 else "sell", D("0.001"), False)],
                   session.bars[-1], "rebalance")                      # an add-on entry that gets deferred
    assert session.deferred is not None
    posts = len(btc.posts)
    session.flatten("kill_switch")
    assert btc.positions["BTCUSDT"] == 0 and session.deferred is None
    assert [p.get("reduceOnly") in (True, "true") for p in btc.posts[posts:]] == [True]
    _tick(world, runtime)
    assert len(btc.posts) == posts + 1                                 # the deferred entry never goes out


def test_a_newer_bar_drops_a_deferred_entry_even_if_no_retry_ran_in_between(tmp_path):
    world, runtime, session, bar = _at_first_entry(tmp_path, RATE)
    _advance(world, runtime, bar)
    assert session.deferred is not None
    world["clock"]["now"] = (world["rows"]["BTCUSDT"][bar + 1]["start"] + HOUR + 5_000) / 1000
    runtime.poll()                              # the bar step alone, without the retry at the top of a cycle
    close = world["rows"]["BTCUSDT"][bar]["start"] + HOUR
    dropped = [r for r in _btc(tmp_path, "blocked") if r["reason"] == "superseded by a newer bar"]
    assert [r["order_id"] for r in dropped] == [f"BTCUSDT:{close}:rebalance:0"]
    assert session.deferred is None or session.deferred.bar.available_at == close + HOUR


def test_the_entry_of_a_flip_keeps_its_own_order_id_when_it_is_deferred(tmp_path):
    world, runtime, session, bar = _at_first_entry(tmp_path)
    btc = world["fakes"]["BTCUSDT"]
    _advance(world, runtime, bar)
    _advance(world, runtime, bar + 1)
    position = btc.positions["BTCUSDT"]
    assert position != 0
    side = "sell" if position > 0 else "buy"
    session.gate.script.append(RATE)
    session.submit([Intent(side, abs(position), True), Intent(side, D("0.004"), False)], session.bars[-1], "flip")
    close = session.bars[-1].available_at
    assert btc.positions["BTCUSDT"] == 0 and [k for k, _ in session.deferred.items] == [1]
    _tick(world, runtime)
    assert abs(btc.positions["BTCUSDT"]) == D("0.004") and (btc.positions["BTCUSDT"] > 0) != (position > 0)
    filled = [r["order_id"] for r in _btc(tmp_path, "fill")]
    assert filled[-2:] == [f"BTCUSDT:{close}:flip:0", f"BTCUSDT:{close}:flip:1"]


def test_a_reduction_is_never_deferred_whatever_the_gate_answers(tmp_path):
    world, runtime, session, bar = _at_first_entry(tmp_path)
    btc = world["fakes"]["BTCUSDT"]
    _advance(world, runtime, bar)
    position = btc.positions["BTCUSDT"]

    class Stubborn:
        def check(self, order):
            return RATE

    session.gate = Stubborn()
    session.submit([Intent("sell" if position > 0 else "buy", abs(position), True)], session.bars[-1], "test")
    assert session.deferred is None and _btc(tmp_path, "blocked")[-1].get("retry") is None


# -- the gate says which refusal may be retried ---------------------------------------------------------------
class _Limiter:
    def __init__(self, rule):
        self.rule = rule

    def check_order(self, *, qty, price=None):
        return SimpleNamespace(allowed=self.rule is None, violated_rule=self.rule,
                               detail=f"{self.rule}: detail" if self.rule else None)


class _Open:
    def allow_order(self, *, symbol):
        return True, ""

    def check(self, *, now=None):
        return True, ""

    def check_would_exceed(self, **kw):
        return True, ""


def _entry():
    return OrderIntent("BTCUSDT:1:t:0", "target.BTCUSDT", "BTCUSDT", "buy", D("0.01"), D("30000"), False, "t")


@pytest.mark.parametrize("rule,retry", [(RATE_LIMIT_RULE, True), ("max_daily_orders", False),
                                        ("max_order_notional", False), ("max_daily_notional", False)])
def test_only_the_per_second_limit_is_a_refusal_that_may_be_retried(rule, retry):
    gate = CompositeRiskGate.for_execution(_Limiter(rule), kill_switch=_Open(), drawdown_guard=_Open(),
                                           portfolio_reader=_Open(), venue="binance")
    result = gate.check(_entry())
    assert (result.allowed, result.retry) == (False, retry) and result.reason == f"{rule}: detail"
    assert RATE_LIMIT_RULE == "max_orders_per_second"


def test_an_allowed_entry_and_every_other_refusal_are_not_marked_for_retry():
    opened = CompositeRiskGate.for_execution(_Limiter(None), kill_switch=_Open(), drawdown_guard=_Open(),
                                             portfolio_reader=_Open(), venue="binance")
    assert opened.check(_entry()) == GateResult(True, "")

    class Killed(_Open):
        def allow_order(self, *, symbol):
            return False, "operator"

    killed = CompositeRiskGate.for_execution(_Limiter(RATE_LIMIT_RULE), kill_switch=Killed(),
                                             drawdown_guard=_Open(), portfolio_reader=_Open(), venue="binance")
    assert killed.check(_entry()) == GateResult(False, "kill_switch: operator")


# -- with the real limiter and both symbols -----------------------------------------------------------------
def test_with_the_real_limiter_the_second_symbol_enters_on_the_same_bar_a_cycle_later(tmp_path):
    world, runtime = _start(tmp_path / "run")
    i, limited = world["warm"], []
    while not limited:
        assert i < world["warm"] + 200, "no bar on which both symbols entered"
        _advance(world, runtime, i)
        limited = [r for r in _events(tmp_path / "run", "blocked") if r.get("retry")] \
            if (tmp_path / "run/exec/target_events.jsonl").exists() else []
        i += 1
    (row,) = limited
    session = runtime.sessions[row["symbol"]]
    posts = len(world["fakes"][row["symbol"]].posts)
    assert row["reason"].startswith("max_orders_per_second") and session.deferred is not None
    time.sleep(1.1)                            # the limiter counts real seconds
    _tick(world, runtime)
    assert session.deferred is None and len(world["fakes"][row["symbol"]].posts) == posts + 1
    assert row["order_id"] in [r["order_id"] for r in _events(tmp_path / "run", "fill")]
