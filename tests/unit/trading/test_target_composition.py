"""How the target runtime is wired since migration step 10: the new execution path, with unchanged settings."""
from __future__ import annotations

import inspect
import json
import sqlite3
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from tests.support import legacy_execution_adapter as legacy
from tests.characterization.test_runtime_decisions import _start, _until_position
from trading.application.execution.executor import OwnerExecutor
from trading.domain.orders import OrderIntent
from trading.domain.rebalance import Lot
from trading.infrastructure import target
from trading.infrastructure.settings import RiskLimits, risk_limits


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def test_the_limits_keep_their_defaults_and_are_read_from_the_environment_only():
    assert risk_limits({}) == RiskLimits(daily_drawdown_pct=5.0, weekly_drawdown_pct=12.0,
                                         max_order_notional=D("5000"), max_daily_notional=D("50000"),
                                         max_daily_orders=50, max_leverage=D("3"))
    env = {"MAX_DAILY_DRAWDOWN_PCT": "2", "MAX_WEEKLY_DRAWDOWN_PCT": "6", "BINANCE_MAX_ORDER_NOTIONAL": "300",
           "BINANCE_MAX_DAILY_NOTIONAL": "900", "BINANCE_MAX_DAILY_ORDERS": "7", "BINANCE_MAX_LEVERAGE": "1.5"}
    assert risk_limits(env) == RiskLimits(2.0, 6.0, D("300"), D("900"), 7, D("1.5"))
    with pytest.raises(ValueError):
        risk_limits({"BINANCE_MAX_DAILY_ORDERS": "many"})


def test_the_order_path_settings_equal_those_of_the_legacy_adapter():
    defaults = {name: p.default for name, p in
                inspect.signature(legacy.BinanceExecutionAdapter.__init__).parameters.items()}
    assert target.ENTRY_WAIT_SECONDS == defaults["wait_seconds"]
    assert target.LIMIT_OFFSET == defaults["limit_offset_bps"] / D(10000)
    assert target.NOT_FOUND_GRACE_SECONDS == legacy.NOT_FOUND_GRACE_SECONDS
    assert target.HOUR_SECONDS == legacy.HOUR_SECONDS


def test_an_execution_session_runs_on_the_new_executor_with_plain_order_intents(tmp_path):
    world, runtime = _start(tmp_path)
    session = runtime.sessions["BTCUSDT"]
    assert type(session.executor) is OwnerExecutor and session._make_order is OrderIntent
    assert (session.executor.owner, session.executor.scope) == ("target.BTCUSDT", "binance-testnet-ut")
    _until_position(world, runtime)
    rows = session.executor.journal.orders("target.BTCUSDT")
    assert rows and all(set(json.loads(r["parent"])) == {"order_id", "symbol", "side", "price", "reason"}
                        for r in rows)
    tca = [json.loads(line) for line in (tmp_path / "exec/tca_binance.jsonl").read_text().splitlines()]
    assert tca and tca[0]["symbol"] in ("BTCUSDT", "ETHUSDT") and tca[0]["ref_price"] > 0


def test_the_entry_limits_reach_the_drawdown_guard_and_the_exposure_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_DAILY_DRAWDOWN_PCT", "2")
    monkeypatch.setenv("MAX_WEEKLY_DRAWDOWN_PCT", "6")
    monkeypatch.setenv("BINANCE_MAX_LEVERAGE", "9")               # above the hard cap: the cap wins
    _, runtime = _start(tmp_path)
    assert runtime.drawdown.limits == {"day": 2.0, "week": 6.0}
    gate = runtime.sessions["BTCUSDT"].gate.inner
    assert gate._portfolio_reader.max_leverage == D("3") and gate._venue == "binance"
    assert gate._kill_switch is runtime.kill and gate._drawdown_guard is runtime.drawdown


def test_mainnet_needs_a_passed_research_gate():
    target.refuse_unverified_mainnet(testnet=True, spec_passed=False)
    target.refuse_unverified_mainnet(testnet=False, spec_passed=True)
    for passed in (False, None, "true", 1):
        with pytest.raises(RuntimeError, match="mainnet execution refused"):
            target.refuse_unverified_mainnet(testnet=False, spec_passed=passed)


def test_a_stop_outside_the_liquidation_distance_is_refused_before_any_state_file_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("BINANCE_EXCHANGE_LEVERAGE", "10")
    monkeypatch.setenv("BINANCE_MARGIN_TYPE", "ISOLATED")
    built = []                                   # S2 audit finding 2: building a store no longer writes a file
    for name in ("exchange_lot_rules", "SqliteDrawdownGuard", "SqliteOrderJournal"):
        monkeypatch.setattr(target, name, lambda *a, _name=name, **k: built.append(_name))
    with pytest.raises(RuntimeError, match="liquidation distance"):
        _start(tmp_path)
    assert built == []                           # refused before the account is read or any store is built
    exec_dir = tmp_path / "exec"                 # only the database the test prepared, untouched (ADR-0009)
    assert sorted(p.name for p in exec_dir.iterdir() if p.suffix == ".sqlite") == ["state.sqlite"]
    with sqlite3.connect(exec_dir / "state.sqlite") as db:
        assert db.execute("SELECT count(*) FROM drawdown_periods").fetchone()[0] == 0


@pytest.mark.parametrize("execution,ttl,order_type,expires", [("market", 1, "market", None),
                                                               ("maker", 4, "limit", 4 * 3600.0)])
def test_the_order_lifecycle_gets_the_spec_s_entry_mode_and_lifetime(tmp_path, execution, ttl, order_type, expires):
    _, runtime = _start(tmp_path, execution=execution, ttl_bars=ttl)
    lifecycle = runtime.sessions["ETHUSDT"].executor._service._orders       # wiring check: private on purpose
    assert (lifecycle.entry_order_type, lifecycle.entry_ttl) == (order_type, expires)
    assert (lifecycle.wait_seconds, lifecycle.not_found_grace, lifecycle.owner) == (0.5, 90.0, "target.ETHUSDT")


def test_a_failed_reconciliation_blocks_entries_until_the_next_clean_pass(tmp_path):
    world, runtime = _start(tmp_path)
    session = runtime.sessions["BTCUSDT"]
    assert session.executor.ready is True
    sync = session.stops.sync

    def broken():
        raise OSError("stop lookup failed")

    session.stops.sync = broken
    runtime.reconcile()
    assert session.executor.ready is False and runtime.sessions["ETHUSDT"].executor.ready is True
    issues = [json.loads(line) for line in (tmp_path / "exec/target_events.jsonl").read_text().splitlines()]
    assert [r["issue"] for r in issues if r["kind"] == "issue"] == [
        "reconciliation failed: OSError: stop lookup failed"]
    session.stops.sync = sync
    runtime.reconcile()
    assert session.executor.ready is True


class _Filters:
    """An account whose exchange minimum (5) is below the research constant for BTC (100)."""

    def filters(self, symbol):
        return SimpleNamespace(step_size=D("0.001"), min_qty=D("0.001"),
                               min_notional=D("5") if symbol == "BTCUSDT" else D("50"))


def test_execution_lot_rules_are_the_exchange_s_own_filters():
    """Decision of 2026-11-14: execution follows the exchange; the research constants are not a floor."""
    assert target.exchange_lot_rules(_Filters(), ["BTCUSDT", "ETHUSDT"]) == {
        "BTCUSDT": Lot(D("0.001"), D("0.001"), D("5")), "ETHUSDT": Lot(D("0.001"), D("0.001"), D("50"))}


def test_an_execution_session_trades_with_the_exchange_minimum(tmp_path):
    _, runtime = _start(tmp_path)
    rules = runtime.sessions["BTCUSDT"].rules
    assert (rules.step, rules.min_qty, rules.min_notional) == (D("0.001"), D("0.001"), D("5"))   # the fake venue's


def test_execution_sends_a_small_order_that_research_and_paper_skip():
    """The stated consequence of that decision: between the exchange minimum and the research constant,
    execution trades and research does not. It matters only for small accounts."""
    from lab.simulate import SYMBOL_RULES
    from trading.domain.rebalance import Intent, plan
    exchange, research = target.exchange_lot_rules(_Filters(), ["BTCUSDT"])["BTCUSDT"], SYMBOL_RULES["BTCUSDT"]
    assert research.min_notional == D("100")
    sizing = dict(equity=D("300"), price=D("60000"), current_qty=D(0), entry_pending=False, band=0.05)

    def orders(rules):
        return plan(0.25, step=rules.step, min_qty=rules.min_qty, min_notional=rules.min_notional, **sizing)

    assert orders(exchange) == [Intent("buy", D("0.001"), False)]        # 60 USDT: above the exchange minimum
    assert orders(research) == []                                         # below the research constant
