"""The order limiter in Python, its daily counts kept per UTC day (migration step 13b; ADR-0008 1, ADR-0007 2b).

The rules and their order are those of the Rust limiter it replaces (a parity test runs both on the same
orders). Two things change by decision: a "day" is a UTC calendar day instead of 24 h since the process
started, and the day's counts are stored, so a restart no longer resets them. A count that cannot be read
or saved refuses the order ("treated as used up").
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path

import pytest

from trading.adapters.sqlite.limit_store import SqliteLimitStore
from trading.application.risk.order_limiter import OrderLimiter
from trading.domain.risk.order_limits import OrderLimitsConfig
from trading.infrastructure.config import RiskLimits, risk_limits
from tests.support.state_db import state_db

SCOPE = "binance-testnet-ut"


def _ts(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def _limiter(tmp_path, config, clock=None, monotonic=None):
    clock = clock or {"t": _ts("2026-11-15T12:00:00")}
    mono = monotonic or {"t": 100.0}
    tmp_path.mkdir(parents=True, exist_ok=True)       # the store never creates the state directory
    store = SqliteLimitStore(state_db(tmp_path / "order_limits.sqlite"), SCOPE)
    return OrderLimiter(config, store=store, clock=lambda: clock["t"], monotonic=lambda: mono["t"]), clock, mono


RUST_DECISIONS = Path(__file__).resolve().parents[2] / "fixtures" / "rust_order_limiter_decisions.json"


@pytest.mark.parametrize("seed", ["1", "2", "3"])
def test_every_rule_but_the_day_decides_as_the_rust_limiter_did(tmp_path, seed):
    """The Rust limiter's decisions were recorded on 2026-11-15, before the Rust extension was deleted: the same
    80 random orders per seed this test compared live until then (fixture written by that comparison's code)."""
    recorded = json.loads(RUST_DECISIONS.read_text())
    limits = {k: int(v) if k == "max_daily_orders" else D(v) for k, v in recorded["limits"].items()}
    python, _, _ = _limiter(tmp_path / seed, OrderLimitsConfig(**limits))
    decisions = []
    for row in recorded["seeds"][seed]:
        b = python.check_order(qty=D(row["qty"]), price=D(row["price"]),
                               current_position_notional=D(row["position"]))
        assert (row["allowed"], row["rule"]) == (b.allowed, b.violated_rule)
        decisions.append(b.violated_rule)
    assert {"max_order_qty", "max_order_notional", "max_position_notional", None} <= set(decisions)
    assert "max_daily_orders" in decisions or "max_daily_notional" in decisions


def test_orders_per_second_count_only_allowed_orders_in_the_last_second(tmp_path):
    limiter, _, mono = _limiter(tmp_path, OrderLimitsConfig(max_orders_per_second=1.0))
    assert limiter.check_order(qty=D("1"), price=D("10")).allowed
    refused = limiter.check_order(qty=D("1"), price=D("10"))
    assert (refused.allowed, refused.violated_rule, refused.detail) == (
        False, "max_orders_per_second", "max_orders_per_second: rate=1/s")
    mono["t"] += 0.999
    assert limiter.check_order(qty=D("1"), price=D("10")).allowed is False    # still inside the second
    mono["t"] += 0.001
    assert limiter.check_order(qty=D("1"), price=D("10")).allowed             # one second: out, as in Rust


def test_a_restart_keeps_the_day_s_counts(tmp_path):
    config = OrderLimitsConfig(max_daily_orders=2, max_daily_notional=D("1000"))
    first, clock, _ = _limiter(tmp_path, config)
    assert [first.check_order(qty=D("1"), price=D("100")).allowed for _ in range(2)] == [True, True]
    restarted, _, _ = _limiter(tmp_path, config, clock=clock)
    refused = restarted.check_order(qty=D("1"), price=D("100"))
    assert (refused.allowed, refused.detail) == (False, "max_daily_orders: count=2")


def test_the_day_is_the_utc_calendar_day(tmp_path):
    config = OrderLimitsConfig(max_daily_orders=1, max_daily_notional=D("150"))
    limiter, clock, mono = _limiter(tmp_path, config, clock={"t": _ts("2026-11-15T23:59:59")})
    assert limiter.check_order(qty=D("1"), price=D("100")).allowed
    assert limiter.check_order(qty=D("1"), price=D("10")).violated_rule == "max_daily_orders"
    clock["t"] = _ts("2026-11-16T00:00:00")                                   # one second later, a new day
    assert limiter.check_order(qty=D("1"), price=D("100")).allowed
    refused = limiter.check_order(qty=D("1"), price=D("100"))
    assert refused.violated_rule == "max_daily_orders"


def test_daily_notional_adds_up_only_allowed_orders(tmp_path):
    limiter, _, _ = _limiter(tmp_path, OrderLimitsConfig(max_order_notional=D("300"), max_daily_notional=D("500")))
    assert limiter.check_order(qty=D("1"), price=D("250")).allowed
    assert limiter.check_order(qty=D("1"), price=D("301")).violated_rule == "max_order_notional"   # not counted
    assert limiter.check_order(qty=D("1"), price=D("250")).allowed
    refused = limiter.check_order(qty=D("1"), price=D("0.01"))
    assert (refused.violated_rule, refused.detail) == ("max_daily_notional", "max_daily_notional: total=500.01")


def test_a_count_that_cannot_be_read_or_saved_refuses_the_order(tmp_path):
    limiter, _, _ = _limiter(tmp_path, OrderLimitsConfig(max_daily_orders=50))
    limiter.store.load = lambda day: (_ for _ in ()).throw(sqlite3.DatabaseError("unreadable"))
    refused = limiter.check_order(qty=D("1"), price=D("10"))
    assert (refused.allowed, refused.violated_rule) == (False, "max_daily_orders")
    assert "unreadable" in refused.detail
    limiter, _, _ = _limiter(tmp_path / "b", OrderLimitsConfig(max_daily_orders=50))
    limiter.store.add = lambda day, notional: (_ for _ in ()).throw(sqlite3.OperationalError("disk full"))
    refused = limiter.check_order(qty=D("1"), price=D("10"))
    assert (refused.allowed, refused.violated_rule) == (False, "max_daily_orders") and "not saved" in refused.detail


def test_counts_are_kept_per_scope(tmp_path):
    config = OrderLimitsConfig(max_daily_orders=1)
    store_a = SqliteLimitStore(state_db(tmp_path / "order_limits.sqlite"), "scope-a")
    store_b = SqliteLimitStore(state_db(tmp_path / "order_limits.sqlite"), "scope-b")
    a = OrderLimiter(config, store=store_a, clock=lambda: 0.0, monotonic=lambda: 0.0)
    b = OrderLimiter(config, store=store_b, clock=lambda: 0.0, monotonic=lambda: 10.0)
    assert a.check_order(qty=D("1"), price=D("1")).allowed and b.check_order(qty=D("1"), price=D("1")).allowed


# -- the drawdown ceiling (tech lead, 2026-11-15) ---------------------------------------------------------------
def test_drawdown_limits_above_the_ceiling_refuse_to_start():
    assert risk_limits({"MAX_DAILY_DRAWDOWN_PCT": "10", "MAX_WEEKLY_DRAWDOWN_PCT": "25"}).daily_drawdown_pct == 10.0
    for env in ({"MAX_DAILY_DRAWDOWN_PCT": "10.01"}, {"MAX_WEEKLY_DRAWDOWN_PCT": "25.5"},
                {"MAX_DAILY_DRAWDOWN_PCT": "50"}, {"MAX_DAILY_DRAWDOWN_PCT": "0"}, {"MAX_DAILY_DRAWDOWN_PCT": "nan"}):
        with pytest.raises(RuntimeError, match="drawdown"):
            risk_limits(env)
    assert isinstance(risk_limits({}), RiskLimits)


def test_preflight_refuses_a_drawdown_limit_above_the_ceiling_before_anything_else(tmp_path, monkeypatch):
    import json

    from lab.strategy import StrategySpec
    from runner.binance_target import preflight
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", SCOPE)
    monkeypatch.setenv("MAX_WEEKLY_DRAWDOWN_PCT", "40")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": ["BTCUSDT"],
                                "spec": StrategySpec().record(), "passed": False}))
    with pytest.raises(RuntimeError, match="drawdown"):
        preflight(spec, execution=True, testnet=True, stop_pct=D("0.08"))
    preflight(spec, execution=False, testnet=True, stop_pct=D("0.08"))       # paper mode does not use them


def test_the_runtime_counts_its_orders_in_its_state_directory_under_its_scope(tmp_path, monkeypatch):
    from tests.characterization.test_runtime_decisions import _start, _until_position
    from trading.domain.risk.drawdown import periods
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", SCOPE)
    world, runtime = _start(tmp_path)
    _until_position(world, runtime)
    day = periods(world["clock"]["now"])["day"]
    orders, notional = SqliteLimitStore(tmp_path / "exec" / "state.sqlite", SCOPE).load(day)
    assert orders >= 1 and notional > 0
