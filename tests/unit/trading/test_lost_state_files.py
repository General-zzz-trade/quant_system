"""A state file that is emptied, replaced or deleted never lets exposure through (audit after M3, findings 1-3).

- At start: a store that is a valid file but has lost its tables (truncated to 0 bytes, or replaced by another
  database) refuses to start, so a truncated drawdown store cannot drop a latched guard.
- While running: a pause store deleted between two reconciliations still blocks entries, because the gate
  derives the pause from the durable stop triggers of the order journal as well; a limit store deleted or
  emptied refuses orders ("unreadable counts as used up", ADR-0007 2b) instead of restarting the day at zero.
"""
from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal as D

import pytest

from tests.characterization.test_exit_paths import _tick
from tests.characterization.test_runtime_decisions import _start, _until_position
from tests.unit.lab.test_target_runtime import _advance
from trading.adapters.sqlite.check import database_tables
from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
from trading.adapters.sqlite.limit_store import SqliteLimitStore
from trading.application.risk.order_limiter import OrderLimiter
from trading.domain.risk.order_limits import OrderLimitsConfig
from trading.infrastructure import state_dir as managed
from trading.infrastructure.state_dir import STORES, TABLES
from trading.infrastructure.target import DRAWDOWN_FILE, PAUSES_FILE

SCOPE = "binance-testnet-ut"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", SCOPE)


def _init(path):
    assert managed.main(["--state-dir", str(path), "init"], stdout=io.StringIO(),
                        env={"RISK_SCOPE": SCOPE, "BINANCE_TESTNET": "1"}) == 0


# -- finding 1: a store without its tables refuses to start ------------------------------------------------------
def test_each_store_declares_exactly_the_tables_initialization_creates(tmp_path):
    _init(tmp_path / "state")
    for name in STORES:
        created = {t for t in database_tables(tmp_path / "state" / name) if not t.startswith("sqlite_")}
        assert set(TABLES[name]) == created, name


@pytest.mark.parametrize("name", STORES)
def test_a_store_truncated_to_zero_bytes_refuses_to_start(tmp_path, name):
    _init(tmp_path / "state")
    (tmp_path / "state" / name).write_bytes(b"")
    with pytest.raises(RuntimeError, match=f"{name} has lost its tables"):
        managed.verify(tmp_path / "state", SCOPE)


def test_a_store_replaced_by_another_database_refuses_to_start(tmp_path):
    _init(tmp_path / "state")
    (tmp_path / "state" / DRAWDOWN_FILE).unlink()
    with sqlite3.connect(tmp_path / "state" / DRAWDOWN_FILE) as db:
        db.execute("CREATE TABLE something_else (x)")
    with pytest.raises(RuntimeError, match="drawdown.sqlite has lost its tables"):
        managed.verify(tmp_path / "state", SCOPE)


def test_a_truncated_drawdown_store_cannot_drop_a_latched_guard_on_restart(tmp_path):
    _init(tmp_path / "state")
    path, now = tmp_path / "state" / DRAWDOWN_FILE, 1_800_000_000.0
    guard = SqliteDrawdownGuard(path, SCOPE, daily_pct=5.0, weekly_pct=12.0, clock=lambda: now)
    guard.observe(1000.0)
    guard.observe(900.0)
    assert guard.latched() is True
    path.write_bytes(b"")
    with pytest.raises(RuntimeError, match="lost its tables"):
        managed.verify(tmp_path / "state", SCOPE)


# -- finding 2: a pause store deleted while running ---------------------------------------------------------------
def _entries(posts):
    return [p for p in posts if str(p.get("reduceOnly", "false")).lower() != "true"
            and p.get("type") != "STOP_MARKET" and not p.get("closePosition")]


def test_a_pause_store_deleted_between_reconciliations_lets_no_entry_through(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    btc = world["fakes"]["BTCUSDT"]
    (client_id,) = [k for k, v in btc.algo.items() if v["algoStatus"] == "NEW"]
    btc.trigger_stop(client_id, btc.mark("BTCUSDT"))
    _tick(world, runtime)
    _tick(world, runtime)
    assert runtime.pauses.reason("BTCUSDT") is not None
    for k in range(1, 25):
        (tmp_path / "exec" / PAUSES_FILE).unlink()           # lost again before every bar
        posts = len(btc.posts)
        _advance(world, runtime, i + k)
        assert _entries(btc.posts[posts:]) == [], f"an entry went out at bar {k}"
    assert runtime.pauses.reason("BTCUSDT") == "protective stop triggered"


# -- finding 3: a limit store deleted or emptied while running ----------------------------------------------------
def _limiter(tmp_path, max_daily_orders=2):
    now = datetime(2026, 11, 15, 12, tzinfo=timezone.utc).timestamp()
    store = SqliteLimitStore(tmp_path / "order_limits.sqlite", SCOPE)
    mono = iter(range(0, 10_000, 10))
    return OrderLimiter(OrderLimitsConfig(max_daily_orders=max_daily_orders), store=store, clock=lambda: now,
                        monotonic=lambda: float(next(mono)))


@pytest.mark.parametrize("damage", ["deleted", "emptied"])
def test_a_limit_store_lost_while_running_refuses_orders_instead_of_counting_from_zero(tmp_path, damage):
    limiter = _limiter(tmp_path)
    assert [limiter.check_order(qty=D("1"), price=D("10")).allowed for _ in range(3)] == [True, True, False]
    path = tmp_path / "order_limits.sqlite"
    path.unlink() if damage == "deleted" else path.write_bytes(b"")
    refused = limiter.check_order(qty=D("1"), price=D("10"))
    assert (refused.allowed, refused.violated_rule) == (False, "max_daily_orders")
    assert "unreadable" in refused.detail
    assert damage == "emptied" or not path.exists()            # a read never re-creates the store
