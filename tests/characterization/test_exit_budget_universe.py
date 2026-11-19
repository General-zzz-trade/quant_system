"""Plan D2 (docs/arch/dynamic-universe-plan.md): at the size of the dynamic universe the exit side's requests stay well
inside the client's request-weight budget, in normal cycles and while a kill switch flattens.

The world: 809 sessions (the tradable USDT perpetuals Binance listed on 2026-11-19, plan D1) on one fake venue, 10 of
them holding a position with its exchange stop. The exit side runs as the runtime composes it: the execution sessions
of ``trading.infrastructure.target`` (journal, lifecycle, stops, flattener), the whole-account check of ADR-0011 and
the equity refresh, in 5-second cycles. The market side is not run: inside an hour it sends nothing (its hourly reads
are plan D4).

Weights are the client's table (``trading.adapters.binance.rate_limit.endpoint_weight``). The client's own pool is
1200 a minute (``rate_limit.py``), Binance's limit 2400 a minute per IP. The bound is half the pool, as in the
10-symbol test (test_account_weight.py): the rest is for the market side's hourly reads, the strategy capital's
funding income and the loss stop's position reads (about 30 and 50 a minute at 10 held symbols) and anything else on
the host's IP. Measured before D2 (code of master 64cef323): see the D2 section of docs/arch/execution-log.md.
"""
from __future__ import annotations

import collections
import sqlite3
from contextlib import contextmanager
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace

from tests.support.state_db import state_dir_db
from tests.unit.binance.fake_venue import FakeBinance
from trading.adapters.binance.adapter import BinanceAdapter, BinanceConfig
from trading.adapters.binance.rate_limit import endpoint_weight
from trading.adapters.files.event_log import JsonlEventLog
from trading.adapters.files.kill_switch import KillFile
from trading.adapters.sqlite import connection, drawdown_store, entry_pause, funding_store, limit_store, order_journal
from trading.application.equity import EquitySource
from trading.application.exit_supervisor import ExitSupervisor
from trading.domain.orders import OrderIntent
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure import target
from trading.infrastructure.config import risk_limits

SCOPE = "binance-testnet-ut"
UNIVERSE = 809
HELD = 10
CLIENT_BUDGET_PER_MINUTE = 1200
SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")


class Counting:
    """The REST client: every request is kept (method, path, params) and weighed as the client's pool weighs it."""

    def __init__(self, fake: FakeBinance) -> None:
        self.fake, self.requests = fake, []
        self.weights: collections.Counter = collections.Counter()

    def _count(self, method: str, path: str, params) -> None:
        self.requests.append((method, path, dict(params or {})))
        self.weights[(method, path, "symbol" if (params or {}).get("symbol") else "account")] += \
            endpoint_weight(path, params)

    def request_public(self, *, method, path, params=None):
        self._count(method, path, params)
        return self.fake.request_public(method=method, path=path, params=params)

    def request_signed(self, *, method, path, params=None):
        self._count(method, path, params)
        return self.fake.request_signed(method=method, path=path, params=params)

    def clear(self) -> None:
        self.requests.clear()
        self.weights.clear()

    def of(self, symbol: str) -> list[tuple[str, str, dict]]:
        """The requests that name ``symbol``."""
        return [r for r in self.requests if r[2].get("symbol") == symbol]


def keep_connections(monkeypatch) -> None:
    """Keep the state database's connections open between uses. 809 sessions read it about 30 times each per cycle,
    and a connection opened for every read (about 1 ms on the build host) would make this test take many minutes.
    Every statement and transaction is as in production (``trading.adapters.sqlite.connection.connect``: the same
    URI, settings and ``with`` transaction; a nested use gets its own connection); only the opening and closing are
    saved. That cost at this size is plan D8's, not the request budget's."""
    kept: dict[tuple[str, int], sqlite3.Connection] = {}
    depth = [0]

    @contextmanager
    def connect(path):
        key = (str(Path(path).resolve()), depth[0])
        db = kept.get(key)
        if db is None:
            db = kept[key] = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=rw", uri=True,
                                             timeout=connection.BUSY_TIMEOUT_MS / 1000)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
        depth[0] += 1
        try:
            with db:
                yield db
        finally:
            depth[0] -= 1
    for module in (connection, drawdown_store, entry_pause, funding_store, limit_store, order_journal):
        monkeypatch.setattr(module, "connect", connect)


def exit_world(tmp_path: Path, symbols: tuple[str, ...], held: tuple[str, ...], monkeypatch) -> SimpleNamespace:
    """The exit side over ``symbols``, each of ``held`` holding a long position (entered through its own executor,
    as a session's entry goes after the gate) with its exchange stop in place."""
    monkeypatch.setattr(target, "ENTRY_WAIT_SECONDS", 0.0)     # no real sleep after each placement
    clock = {"now": 1_800_000_000.0}
    fake = FakeBinance(clock=lambda: clock["now"])
    fake.listed = symbols
    for k, symbol in enumerate(symbols):
        fake.price[symbol] = D(100 + k % 50)
    client = Counting(fake)
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=client)
    assert adapter.connect(symbols=symbols)
    state = tmp_path / "exec"
    state_dir_db(state)
    kill = KillFile(state / "kill_switch.json", missing_is_active=False)
    events = JsonlEventLog(state / "target_events.jsonl")
    now = lambda: clock["now"]          # noqa: E731
    equity = EquitySource(now)
    inputs = target.ExecutionInputs(account=adapter, scope=SCOPE, stop_pct=D("0.08"), limits=risk_limits({}),
                                    kill=kill, leverage=3, margin=None)
    wiring = target.execution_sessions(SPEC, list(symbols), state_dir=state, journal=events, equity=equity,
                                       inputs=inputs, clock=now)
    check = target.account_checker(adapter, state_dir=state, scope=SCOPE, symbols=list(symbols), leverage=3,
                                   margin=None)
    exits = ExitSupervisor(wiring.sessions, spec=SPEC, journal=events, clock=now, account=adapter, equity=equity,
                           kill=kill, drawdown=wiring.drawdown, pauses=wiring.pauses, notifier=None, alert_prefix="",
                           market_refused=lambda: None, decision_error=lambda symbol: None, cycle_errors=lambda: [],
                           account_check=check)
    world = SimpleNamespace(clock=clock, fake=fake, client=client, adapter=adapter, kill=kill, exits=exits,
                            sessions=wiring.sessions, state=state)
    exits.refresh_equity()
    exits.reconcile()                                   # a clean reconciliation readies the entries
    for symbol in held:
        session = wiring.sessions[symbol]
        session.executor.send_order(OrderIntent(order_id=f"{symbol}:entry", source=f"target.{symbol}", symbol=symbol,
                                                side="buy", qty=D("1"), price=fake.price[symbol], reduce_only=False,
                                                reason="test"))
        session.consume_fills()
        assert fake.positions[symbol] == D("1")
    cycles(world, 1)                                    # the stops are placed
    return world


def cycles(world: SimpleNamespace, n: int) -> None:
    """``n`` 5-second cycles of the runtime's requests outside the market side's hourly reads: the equity refresh
    (once a minute) and the exit side."""
    for _ in range(n):
        world.clock["now"] += 5
        world.exits.refresh_equity_if_due()
        world.exits.run()


def active_stops(world: SimpleNamespace) -> set[str]:
    return {a["symbol"] for a in world.fake.algo.values() if a["algoStatus"] == "NEW"}


def _minute(world: SimpleNamespace) -> float:
    world.client.clear()
    cycles(world, 12)
    return sum(world.client.weights.values())


def test_809_symbols_10_held_stay_inside_the_budget_in_normal_cycles_and_in_a_kill_flatten(tmp_path, monkeypatch):
    keep_connections(monkeypatch)
    symbols = tuple(f"S{k:03d}USDT" for k in range(UNIVERSE))
    held = symbols[::UNIVERSE // HELD][:HELD]              # spread over the universe, not the first ten
    world = exit_world(tmp_path, symbols, held, monkeypatch)
    assert active_stops(world) == set(held)
    normal = _minute(world)
    by_kind = sorted(world.client.weights.items(), key=lambda kv: -kv[1])
    print("normal weight per minute", normal, by_kind)
    assert active_stops(world) == set(held)              # every held position keeps its stop
    assert {r[2]["symbol"] for r in world.client.requests if r[2].get("symbol")} == set(held)
    assert normal <= CLIENT_BUDGET_PER_MINUTE / 2, (normal, by_kind)

    world.kill.write(active=True, reason="test", flatten=True)
    flatten = _minute(world)
    by_kind = sorted(world.client.weights.items(), key=lambda kv: -kv[1])
    print("kill-flatten weight per minute", flatten, by_kind)
    assert all(world.fake.positions.get(symbol, D(0)) == 0 for symbol in symbols)    # every position closed
    assert active_stops(world) == set()                  # and the stops it no longer needs are gone
    assert {r[2]["symbol"] for r in world.client.requests if r[2].get("symbol")} == set(held)
    assert flatten <= CLIENT_BUDGET_PER_MINUTE / 2, (flatten, by_kind)
