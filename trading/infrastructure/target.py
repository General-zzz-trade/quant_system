"""Composition root of the target runtime's sessions (migration step 10). It decides nothing about trading.

An execution session is built from the durable order journal, the order lifecycle, the execution service,
the owner's executor, its fill consumer, the protective stop manager and the entry gates. What still lives
in the legacy packages is passed in by the caller: the order limiter, the kill switch, the
transaction-cost log and, for paper trading, the research sleeve.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.equity import EquitySource
from trading.application.execution.executor import OwnerExecutor
from trading.application.execution.fills import TargetFillConsumer
from trading.application.execution.lifecycle import OrderLifecycle
from trading.application.execution.service import ExecutionService
from trading.application.execution.stops import ProtectiveStopManager, StopVenue, TriggeredOrder
from trading.application.ports import EventLog, ExchangeAccount, KillSwitch, OrderGateway, OrderLimits
from trading.application.risk.entry_gate import CompositeRiskGate
from trading.application.risk.exposure_gate import AccountExposureGate
from trading.application.risk.readiness import EntryReadinessGate
from trading.application.session import ExecutionParts, LotRules, OrderFactory, TargetSession
from trading.domain.orders import OrderIntent
from trading.domain.rebalance import Lot
from trading.domain.risk.exposure import MAX_TOTAL_LEVERAGE
from trading.domain.risk.liquidation import check_stop_inside_liquidation
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure.settings import RiskLimits, exchange_settings

ENTRY_WAIT_SECONDS = 0.5
NOT_FOUND_GRACE_SECONDS = 90.0      # > Binance max recvWindow (60 s) + clock skew margin
LIMIT_OFFSET = Decimal("0.0002")    # a maker entry rests 2 bp inside the reference, as research assumes
HOUR_SECONDS = 3600.0
ORDERS_FILE, DRAWDOWN_FILE = "orders.sqlite", "drawdown.sqlite"


class Account(ExchangeAccount, OrderGateway, StopVenue, Protocol):
    """The connected exchange account: snapshot, orders and stop orders."""

    def get_order(self, symbol: str, *, order_id: str = "",
                  client_order_id: str = "") -> TriggeredOrder | None: ...


@dataclass(frozen=True)
class ExecutionInputs:
    account: Account
    scope: str
    stop_pct: Decimal
    limits: RiskLimits
    limiter: OrderLimits
    kill: KillSwitch
    tca_log: EventLog | None = None


@dataclass(frozen=True)
class ExecutionWiring:
    sessions: dict[str, TargetSession]
    drawdown: SqliteDrawdownGuard


def refuse_unverified_mainnet(*, testnet: bool, spec_passed: object) -> None:
    if not testnet and spec_passed is not True:
        raise RuntimeError("research gate not passed; mainnet execution refused")


def lot_rules(account: ExchangeAccount | None, symbols: Sequence[str],
              floor: Mapping[str, LotRules]) -> dict[str, LotRules]:
    """Lot step and minimums per symbol: the exchange's filters when an account is connected, with the
    minimum notional never below the research floor, so an order research would skip is skipped here."""
    rules: dict[str, LotRules] = {}
    for symbol in symbols:
        rules[symbol] = floor[symbol]
        if account is not None:
            f = account.filters(symbol)
            rules[symbol] = Lot(f.step_size, f.min_qty, max(f.min_notional, floor[symbol].min_notional))
    return rules


def paper_sessions(spec: StrategySpec, symbols: Sequence[str], *, journal: EventLog,
                   rules: Mapping[str, LotRules], make_order: OrderFactory,
                   sleeve: Callable[[str, LotRules], object]) -> dict[str, TargetSession]:
    allocation = Decimal(1) / Decimal(len(symbols))
    return {symbol: TargetSession(symbol, spec, journal=journal, rules=rules[symbol], make_order=make_order,
                                  allocation=allocation, sleeve=sleeve(symbol, rules[symbol]))
            for symbol in symbols}


def execution_sessions(spec: StrategySpec, symbols: Sequence[str], *, state_dir: Path, journal: EventLog,
                       lot_floor: Mapping[str, LotRules], equity: EquitySource,
                       inputs: ExecutionInputs) -> ExecutionWiring:
    leverage, margin = exchange_settings()
    check_stop_inside_liquidation(float(inputs.stop_pct), leverage, margin)   # before anything is created
    account, scope, limits = inputs.account, inputs.scope, inputs.limits
    rules = lot_rules(account, symbols, lot_floor)
    orders_path = Path(state_dir) / ORDERS_FILE
    drawdown = SqliteDrawdownGuard(Path(state_dir) / DRAWDOWN_FILE, scope, daily_pct=limits.daily_drawdown_pct,
                                   weekly_pct=limits.weekly_drawdown_pct)
    exposure = AccountExposureGate(account, SqliteOrderJournal(orders_path, scope),
                                   max_leverage=min(MAX_TOTAL_LEVERAGE, limits.max_leverage))
    gate = CompositeRiskGate.for_execution(inputs.limiter, kill_switch=inputs.kill, drawdown_guard=drawdown,
                                           portfolio_reader=exposure, venue="binance")
    maker = spec.execution == "maker"
    settings: dict[str, Any] = dict(
        wait_seconds=ENTRY_WAIT_SECONDS, entry_order_type="limit" if maker else "market",
        not_found_grace_seconds=NOT_FOUND_GRACE_SECONDS,
        entry_ttl_seconds=spec.ttl_bars * HOUR_SECONDS if maker else None)
    OrderLifecycle.check_settings(scope, **settings)      # before the journal file is created
    allocation = Decimal(1) / Decimal(len(symbols))
    sessions: dict[str, TargetSession] = {}
    for symbol in symbols:
        owner = f"target.{symbol}"
        lifecycle = OrderLifecycle(account, SqliteOrderJournal(orders_path, scope), scope, clock=time.time,
                                   sleeper=lambda seconds: time.sleep(seconds), lock=threading.RLock(),
                                   owner=owner, **settings)
        service = ExecutionService(account, lifecycle, owner=owner, entry_mode=spec.execution,
                                   limit_offset=LIMIT_OFFSET, lock=threading.RLock(), clock=time.time,
                                   monotonic=time.monotonic, event_log=inputs.tca_log)
        executor = OwnerExecutor(service, scope=scope)
        stops = ProtectiveStopManager(account, executor, symbol, stop_pct=inputs.stop_pct, clock=time.time)
        session = TargetSession(symbol, spec, journal=journal, rules=rules[symbol], make_order=OrderIntent,
                                allocation=allocation,
                                execution=ExecutionParts(executor, TargetFillConsumer(executor), stops, None, equity))
        executor.on_status(session._on_status)
        session.gate = EntryReadinessGate(session, gate)
        sessions[symbol] = session
    return ExecutionWiring(sessions, drawdown)


__all__ = ["Account", "ExecutionInputs", "ExecutionWiring", "execution_sessions", "lot_rules", "paper_sessions",
           "refuse_unverified_mainnet"]
