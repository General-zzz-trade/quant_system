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

from trading.adapters.binance.adapter import MARGIN_TYPES
from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.adapters.sqlite.limit_store import SqliteLimitStore
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.adapters.sqlite.schema import STATE_DB
from trading.application.equity import EquitySource
from trading.application.execution.executor import OwnerExecutor
from trading.application.execution.fills import TargetFillConsumer
from trading.application.execution.flatten import Flattener
from trading.application.execution.lifecycle import OrderLifecycle
from trading.application.execution.service import ExecutionService
from trading.application.execution.stops import ProtectiveStopManager, StopVenue, TriggeredOrder
from trading.application.account_ownership import foreign_holdings
from trading.application.ports import AccountSnapshot, Clock, EventLog, ExchangeAccount, KillSwitch, OrderGateway
from trading.application.risk.entry_gate import CompositeRiskGate
from trading.application.risk.exposure_gate import AccountExposureGate
from trading.application.risk.order_limiter import OrderLimiter
from trading.application.risk.readiness import EntryReadinessGate
from trading.application.session import ExecutionParts, LotRules, OrderFactory, TargetSession
from trading.application.execution.recorded_journal import RecordedOrderJournal
from trading.domain.orders import OrderIntent
from trading.domain.rebalance import Lot
from trading.domain.risk.order_limits import OrderLimitsConfig
from trading.domain.risk.exposure import MAX_TOTAL_LEVERAGE
from trading.domain.risk.liquidation import check_stop_inside_liquidation
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure.settings import RiskLimits, exchange_settings

ENTRY_WAIT_SECONDS = 0.5
NOT_FOUND_GRACE_SECONDS = 90.0      # > Binance max recvWindow (60 s) + clock skew margin
LIMIT_OFFSET = Decimal("0.0002")    # a maker entry rests 2 bp inside the reference, as research assumes
HOUR_SECONDS = 3600.0


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
    kill: KillSwitch
    tca_log: EventLog | None = None


@dataclass(frozen=True)
class ExecutionWiring:
    sessions: dict[str, TargetSession]
    drawdown: SqliteDrawdownGuard
    pauses: SqliteEntryPauses


def refuse_unverified_mainnet(*, testnet: bool, spec_passed: object) -> None:
    if not testnet and spec_passed is not True:
        raise RuntimeError("research gate not passed; mainnet execution refused")


def refuse_foreign_holdings(account: AccountSnapshot, *, state_dir: Path, scope: str, symbols: Sequence[str]) -> None:
    """ADR-0010: execution starts only if the account holds nothing outside the strategy ledger and the leverage
    and margin mode of the strategy's symbols are as configured. Read-only; run before the runtime is built."""
    leverage, margin = exchange_settings()
    findings = foreign_holdings(account, SqliteOrderJournal(Path(state_dir) / STATE_DB, scope),
                                symbols=tuple(symbols), leverage=leverage,
                                margin=MARGIN_TYPES[margin] if margin else MARGIN_TYPES["CROSSED"])
    if findings:
        raise RuntimeError("execution refused (ADR-0010): " + "; ".join(findings) + ". Close or cancel what the "
                           "strategy does not own on Binance, or set the configured leverage and margin mode there, "
                           "then start again; the runtime changes neither")


def exchange_lot_rules(account: ExchangeAccount, symbols: Sequence[str]) -> dict[str, LotRules]:
    """Lot step and minimums per symbol exactly as the exchange publishes them (decision of 2026-11-14).

    Research and paper keep their own constants, which are higher for BTC; an order between the two
    minimums is therefore sent here and skipped there. That only happens on small accounts."""
    rules: dict[str, LotRules] = {}
    for symbol in symbols:
        f = account.filters(symbol)
        rules[symbol] = Lot(f.step_size, f.min_qty, f.min_notional)
    return rules


def paper_sessions(spec: StrategySpec, symbols: Sequence[str], *, journal: EventLog,
                   rules: Mapping[str, LotRules], make_order: OrderFactory,
                   sleeve: Callable[[str, LotRules], object]) -> dict[str, TargetSession]:
    allocation = Decimal(1) / Decimal(len(symbols))
    return {symbol: TargetSession(symbol, spec, journal=journal, rules=rules[symbol], make_order=make_order,
                                  allocation=allocation, sleeve=sleeve(symbol, rules[symbol]))
            for symbol in symbols}


def execution_sessions(spec: StrategySpec, symbols: Sequence[str], *, state_dir: Path, journal: EventLog,
                       equity: EquitySource, inputs: ExecutionInputs, clock: Clock = time.time) -> ExecutionWiring:
    leverage, margin = exchange_settings()
    check_stop_inside_liquidation(float(inputs.stop_pct), leverage, margin)   # before anything is created
    account, scope, limits = inputs.account, inputs.scope, inputs.limits
    rules = exchange_lot_rules(account, symbols)
    orders_path = Path(state_dir) / STATE_DB         # one database for every store (ADR-0009)
    drawdown = SqliteDrawdownGuard(orders_path, scope, daily_pct=limits.daily_drawdown_pct,
                                   weekly_pct=limits.weekly_drawdown_pct, clock=clock)
    exposure = AccountExposureGate(account, SqliteOrderJournal(orders_path, scope),
                                   max_leverage=min(MAX_TOTAL_LEVERAGE, limits.max_leverage))
    pauses = SqliteEntryPauses(orders_path, clock=clock)
    stops_of: dict[str, ProtectiveStopManager] = {}

    def paused(symbol: str) -> bool:
        """Paused in the store, or by a stop trigger no pause covers yet. The runtime writes that pause at its
        next reconciliation; until then a lost or emptied store must not let an entry through (audit after M3)."""
        if pauses.reason(symbol) is not None:
            return True
        stops = stops_of.get(symbol)
        return stops is None or bool(set(stops.triggers()) - pauses.covered(symbol))

    limiter = OrderLimiter(OrderLimitsConfig(max_order_notional=limits.max_order_notional,
                                             max_daily_notional=limits.max_daily_notional,
                                             max_daily_orders=limits.max_daily_orders, max_orders_per_second=1.0),
                           store=SqliteLimitStore(orders_path, scope), clock=clock,
                           monotonic=time.monotonic)
    gate = CompositeRiskGate.for_execution(limiter, kill_switch=inputs.kill, drawdown_guard=drawdown,
                                           portfolio_reader=exposure, venue="binance",
                                           paused=paused)
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
        # One clock for every part (step 13a): the runtime's. Waiting and latency stay on real time.
        # Every order and stop status change is also an event (2026-11-15); the journal stays the state.
        lifecycle = OrderLifecycle(account, RecordedOrderJournal(SqliteOrderJournal(orders_path, scope), journal),
                                   scope, clock=clock,
                                   sleeper=lambda seconds: time.sleep(seconds), lock=threading.RLock(),
                                   owner=owner, **settings)
        service = ExecutionService(account, lifecycle, owner=owner, entry_mode=spec.execution,
                                   limit_offset=LIMIT_OFFSET, lock=threading.RLock(), clock=clock,
                                   monotonic=time.monotonic, event_log=inputs.tca_log)
        executor = OwnerExecutor(service, scope=scope)
        stops = ProtectiveStopManager(account, executor, symbol, stop_pct=inputs.stop_pct, clock=clock)
        stops_of[symbol] = stops
        session = TargetSession(symbol, spec, journal=journal, rules=rules[symbol], make_order=OrderIntent,
                                allocation=allocation,
                                execution=ExecutionParts(executor, TargetFillConsumer(executor), stops, None, equity,
                                                         Flattener(symbol, executor, account, journal, clock)))
        executor.on_status(session._on_status)
        session.gate = EntryReadinessGate(session, gate)
        sessions[symbol] = session
    return ExecutionWiring(sessions, drawdown, pauses)


__all__ = ["Account", "ExecutionInputs", "ExecutionWiring", "exchange_lot_rules", "execution_sessions",
           "paper_sessions", "refuse_foreign_holdings", "refuse_unverified_mainnet"]
