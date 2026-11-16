"""Composition root of the target runtime's sessions (migration step 10). It decides nothing about trading.

A session is built from the durable order journal, the order lifecycle, the execution service, the owner's
executor, its fill consumer, the protective stop manager and the entry gates. Paper builds the same sessions over
the simulated exchange (ADR-0013 plan B, S10 part 3); the ``ModePolicy`` decides what differs and ``PaperParts``
supplies what only paper has. What lives outside ``trading`` is passed in by the caller: the transaction-cost log.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from trading.adapters.binance.adapter import MARGIN_TYPES
from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.adapters.sqlite.limit_store import SqliteLimitStore
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.adapters.sqlite.paper_state import PaperOrderJournal, SqlitePaperState
from trading.adapters.sqlite.schema import STATE_DB
from trading.application.equity import CappedEquity, EquitySource, StrategyFunding
from trading.application.execution.executor import OwnerExecutor
from trading.application.execution.fills import TargetFillConsumer
from trading.application.execution.flatten import Flattener
from trading.application.execution.lifecycle import OrderLifecycle
from trading.application.execution.service import ExecutionService
from trading.application.execution.stops import ProtectiveStopManager, StopVenue, TriggeredOrder
from trading.application.account_ownership import foreign_holdings
from trading.application.issues import Issue
from trading.application.mode_policy import ModePolicy
from trading.application.ports import (AccountSnapshot, Clock, EntryPauses, EventLog, ExchangeAccount, FundingIncome,
                                       KillLatch, OrderGateway, Sleeper)
from trading.application.risk.entry_gate import CompositeRiskGate
from trading.application.risk.exposure_gate import AccountExposureGate
from trading.application.risk.loss_stop import StrategyLossStop, strategy_unrealized
from trading.application.risk.order_limiter import OrderLimiter
from trading.application.risk.readiness import EntryReadinessGate
from trading.application.session import ExecutionParts, LotRules, SizingEquity, TargetSession
from trading.application.execution.recorded_journal import RecordedOrderJournal
from trading.domain.orders import LIMIT_OFFSET_BPS, OrderIntent
from trading.domain.rebalance import Lot
from trading.domain.risk.order_limits import OrderLimitsConfig
from trading.domain.risk.exposure import MAX_TOTAL_LEVERAGE
from trading.domain.risk.liquidation import check_stop_inside_liquidation
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure.config import RiskLimits
from trading.infrastructure.modes import EXECUTION

ENTRY_WAIT_SECONDS = 0.5
NOT_FOUND_GRACE_SECONDS = 90.0      # > Binance max recvWindow (60 s) + clock skew margin
LIMIT_OFFSET = LIMIT_OFFSET_BPS / 10000    # the same maker offset as research, as the fraction the service takes
HOUR_SECONDS = 3600.0
OWNER_PREFIX = "target."            # the strategy's ledger owners: target.<SYMBOL>


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
    kill: KillLatch
    leverage: int                       # the exchange settings the configuration expects (RuntimeConfig)
    margin: str | None
    tca_log: EventLog | None = None
    strategy_capital: Decimal | None = None   # owner 2026-11-16: size from min(account, this + own realized PnL)
    funding: FundingIncome | None = None      # the account's funding income, which that PnL includes (2026-11-16)
    strategy_loss_stop: Decimal | None = None   # owner 2026-11-16: at this share of the capital lost, flatten and stop


@dataclass(frozen=True)
class ExecutionWiring:
    sessions: dict[str, TargetSession]
    drawdown: SqliteDrawdownGuard
    pauses: EntryPauses
    loss_stop: StrategyLossStop | None = None


@dataclass(frozen=True)
class PaperParts:
    """What only paper plugs in (S10 part 3): each symbol's own simulated equity (``sizing_equity``), the gate clock on
    bar time (``limiter_clock``), no wait after a placement, the pause store that reports new pauses, and the paper
    state its order journal saves in every transaction (S10 decision 4)."""
    symbol_equity: Callable[[str], SizingEquity]
    gate_clock: Clock
    sleeper: Sleeper
    pauses: Callable[[EntryPauses], EntryPauses] = lambda pauses: pauses
    state: SqlitePaperState | None = None


def refuse_unverified_mainnet(*, testnet: bool, spec_passed: object) -> None:
    if not testnet and spec_passed is not True:
        raise RuntimeError("research gate not passed; mainnet execution refused")


def account_checker(account: AccountSnapshot, *, state_dir: Path, scope: str, symbols: Sequence[str], leverage: int,
                    margin: str | None) -> Callable[[], list[Issue]]:
    """The whole-account check of ADR-0010, as the runtime repeats it while running (ADR-0011)."""
    journal = SqliteOrderJournal(Path(state_dir) / STATE_DB, scope)
    spelled = MARGIN_TYPES[margin] if margin else MARGIN_TYPES["CROSSED"]
    return lambda: foreign_holdings(account, journal, symbols=tuple(symbols), leverage=leverage, margin=spelled)


def refuse_foreign_holdings(account: AccountSnapshot, *, state_dir: Path, scope: str, symbols: Sequence[str],
                            leverage: int, margin: str | None) -> None:
    """ADR-0010: execution starts only if the account holds nothing outside the strategy ledger and the leverage
    and margin mode of the strategy's symbols are as configured (RuntimeConfig, ADR-0017). Read-only; run before
    the runtime is built."""
    findings = account_checker(account, state_dir=state_dir, scope=scope, symbols=symbols, leverage=leverage,
                               margin=margin)()
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


def strategy_realized_pnl(ledger: SqliteOrderJournal, owners: list[str]) -> Decimal:
    """The strategy's cumulative realized PnL after its USDT fees, from its owners' ledger checkpoints (the strategy
    capital cap, owner 2026-11-16). A checkpoint that cannot be read raises: the cap then sizes to zero."""
    total = Decimal(0)
    for owner in owners:
        checkpoint = ledger.checkpoint(owner)
        if checkpoint is None:
            continue
        position = checkpoint["position"]
        total += Decimal(str(position["realized_pnl"])) - Decimal(str(position.get("fees", {}).get("USDT", "0")))
    return total


def strategy_positions(ledger: SqliteOrderJournal, owners: list[str]) -> list[tuple[str, Decimal, Decimal]]:
    """Each owner's ledger position as (symbol, quantity, entry price), from its checkpoint: what the loss stop
    values at the venue's mark. An owner without a checkpoint holds nothing; one that cannot be read raises."""
    rows = []
    for owner in owners:
        checkpoint = ledger.checkpoint(owner)
        if checkpoint is not None:
            position = checkpoint["position"]
            rows.append((owner[len(OWNER_PREFIX):], Decimal(str(position["qty"])),
                         Decimal(str(position["entry_price"]))))
    return rows


def execution_sessions(spec: StrategySpec, symbols: Sequence[str], *, state_dir: Path, journal: EventLog,
                       equity: EquitySource, inputs: ExecutionInputs, clock: Clock = time.time,
                       policy: ModePolicy = EXECUTION, paper: PaperParts | None = None) -> ExecutionWiring:
    """The sessions of either mode: paper differs only by ``policy`` and the ``paper`` parts it needs."""
    if (paper is None) != (policy.sizing_equity == "account_share" and policy.limiter_clock == "monotonic"):
        raise ValueError("per-symbol sizing and the bar-time gate clock need the paper parts, and only they")
    check_stop_inside_liquidation(float(inputs.stop_pct), inputs.leverage, inputs.margin)   # before anything exists
    account, scope, limits = inputs.account, inputs.scope, inputs.limits
    rules = exchange_lot_rules(account, symbols)
    orders_path = Path(state_dir) / STATE_DB         # one database for every store (ADR-0009)
    drawdown = SqliteDrawdownGuard(orders_path, scope, daily_pct=limits.daily_drawdown_pct,
                                   weekly_pct=limits.weekly_drawdown_pct, clock=clock)
    exposure = AccountExposureGate(account, SqliteOrderJournal(orders_path, scope),
                                   max_leverage=min(MAX_TOTAL_LEVERAGE, limits.max_leverage))
    pauses = SqliteEntryPauses(orders_path, clock=clock) if paper is None \
        else paper.pauses(SqliteEntryPauses(orders_path, clock=clock))
    gate_clock = paper.gate_clock if paper is not None else None
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
                           store=SqliteLimitStore(orders_path, scope), clock=gate_clock or clock,
                           monotonic=gate_clock or time.monotonic)
    gate = CompositeRiskGate.for_execution(limiter, kill_switch=inputs.kill, drawdown_guard=drawdown,
                                           portfolio_reader=exposure, venue="binance",
                                           paused=paused)
    maker = spec.execution == "maker"
    settings: dict[str, Any] = dict(
        wait_seconds=ENTRY_WAIT_SECONDS, entry_order_type="limit" if maker else "market",
        not_found_grace_seconds=NOT_FOUND_GRACE_SECONDS,
        entry_ttl_seconds=spec.ttl_bars * HOUR_SECONDS if maker else None)
    OrderLifecycle.check_settings(scope, **settings)      # before the journal file is created
    allocation = Decimal(1) / Decimal(len(symbols)) if policy.sizing_equity == "account_share" else Decimal(1)
    sleeper: Sleeper = paper.sleeper if paper is not None else (lambda seconds: time.sleep(seconds))
    capped: SizingEquity | None = None
    loss_stop: StrategyLossStop | None = None
    if paper is None and inputs.strategy_capital is not None:
        # The base's PnL: the ledger's realized PnL after USDT fees plus the funding income of the strategy's symbols
        # since its first fill on each (tech lead 2026-11-16); either one unknown sizes to zero.
        if inputs.funding is None:
            raise ValueError("the strategy capital base needs the account's funding income")
        ledger = SqliteOrderJournal(orders_path, scope)
        funding = StrategyFunding(inputs.funding, symbols, lambda s: ledger.first_fill_at(OWNER_PREFIX + s), clock)
        owners = [OWNER_PREFIX + s for s in symbols]
        capped = CappedEquity(equity, inputs.strategy_capital,
                              lambda: strategy_realized_pnl(ledger, owners) + funding.current())
        if inputs.strategy_loss_stop is not None:
            def strategy_pnl() -> Decimal:
                """The loss stop also counts what is still open: each owner's ledger position at the venue's mark
                for its symbol (never the venue's own quantity, which a person's position is netted into)."""
                return (strategy_realized_pnl(ledger, owners) + funding.current()
                        + strategy_unrealized(strategy_positions(ledger, owners), account.get_position))
            loss_stop = StrategyLossStop(inputs.strategy_capital, inputs.strategy_loss_stop, strategy_pnl,
                                         inputs.kill, clock=clock)
    sessions: dict[str, TargetSession] = {}
    for symbol in symbols:
        owner = OWNER_PREFIX + symbol
        # One clock for every part (step 13a): the runtime's. Waiting and latency stay on real time; in paper the
        # entry gates run on bar time (ModePolicy.limiter_clock) and nothing waits for the simulated venue.
        # Every order and stop status change is also an event (2026-11-15); the journal stays the state.
        flip = policy.entry_beside_working_reduction
        orders = SqliteOrderJournal(orders_path, scope, entry_beside_working_reduction=flip) \
            if paper is None or paper.state is None \
            else PaperOrderJournal(orders_path, scope, state=paper.state, entry_beside_working_reduction=flip)
        lifecycle = OrderLifecycle(account, RecordedOrderJournal(orders, journal), scope, clock=clock,
                                   sleeper=sleeper, lock=threading.RLock(), owner=owner,
                                   venue_expires_entries=policy.entry_ttl == "venue", **settings)
        service = ExecutionService(account, lifecycle, owner=owner, entry_mode=spec.execution,
                                   limit_offset=LIMIT_OFFSET, lock=threading.RLock(), clock=clock,
                                   monotonic=time.monotonic, event_log=inputs.tca_log)
        executor = OwnerExecutor(service, scope=scope)
        stops = ProtectiveStopManager(account, executor, symbol, stop_pct=inputs.stop_pct, clock=clock)
        stops_of[symbol] = stops
        sizing: SizingEquity = paper.symbol_equity(symbol) if paper is not None else (capped or equity)
        session = TargetSession(symbol, spec, journal=journal, rules=rules[symbol], make_order=OrderIntent,
                                allocation=allocation,
                                execution=ExecutionParts(executor, TargetFillConsumer(executor, journal), stops, None,
                                                         sizing, Flattener(symbol, executor, account, journal, clock)))
        session.restore_pending()
        executor.on_status(session._on_status)
        session.gate = EntryReadinessGate(session, gate)
        sessions[symbol] = session
    return ExecutionWiring(sessions, drawdown, pauses, loss_stop)


__all__ = ["Account", "ExecutionInputs", "ExecutionWiring", "OWNER_PREFIX", "PaperParts", "exchange_lot_rules",
           "execution_sessions", "refuse_foreign_holdings", "refuse_unverified_mainnet", "strategy_positions",
           "strategy_realized_pnl"]
