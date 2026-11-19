"""Composition root of the target runtime's sessions (migration step 10). It decides nothing about trading.

A session is built from the durable order journal, the order lifecycle, the execution service, the owner's
executor, its fill consumer, the protective stop manager and the entry gates. Paper builds the same sessions over
the simulated exchange (ADR-0013 plan B, S10 part 3); the ``ModePolicy`` decides what differs and ``PaperParts``
supplies what only paper has. What lives outside ``trading`` is passed in by the caller: the transaction-cost log.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from trading.adapters.binance.adapter import MARGIN_TYPES
from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.adapters.sqlite.funding_store import SqliteFundingStore
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
from trading.application.ports import (AccountSnapshot, Clock, ContractInfo, EntryPauses, EventLog, ExchangeAccount,
                                       FundingIncome, KillLatch, OrderGateway, Sleeper)
from trading.application.risk.entry_gate import CompositeRiskGate
from trading.application.risk.exposure_gate import AccountExposureGate
from trading.application.risk.loss_stop import StrategyLossStop, strategy_unrealized
from trading.application.risk.order_limiter import OrderLimiter
from trading.application.risk.readiness import EntryReadinessGate
from trading.application.session import ExecutionParts, LotRules, SizingEquity, TargetSession
from trading.application.execution.recorded_journal import RecordedOrderJournal
from trading.application.reset_wait import waits_for_reset
from trading.domain.contracts import contract_problems, planned_notional  # noqa: F401 (the start-up's)
from trading.domain.issues import AFTER_STOP_PAUSE
from trading.domain.orders import LIMIT_OFFSET_BPS, OrderIntent
from trading.domain.strategy.base import Strategy
from trading.domain.rebalance import Lot
from trading.domain.risk.order_limits import OrderLimitsConfig
from trading.domain.risk.exposure import MAX_TOTAL_LEVERAGE
from trading.domain.risk.liquidation import check_stop_inside_liquidation
from trading.domain.strategy.overlay import StrategySpec
from trading.domain.strategy.spec import MAINNET_SYMBOLS
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
    strategy_leverage: Decimal = Decimal(1)     # owner 2026-11-16: a full position, in multiples of the capital


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


def refuse_mainnet_symbols(*, testnet: bool, symbols: Sequence[str]) -> None:
    """Mainnet trades BTCUSDT and ETHUSDT only (owner 2026-11-17, ``MAINNET_SYMBOLS``); the testnet stays general."""
    other = sorted(set(symbols) - MAINNET_SYMBOLS)
    if not testnet and other:
        raise RuntimeError(f"mainnet trades only BTCUSDT and ETHUSDT (owner 2026-11-17); the spec also lists "
                           f"{', '.join(other)}; mainnet execution refused")


def held_symbols(ledger: SqliteOrderJournal, symbols: Sequence[str]) -> set[str]:
    """The symbols any ``target.*`` owner of the scope holds in the ledger; a ledger that cannot be read raises."""
    held = set()
    for owner in strategy_owners(ledger, symbols):
        checkpoint = ledger.checkpoint(owner)
        if checkpoint is not None and Decimal(str(checkpoint["position"]["qty"])) != 0:
            held.add(owner[len(OWNER_PREFIX):])
    return held


def refuse_unsafe_contracts(account: ContractInfo, *, symbols: Sequence[str], stop_pct: Decimal, leverage: int,
                            margin: str | None, notional: Decimal, held: Collection[str] = (),
                            pause: Callable[[str, str], object] | None = None) -> dict[str, list[str]]:
    """Plan D1: every symbol must be a trading USDT perpetual (crypto or TradFi) with no delisting scheduled, allow
    the configured exchange leverage at its first tier and at the planned position's tier, and, with isolated margin,
    keep the stop inside half the liquidation distance at that tier's maintenance margin rate. Read-only.

    With nothing held an unfit symbol (or unreadable tiers) refuses the start. While the strategy holds a position
    (``held``) a refusal would leave it with nobody to reduce, flatten or stop it (audit of D1, M1): every unfit
    symbol's entries are paused instead, until a person releases them, and the problems are returned for the
    alert; reductions and exits go on."""
    contracts = account.contracts()
    try:
        tiers = account.get_margin_tiers()
    except Exception as exc:
        if not held or pause is None:
            raise RuntimeError(f"margin tiers could not be read ({type(exc).__name__}: {exc}); execution refused") \
                from exc
        tiers = {}
    problems = {symbol: found for symbol in symbols
                if (found := contract_problems(symbol, contracts.get(symbol), tiers.get(symbol), stop_pct=stop_pct,
                                               leverage=leverage, margin=margin, notional=notional))}
    if not problems:
        return {}
    if not held or pause is None:
        raise RuntimeError("contracts not fit to trade; execution refused: "
                           + "; ".join(p for found in problems.values() for p in found))
    for symbol, found in problems.items():
        pause(symbol, "contract not fit to trade (plan D1): " + "; ".join(found))
    return problems


class StartAccount(ContractInfo, AccountSnapshot, Protocol):
    """What the start-up's contract gate reads: the contracts, the margin tiers and the venue's positions."""


def contract_gate(account: StartAccount, *, state_dir: Path, scope: str, symbols: Sequence[str], stop_pct: Decimal,
                  leverage: int, margin: str | None, notional: Decimal) -> dict[str, list[str]]:
    """``refuse_unsafe_contracts`` with the scope's ledger and entry pauses (the start-up's call, after the account
    check so a refused start leaves no pause behind - re-audit of D1, N4). Held: a non-zero ledger position, an
    unsettled strategy order (a fill not yet booked, N1) or a venue position on a spec symbol. Every failure to read
    them or to write a pause is a refusal (N3)."""
    ledger = SqliteOrderJournal(Path(state_dir) / STATE_DB, scope)
    pauses = SqliteEntryPauses(Path(state_dir) / STATE_DB)
    try:
        held = held_symbols(ledger, symbols)
        held |= {owner[len(OWNER_PREFIX):] for owner in strategy_owners(ledger, symbols)
                 if ledger.orders(owner, unresolved=True)}
        held |= {p.symbol for p in account.get_positions(strict=True) if p.symbol in symbols and p.qty != 0}
    except Exception as exc:
        raise RuntimeError(f"the strategy ledger or the venue's positions could not be read ({type(exc).__name__}: "
                           f"{exc}); execution refused") from exc

    def pause(symbol: str, cause: str) -> None:
        try:
            pauses.pause(symbol, cause)
        except Exception as exc:
            raise RuntimeError(f"{symbol}: the entry pause could not be written ({type(exc).__name__}: {exc}); "
                               "execution refused") from exc
    return refuse_unsafe_contracts(account, symbols=symbols, stop_pct=stop_pct, leverage=leverage, margin=margin,
                                   notional=notional, held=held, pause=pause)


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


def strategy_owners(ledger: SqliteOrderJournal, symbols: Sequence[str]) -> list[str]:
    """Every ``target.*`` owner of the scope: the spec's symbols and whatever the ledger holds of an earlier spec
    (audit 2026-11-16, finding 7). A ledger that cannot be listed raises."""
    return sorted({OWNER_PREFIX + s for s in symbols} | {o for o in ledger.owners() if o.startswith(OWNER_PREFIX)})


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
                       policy: ModePolicy = EXECUTION, paper: PaperParts | None = None,
                       strategy: Strategy | None = None, allow_short: bool | None = None,
                       after_stop: str = AFTER_STOP_PAUSE) -> ExecutionWiring:
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
        if stops is None or bool(set(stops.triggers()) - pauses.covered(symbol)):
            return True
        # The wait for the strategy's reset after a stop (tech lead 2026-11-17), here too, so a refused entry never
        # reaches the order limiter's daily counts (audit of PR #142, finding 1). A read that fails raises: refused.
        session = sessions.get(symbol)
        return session is not None and waits_for_reset(session, pauses)

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
    if paper is None and inputs.strategy_capital is not None:
        allocation *= inputs.strategy_leverage           # the owner's multiple of the capital (1 to 3)
    sleeper: Sleeper = paper.sleeper if paper is not None else (lambda seconds: time.sleep(seconds))
    capped: SizingEquity | None = None
    loss_stop: StrategyLossStop | None = None
    if paper is None and inputs.strategy_capital is not None:
        # The base's PnL: the ledger's realized PnL after USDT fees plus the funding income of the strategy's symbols
        # since its first fill on each (tech lead 2026-11-16); either one unknown sizes to zero.
        if inputs.funding is None:
            raise ValueError("the strategy capital base needs the account's funding income")
        # Every target.* owner of the scope counts, also one of an earlier spec (audit 2026-11-16, finding 7).
        ledger = SqliteOrderJournal(orders_path, scope)

        def owners() -> list[str]:
            return strategy_owners(ledger, symbols)
        # The funding rows are kept in the state database (version 2, tech lead 2026-11-17): no 88-day limit.
        funding = StrategyFunding(inputs.funding, lambda: [o[len(OWNER_PREFIX):] for o in owners()],
                                  lambda s: ledger.first_fill_at(OWNER_PREFIX + s), clock,
                                  store=SqliteFundingStore(orders_path, scope))
        capped = CappedEquity(equity, inputs.strategy_capital,
                              lambda: strategy_realized_pnl(ledger, owners()) + funding.current())
        if inputs.strategy_loss_stop is not None:
            def strategy_pnl() -> Decimal:
                """The loss stop also counts what is still open: each owner's ledger position at the venue's mark
                for its symbol (never the venue's own quantity, which a person's position is netted into). Its
                funding is the last total read in this process, however old: only the sizing base ages it."""
                held = owners()
                return (strategy_realized_pnl(ledger, held) + funding.last_known()
                        + strategy_unrealized(strategy_positions(ledger, held), account.get_position))
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
                                allocation=allocation, strategy=strategy, allow_short=allow_short,
                                after_stop=after_stop,
                                execution=ExecutionParts(executor, TargetFillConsumer(executor, journal), stops, None,
                                                         sizing, Flattener(symbol, executor, account, journal, clock)))
        session.entered_since = partial(orders.entry_filled_since, owner)
        session.restore_pending()
        executor.on_status(session._on_status)
        session.gate = EntryReadinessGate(session, gate)
        sessions[symbol] = session
    return ExecutionWiring(sessions, drawdown, pauses, loss_stop)


__all__ = ["Account", "ExecutionInputs", "ExecutionWiring", "OWNER_PREFIX", "PaperParts", "exchange_lot_rules",
           "execution_sessions", "refuse_foreign_holdings", "refuse_mainnet_symbols", "refuse_unverified_mainnet",
           "strategy_owners", "strategy_positions", "strategy_realized_pnl"]
