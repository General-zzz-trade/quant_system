"""Ports: the only way the application reaches the exchange, storage, files, clock and operators.

Method names and signatures follow the code that already implements them (see docs/arch/target.md
section 3), so the existing Binance adapter and the test venues conform without changes. A port method
that cannot answer raises; it never returns an empty result that could be mistaken for "nothing there".

What the exchange returns is typed (migration step 10b): records the application only reads are described
by the ``...View`` protocols below (the Binance adapter returns the frozen records of
``trading.domain.venue``; the frozen venues and the test venues return objects with the same attributes),
and placement results are the typed dictionaries of ``trading.domain.venue``.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from decimal import Decimal
from collections.abc import Iterable
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from trading.domain.bars import Bar
from trading.domain.venue import Placement, StopPlacement, Ticker
from trading.domain.events import EventKind

Clock = Callable[[], float]          # wall-clock seconds since the epoch
Sleeper = Callable[[float], None]
Lock = AbstractContextManager[object]   # re-entrant; supplied by the composition root (threading.RLock)


class PositionView(Protocol):
    """A position in the exchange's snapshot; ``qty`` is signed."""

    @property
    def symbol(self) -> str: ...

    @property
    def qty(self) -> Decimal: ...

    @property
    def mark_price(self) -> Decimal | None: ...


class OpenOrderView(Protocol):
    @property
    def order_id(self) -> str: ...


class OrderView(Protocol):
    """An order looked up at the exchange. Its symbol, side and state are read where present."""

    @property
    def order_id(self) -> str: ...

    @property
    def client_order_id(self) -> str: ...

    @property
    def filled_qty(self) -> Decimal: ...


class FillView(Protocol):
    """One exchange trade. ``trade_id``, ``fee`` and ``fee_asset`` are read where present."""

    @property
    def order_id(self) -> str: ...

    @property
    def symbol(self) -> str: ...

    @property
    def side(self) -> str: ...

    @property
    def qty(self) -> Decimal: ...

    @property
    def price(self) -> Decimal: ...


class LotFilters(Protocol):
    @property
    def step_size(self) -> Decimal: ...

    @property
    def min_qty(self) -> Decimal: ...

    @property
    def min_notional(self) -> Decimal: ...


class StopView(Protocol):
    """A protective stop at the exchange: ``state`` is one of the three states of trading.domain.venue."""

    @property
    def state(self) -> str: ...

    @property
    def actual_order_id(self) -> str: ...


@runtime_checkable
class MarketData(Protocol):
    """Public market data. Only closed bars and settled funding are returned."""

    def get_klines(self, symbol: str, interval: str = "60", limit: int = 200,
                   *, now_ms: int | None = None) -> list[dict[str, Any]]: ...

    def get_funding_history(self, symbol: str, *, limit: int = 1000) -> list[tuple[int, float]]: ...

    def get_funding_intervals(self) -> dict[str, int]: ...     # symbol -> hours between settlements (plan R4)


@runtime_checkable
class ExchangeAccount(Protocol):
    """Read-only account snapshot: equity, positions, open orders, trading rules, mark prices."""

    def get_account_equity(self) -> float: ...

    def get_positions(self, symbol: str = "", *, strict: bool = True) -> tuple[PositionView, ...]: ...

    def get_open_orders(self, symbol: str = "", *, strict: bool = True) -> tuple[OpenOrderView, ...]: ...

    def filters(self, symbol: str) -> LotFilters: ...

    def execution_blockers(self) -> list[str]: ...

    def get_ticker(self, symbol: str) -> Ticker: ...


class FundingIncome(Protocol):
    """The account's funding income on one symbol from a time on (read-only; the strategy capital base). Raises when
    the total cannot be known whole; ``now_ms`` is the runtime's time."""

    def get_funding_income(self, symbol: str, *, since_ms: int, now_ms: int) -> Decimal: ...


@runtime_checkable
class OrderGateway(Protocol):
    """Placement and lookup of ordinary orders. Results of placement are submitted / rejected / error."""

    def send_market_order(self, symbol: str, side: str, qty: float, *, reduce_only: bool = False,
                          client_order_id: str = "") -> Placement: ...

    def send_limit_order(self, symbol: str, side: str, qty: float, price: float, *, post_only: bool = True,
                         reduce_only: bool = False, client_order_id: str = "") -> Placement: ...

    def get_order(self, symbol: str, *, order_id: str = "", client_order_id: str = "") -> OrderView | None: ...

    def cancel_order(self, symbol: str, order_id: str) -> object: ...   # an acknowledgement; never read

    def get_recent_fills(self, *, symbol: Optional[str] = None, order_id: str = "",
                         since_ms: int = 0, limit: int = 100) -> tuple[FillView, ...]: ...

    def get_ticker(self, symbol: str) -> Ticker: ...  # reference quote for a limit entry without a price


@runtime_checkable
class StopOrderGateway(Protocol):
    """Exchange-side protective stops that close the whole position."""

    def place_stop_market(self, symbol: str, side: str, trigger_price: Decimal, *,
                          client_algo_id: str) -> StopPlacement: ...

    def get_algo_order(self, *, client_algo_id: str) -> StopView | None: ...

    def cancel_algo_order(self, *, client_algo_id: str) -> object: ...   # an acknowledgement; never read


@runtime_checkable
class BarListener(Protocol):
    """Matches resting orders on each closed bar: the simulated exchange (ADR-0013). Execution mode has none."""

    def on_closed_bar(self, symbol: str, bar: Bar, funding_rate: Decimal | None) -> None: ...


class PaperVenue(BarListener, Protocol):
    """The simulated exchange as the paper runtime keeps it: a bar listener whose state is saved with the cursors
    and loaded at a restart (ADR-0013 consequences: paper state = the venue's snapshot + the order journal).
    ``revision`` changes with every change of its state; ``has_matched`` says whether a bar was matched already."""

    @property
    def revision(self) -> int: ...

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, state: dict[str, Any]) -> None: ...

    def has_matched(self, symbol: str, bar: Bar) -> bool: ...


class GateClock(Protocol):
    """The entry gates' clock when they run on bar time (``ModePolicy.limiter_clock == "bar_time"``, paper): set to
    a bar's close before the decision, moved on by one poll for the retry of deferred entries; never goes back."""

    def at(self, ms: int) -> None: ...

    def advance(self, seconds: float) -> None: ...

    def ms(self) -> int: ...


@runtime_checkable
class OrderJournal(Protocol):
    """Durable order intents, exchange fills and consumer checkpoints (one writer: the SQLite adapter)."""

    def orders(self, owner: str | None = None, *, unresolved: bool = False) -> list[dict[str, Any]]: ...

    def fills(self, owner: str, *, undelivered: bool = True) -> list[dict[str, Any]]: ...

    def checkpoint(self, owner: str) -> dict[str, Any] | None: ...

    def consume(self, owner: str, fill_id: str, apply: Callable[[], dict[str, Any]], rollback: Callable[[], None],
                *, expected_revision: str | None = None) -> bool: ...

    def intent_fills(self, intent_id: str) -> list[dict[str, Any]]: ...

    def fill(self, fill_id: str, *, owner: str | None = None) -> dict[str, Any] | None: ...

    def insert_fill(self, *, fill_id: str, intent_id: str, owner: str, payload: dict[str, Any]) -> bool: ...

    def reserve(self, *, intent_id: str, symbol: str, client_id: str, side: str, qty: str, reducing: bool,
                owner: str, parent: dict[str, Any] | None, order_type: str,
                clock: Callable[[], float], venue_close: bool = False) -> tuple[str, float]: ...

    def adopt(self, *, intent_id: str, symbol: str, client_id: str, order_id: str, side: str, owner: str,
              parent: dict[str, Any], created_at: float) -> None: ...

    def set_order_id(self, intent_id: str, order_id: str) -> None: ...

    def update_status(self, intent_id: str, status: str, *, settle: bool) -> str | None: ...

    def stop_rows(self, owner: str, symbol: str, *, live_only: bool) -> list[dict[str, Any]]: ...

    def add_stop(self, *, owner: str, client_algo_id: str, symbol: str, revision: str, attempt: int, side: str,
                 trigger_price: str, created_at: float) -> None: ...

    def set_stop_status(self, client_algo_id: str, status: str) -> None: ...


class AccountOrderView(Protocol):
    """An open order of the account, any symbol (ADR-0010 start-up check)."""

    @property
    def order_id(self) -> str: ...

    @property
    def client_order_id(self) -> str: ...

    @property
    def symbol(self) -> str: ...


class AccountStopView(Protocol):
    """An open conditional order of the account, any symbol (ADR-0010 start-up check)."""

    @property
    def client_algo_id(self) -> str: ...

    @property
    def symbol(self) -> str: ...


class AccountSnapshot(Protocol):
    """The read-only view of the whole account the start-up check needs (ADR-0010)."""

    def get_positions(self, symbol: str = "", *, strict: bool = True) -> tuple[PositionView, ...]: ...

    def get_open_orders(self, symbol: str = "", *, strict: bool = True) -> tuple[AccountOrderView, ...]: ...

    def get_open_algo_orders(self) -> tuple[AccountStopView, ...]: ...

    def get_symbol_settings(self, symbol: str) -> tuple[int, str]: ...

    def get_algo_order(self, *, client_algo_id: str) -> StopView | None: ...


@runtime_checkable
class KillSwitch(Protocol):
    """Operator kill switch. Read-only for the runtime: an unreadable switch counts as active."""

    def state(self) -> Any: ...

    def allow_order(self, *, symbol: str) -> tuple[bool, str]: ...


class KillLatch(KillSwitch, Protocol):
    """The kill switch as the strategy loss stop uses it (owner 2026-11-16): the runtime may set it, with flatten;
    it never clears it."""

    def trip(self, reason: str) -> None: ...


@runtime_checkable
class DrawdownGuard(Protocol):
    """Period drawdown latch. Unknown state blocks entries and counts as latched."""

    def observe(self, equity: float, *, now: float | None = None) -> tuple[bool, str]: ...

    def invalidate(self, reason: str) -> None: ...

    def check(self, *, now: float | None = None) -> tuple[bool, str]: ...

    def latched(self, *, now: float | None = None) -> bool: ...


@runtime_checkable
class OrderLimits(Protocol):
    """Per-order and per-day notional / count limits; a passing check also records the order."""

    def check_order(self, *, qty: Decimal, price: Optional[Decimal] = None) -> Any: ...


class OrderRequest(Protocol):
    """What a gate needs to know about an order; ``reduce_only`` is read with getattr (default False)."""

    @property
    def symbol(self) -> str: ...

    @property
    def side(self) -> str: ...

    @property
    def qty(self) -> Decimal: ...

    @property
    def price(self) -> Decimal | None: ...


@runtime_checkable
class EntryReservation(Protocol):
    """Shared portfolio quota claimed by an entry (used by the frozen Bybit runtime only)."""

    def reserve(self, event: OrderRequest, *, quantity: Decimal, notional: Decimal) -> Any: ...


@runtime_checkable
class WalletCheck(Protocol):
    """Wallet reconciliation status (used by the frozen Bybit runtime only)."""

    def check(self) -> tuple[bool, str]: ...


@runtime_checkable
class ExposureReader(Protocol):
    """Account gross exposure against the hard leverage cap."""

    def check_would_exceed(self, *, symbol: str, venue: str, side: str,
                           proposed_notional_usd: float) -> tuple[bool, str]: ...


@runtime_checkable
class EventLog(Protocol):
    """Append-only operational events (observability only, never state). The kind is always an EventKind."""

    def write(self, kind: EventKind, **fields: Any) -> None: ...


@runtime_checkable
class PaperStore(Protocol):
    """Durable state of the paper sleeves; one document for all symbols."""

    def load(self) -> dict[str, Any] | None: ...          # the saved document, or None when there is none

    def save(self, document: dict[str, Any]) -> None: ...  # atomic: a crash leaves the old or the new document

    def archive(self) -> str | None: ...                   # set the saved document aside; its new name, or None


@runtime_checkable
class StatusSink(Protocol):
    """Where the runtime publishes its status snapshot each cycle (observability only)."""

    def write(self, status: dict[str, Any]) -> None: ...


@runtime_checkable
class LimitStore(Protocol):
    """Today's order count and notional per UTC day ("YYYY-MM-DD"). A store that cannot answer raises."""

    def load(self, day: str) -> tuple[int, Decimal]: ...

    def add(self, day: str, notional: Decimal) -> None: ...   # one more order of this notional today


class EntryPauses(Protocol):
    """Entries paused per symbol after a protective stop. The runtime sets pauses; a stop's pause ends by itself at
    ``ends_at`` (tech lead 2026-11-16), any other waits for a person, who can also end a scheduled one early, outside
    the runtime. A store that cannot be read raises; it never reads as "not paused"."""

    def pause(self, symbol: str, cause: str, triggers: Iterable[str] = (), *,
              ends_at: float | None = None) -> bool: ...                  # False if a pause already lasts as long

    def reason(self, symbol: str) -> str | None: ...

    def covered(self, symbol: str) -> set[str]: ...          # stop triggers a pause has been set for


class Notifier(Protocol):
    """Operator alerts. Delivery failures must never break the trading loop."""

    channel: str                                  # where alerts go, for the status file

    def alert(self, level: str, title: str, detail: str = "") -> None: ...


__all__ = ["Clock", "Sleeper", "Lock", "PaperStore", "StatusSink",
           "MarketData", "ExchangeAccount", "FundingIncome", "OrderGateway", "StopOrderGateway", "BarListener",
           "PaperVenue", "GateClock",
           "OrderJournal", "KillSwitch", "KillLatch", "DrawdownGuard", "OrderLimits", "OrderRequest",
           "EntryReservation", "WalletCheck", "ExposureReader", "EventLog", "Notifier", "AccountOrderView",
           "AccountStopView", "AccountSnapshot"]
