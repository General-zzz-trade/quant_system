"""Ports: the only way the application reaches the exchange, storage, files, clock and operators.

Method names and signatures follow the code that already implements them (see docs/arch/target.md
section 3), so the existing Binance adapter and the test venues conform without changes. A port method
that cannot answer raises; it never returns an empty result that could be mistaken for "nothing there".
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable, Optional, Protocol, runtime_checkable

Clock = Callable[[], float]          # wall-clock seconds since the epoch
Sleeper = Callable[[float], None]


@runtime_checkable
class MarketData(Protocol):
    """Public market data. Only closed bars and settled funding are returned."""

    def get_klines(self, symbol: str, interval: str = "60", limit: int = 200,
                   *, now_ms: int | None = None) -> list[dict[str, Any]]: ...

    def get_funding_history(self, symbol: str, *, limit: int = 1000) -> list[tuple[int, float]]: ...


@runtime_checkable
class ExchangeAccount(Protocol):
    """Read-only account snapshot: equity, positions, open orders, trading rules, mark prices."""

    def get_account_equity(self) -> float: ...

    def get_positions(self, symbol: str = "", *, strict: bool = True) -> tuple[Any, ...]: ...

    def get_open_orders(self, symbol: str = "", *, strict: bool = True) -> tuple[Any, ...]: ...

    def filters(self, symbol: str) -> Any: ...

    def execution_blockers(self) -> list[str]: ...

    def get_ticker(self, symbol: str) -> dict[str, Any]: ...


@runtime_checkable
class OrderGateway(Protocol):
    """Placement and lookup of ordinary orders. Results of placement are submitted / rejected / error."""

    def send_market_order(self, symbol: str, side: str, qty: float, *, reduce_only: bool = False,
                          client_order_id: str = "") -> dict[str, Any]: ...

    def send_limit_order(self, symbol: str, side: str, qty: float, price: float, *, post_only: bool = True,
                         reduce_only: bool = False, client_order_id: str = "") -> dict[str, Any]: ...

    def get_order(self, symbol: str, *, order_id: str = "", client_order_id: str = "") -> Any: ...

    def cancel_order(self, symbol: str, order_id: str) -> Any: ...

    def get_recent_fills(self, *, symbol: Optional[str] = None, order_id: str = "",
                         since_ms: int = 0, limit: int = 100) -> tuple[Any, ...]: ...

    def get_ticker(self, symbol: str) -> dict[str, Any]: ...  # reference quote for a limit entry without a price


@runtime_checkable
class StopOrderGateway(Protocol):
    """Exchange-side protective stops (STOP_MARKET closePosition)."""

    def place_stop_market(self, symbol: str, side: str, trigger_price: Decimal, *,
                          client_algo_id: str) -> dict[str, Any]: ...

    def get_algo_order(self, *, client_algo_id: str) -> Any: ...

    def cancel_algo_order(self, *, client_algo_id: str) -> dict[str, Any]: ...


@runtime_checkable
class OrderJournal(Protocol):
    """Durable order intents, exchange fills and consumer checkpoints (one writer: the SQLite adapter)."""

    def orders(self, owner: str | None = None, *, unresolved: bool = False) -> list[dict[str, Any]]: ...

    def fills(self, owner: str, *, undelivered: bool = True) -> list[dict[str, Any]]: ...

    def checkpoint(self, owner: str) -> dict[str, Any] | None: ...

    def consume(self, owner: str, fill_id: str, apply: Callable[[], dict[str, Any]], rollback: Callable[[], None],
                *, expected_revision: str | None = None) -> bool: ...


@runtime_checkable
class KillSwitch(Protocol):
    """Operator kill switch. Read-only for the runtime: an unreadable switch counts as active."""

    def state(self) -> Any: ...

    def allow_order(self, *, symbol: str) -> tuple[bool, str]: ...


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
    """Append-only operational events (observability only, never state)."""

    def write(self, kind: str, **fields: Any) -> None: ...


@runtime_checkable
class Notifier(Protocol):
    """Operator alerts. Delivery failures must never break the trading loop."""

    def alert(self, level: str, title: str, detail: str = "") -> None: ...


__all__ = ["Clock", "Sleeper", "MarketData", "ExchangeAccount", "OrderGateway", "StopOrderGateway",
           "OrderJournal", "KillSwitch", "DrawdownGuard", "OrderLimits", "OrderRequest", "EntryReservation",
           "WalletCheck", "ExposureReader", "EventLog", "Notifier"]
