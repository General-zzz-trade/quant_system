"""SimulatedExchange: the exchange ports over one research bar broker per symbol (ADR-0013 decision 1, S10).

Paper mode is to send its orders through the same gates, order lifecycle, fill consumer and stop manager as
execution (S10 part 3); this adapter is the venue they then talk to. Its fills come from the research
``BarBroker``, so paper and research fills stay identical by construction:

- an order is handed to its symbol's broker as research hands it: the reference price is the last close and the
  quantity is floored to the lot step and written with the step's exponent ("0.150", not the float's "0.15"),
  because the broker's own rounding keeps the exponent it is given and research compares fills as strings;
- it rests until ``on_closed_bar`` matches the next bar exactly as the research sleeve does: a market order fills
  at that bar's open plus slippage, a maker entry at its limit when the bar trades through it, within the
  broker's share of the bar's volume, and the broker expires a maker entry after ``ttl_bars`` bars;
- a maker entry is taken only at the research maker price (the last close -/+ ``LIMIT_OFFSET_BPS``), within
  ``LIMIT_TOLERANCE`` relative because the lifecycle sends the price as a float; any other price could not be
  simulated with research parity.

A protective stop (``STOP_MARKET closePosition``) is modelled by a ``StopModel`` (decision L): it triggers when a
closed bar's low (a sell stop) or high (a buy stop) reaches the trigger.

Like the real venue, it raises when it cannot answer. It reads no clock and no environment, holds no
credentials and sends no request.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any, Protocol

from trading.adapters.simulated.bar_broker import BarBroker, ReplayConfig, Row
from trading.domain.bars import Bar
from trading.domain.orders import OrderIntent
from trading.domain.venue import CanonicalFill, OpenOrder, Placement, Ticker, VenueOrder, VenuePosition

VENUE = "simulated"
LIMIT_TOLERANCE = Decimal("1e-9")   # relative; the lifecycle sends the limit price as a float
# The broker's order-row status, as the order lifecycle's venue-order state.
STATES = {"created": "new", "pending": "new", "filled": "filled", "canceled": "canceled", "rejected": "rejected"}


@dataclass(frozen=True, slots=True)
class SimFilters:
    """The lot rules sessions size with: the research constants (``SYMBOL_RULES``), as the broker applies them."""

    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal


class StopModel(Protocol):
    """Decision L: whether a close-all stop on ``side`` at ``trigger`` fills within ``bar``, and at what price."""

    def triggered(self, side: str, trigger: Decimal, bar: Bar) -> Decimal | None: ...


@dataclass(frozen=True)
class BarStopModel:
    """Reached by the bar's low (a sell stop) or high (a buy stop); fills at the trigger, or at the open when the
    open has already gapped through it, worse by ``slippage_bps``: the research broker's isolated-mode stop rule
    and its stop slippage."""

    slippage_bps: Decimal = Decimal(ReplayConfig.stop_slippage_bps)

    def __post_init__(self) -> None:
        if not self.slippage_bps.is_finite() or not 0 <= self.slippage_bps < 10000:
            raise ValueError("stop slippage must be in [0, 10000) bps")

    def triggered(self, side: str, trigger: Decimal, bar: Bar) -> Decimal | None:
        slip = self.slippage_bps / 10000
        if side == "sell" and bar.low <= trigger:
            return min(bar.open, trigger) * (1 - slip)
        if side == "buy" and bar.high >= trigger:
            return max(bar.open, trigger) * (1 + slip)
        return None


def _rejected(msg: str) -> Placement:
    return {"status": "rejected", "msg": msg}


class SimulatedExchange:
    """ExchangeAccount, OrderGateway, StopOrderGateway, AccountSnapshot and BarListener over the brokers."""

    venue = VENUE

    def __init__(self, brokers: Mapping[str, BarBroker], *, filters: Mapping[str, SimFilters],
                 stop_model: StopModel) -> None:
        if not brokers or set(brokers) != set(filters):
            raise ValueError("every simulated symbol needs a broker and its filters")
        for symbol, broker in brokers.items():
            if broker.leverage != 0:
                raise ValueError(f"{symbol}: an isolated-margin broker models its own stops and liquidation")
            rules, config = filters[symbol], broker.config
            if rules.step_size != Decimal(config.qty_step) or rules.min_notional != Decimal(config.min_notional):
                raise ValueError(f"{symbol}: the filters differ from the broker's lot step or minimum notional")
        self._brokers, self._filters, self.stop_model = dict(brokers), dict(filters), stop_model
        self._orders: dict[str, dict[str, Any]] = {}    # venue order id -> {symbol, client_id, type, side}
        self._by_client: dict[str, str] = {}            # client order id -> venue order id
        self._next_id = 1

    # -- BarListener ------------------------------------------------------------------------------------------
    def on_closed_bar(self, symbol: str, bar: Bar, funding_rate: Decimal | None) -> None:
        """The research sleeve's ``on_bar``: funding at the open, then the broker matches the bar."""
        broker = self._broker(symbol)
        if bar.open_time < broker.now:
            raise ValueError(f"{symbol}: the bar at {bar.open_time} is before the broker's time {broker.now}")
        broker.now, broker.mark = bar.open_time, bar.open
        if funding_rate is not None:
            broker.settle_funding(bar.open_time, funding_rate, bar.open)
        broker.process_bar(bar)

    # -- account ----------------------------------------------------------------------------------------------
    def filters(self, symbol: str) -> SimFilters:
        try:
            return self._filters[symbol]
        except KeyError:
            raise RuntimeError(f"no simulated filters for {symbol}") from None

    def execution_blockers(self) -> list[str]:
        return []

    def get_symbol_settings(self, symbol: str) -> tuple[int, str]:
        """(leverage, margin mode): the brokers keep unlevered accounts."""
        self._broker(symbol)
        return 1, "cross"

    def get_ticker(self, symbol: str) -> Ticker:
        mark = float(self._broker(symbol).mark)
        return {"symbol": symbol, "lastPrice": mark, "bid1Price": mark, "ask1Price": mark}

    def get_account_equity(self) -> float:
        return float(sum((b.equity for b in self._brokers.values()), Decimal(0)))

    def get_symbol_equity(self, symbol: str) -> Decimal:
        """The symbol's own simulated account, as research sizes it (``ModePolicy.sizing_equity``)."""
        return self._broker(symbol).equity

    def get_positions(self, symbol: str = "", *, strict: bool = True) -> tuple[VenuePosition, ...]:
        return tuple(VenuePosition(venue=VENUE, symbol=name, qty=b.position.qty, entry_price=b.position.entry_price,
                                   mark_price=b.mark)
                     for name, b in self._selected(symbol) if b.position.qty != 0)

    def get_open_orders(self, symbol: str = "", *, strict: bool = True) -> tuple[OpenOrder, ...]:
        out = []
        for name, broker in self._selected(symbol):
            for row in broker.pending:
                order_id = self._by_client[row["internal_id"]]
                record = self._orders[order_id]
                out.append(OpenOrder(order_id=order_id, client_order_id=record["client_id"], symbol=name,
                                     side=record["side"], type=record["type"]))
        return tuple(out)

    # -- orders -----------------------------------------------------------------------------------------------
    def send_market_order(self, symbol: str, side: str, qty: float, *, reduce_only: bool = False,
                          client_order_id: str = "") -> Placement:
        side = side.lower()
        if side not in ("buy", "sell") or not client_order_id:
            return _rejected("side and client_order_id are required")
        if not reduce_only and self._broker(symbol).config.entry_mode != "market":
            return _rejected("this simulated venue takes maker entries only (entry_mode maker)")
        return self._place(symbol, side, qty, reduce_only=reduce_only, client_order_id=client_order_id,
                           kind="MARKET")

    def send_limit_order(self, symbol: str, side: str, qty: float, price: float, *, post_only: bool = True,
                         reduce_only: bool = False, client_order_id: str = "") -> Placement:
        side = side.lower()
        if side not in ("buy", "sell") or not client_order_id:
            return _rejected("side and client_order_id are required")
        broker = self._broker(symbol)
        if reduce_only or not post_only or broker.config.entry_mode != "maker":
            return _rejected("this simulated venue takes post-only maker entries only (entry_mode maker)")
        sign = Decimal(1) if side == "buy" else Decimal(-1)
        maker = broker.mark * (1 - sign * Decimal(broker.config.limit_offset_bps) / 10000)   # as the broker rests it
        limit = Decimal(str(price))
        if maker <= 0 or not limit.is_finite() or abs(limit - maker) > maker * LIMIT_TOLERANCE:
            return _rejected(f"limit {price} is not the research maker price {maker}")
        return self._place(symbol, side, qty, reduce_only=False, client_order_id=client_order_id, kind="LIMIT")

    def _place(self, symbol: str, side: str, qty: float, *, reduce_only: bool, client_order_id: str,
               kind: str) -> Placement:
        broker, rules = self._broker(symbol), self.filters(symbol)
        if client_order_id in self._by_client:
            return _rejected("duplicate client order id")
        requested = Decimal(str(qty))
        if not requested.is_finite() or requested <= 0:
            return _rejected(f"invalid quantity {qty}")
        step = rules.step_size
        quantity = ((requested / step).to_integral_value(rounding=ROUND_DOWN) * step).quantize(step)
        if quantity < rules.min_qty or quantity <= 0:     # zero would read as "close all" to the broker
            return _rejected(f"quantity {qty} below minimum {rules.min_qty}")
        order_id = self._new_id(symbol)
        self._orders[order_id] = {"symbol": symbol, "client_id": client_order_id, "type": kind, "side": side}
        self._by_client[client_order_id] = order_id
        broker.send_order(OrderIntent(order_id=client_order_id, source=VENUE, symbol=symbol, side=side, qty=quantity,
                                      price=broker.mark, reduce_only=reduce_only, reason=VENUE))
        row = self._row(order_id)
        if row["status"] == "rejected":      # the broker's own minimums; the order exists, in state rejected
            return {"status": "rejected", "orderId": order_id, "clientOrderId": client_order_id,
                    "msg": str(row.get("rejection", "rejected"))}
        return {"status": "submitted", "orderId": order_id, "clientOrderId": client_order_id}

    def get_order(self, symbol: str, *, order_id: str = "", client_order_id: str = "") -> VenueOrder | None:
        if not order_id and not client_order_id:
            raise ValueError("order_id or client_order_id required")
        order_id = order_id or self._by_client.get(client_order_id, "")
        record = self._orders.get(order_id)
        if record is None or record["symbol"] != symbol:
            return None
        fills = self._fills_of(order_id)
        filled = sum((Decimal(f["qty"]) for f in fills), Decimal(0))
        cost = sum((Decimal(f["qty"]) * Decimal(f["price"]) for f in fills), Decimal(0))
        status = self._row(order_id)["status"]
        state = "partially_filled" if status == "pending" and filled > 0 else STATES[status]
        return VenueOrder(order_id=order_id, client_order_id=record["client_id"], symbol=symbol, side=record["side"],
                          state=state, filled_qty=filled, avg_price=cost / filled if filled else Decimal(0))

    def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        record = self._orders.get(order_id)
        if record is None or record["symbol"] != symbol:
            raise RuntimeError(f"unknown simulated order {order_id} for {symbol}")
        self._broker(symbol).cancel(record["client_id"])    # False when it no longer rests: nothing changes
        return {"status": "cancel_requested"}

    def get_recent_fills(self, *, symbol: str | None = None, order_id: str = "", since_ms: int = 0,
                         limit: int = 100) -> tuple[CanonicalFill, ...]:
        """Every trade of ``order_id``; without one, the symbol's trades from ``since_ms`` on, oldest first, at most
        ``limit``."""
        if not symbol:
            raise ValueError("symbol is required for trade history")
        broker = self._broker(symbol)
        if order_id:
            record = self._orders.get(order_id)
            if record is None or record["symbol"] != symbol:
                return ()
            return tuple(self._fill(order_id, f) for f in self._fills_of(order_id))
        owners = {f["id"]: oid for oid, r in self._orders.items() if r["symbol"] == symbol
                  for f in self._fills_of(oid)}
        fills = [self._fill(owners[f["id"]], f) for f in broker.fills if f["id"] in owners and f["at"] >= since_ms]
        return tuple(fills[:limit])

    # -- helpers ----------------------------------------------------------------------------------------------
    def _broker(self, symbol: str) -> BarBroker:
        try:
            return self._brokers[symbol]
        except KeyError:
            raise RuntimeError(f"{symbol} is not traded on this simulated venue") from None

    def _selected(self, symbol: str) -> list[tuple[str, BarBroker]]:
        return [(symbol, self._broker(symbol))] if symbol else list(self._brokers.items())

    def _new_id(self, symbol: str) -> str:
        order_id = f"{symbol}-{self._next_id}"
        self._next_id += 1
        return order_id

    def _row(self, order_id: str) -> Row:
        record = self._orders[order_id]
        row = next((r for r in self._brokers[record["symbol"]].orders if r["internal_id"] == record["client_id"]),
                   None)
        if row is None:
            raise RuntimeError(f"simulated order {order_id} has no broker row")
        return row

    def _fills_of(self, order_id: str) -> list[Row]:
        record = self._orders[order_id]
        row_id = self._row(order_id)["id"]
        return [f for f in self._brokers[record["symbol"]].fills if f["order"] == row_id]

    def _fill(self, order_id: str, f: Row) -> CanonicalFill:
        record = self._orders[order_id]
        return CanonicalFill(venue=VENUE, symbol=record["symbol"], order_id=order_id, trade_id=f["id"],
                             fill_id=f["id"], side=f["side"], qty=Decimal(f["qty"]), price=Decimal(f["price"]),
                             fee=Decimal(f["fee"]), fee_asset="USDT",
                             liquidity="maker" if record["type"] == "LIMIT" else "taker", ts_ms=int(f["at"]))


__all__ = ["LIMIT_TOLERANCE", "BarStopModel", "SimFilters", "SimulatedExchange", "StopModel", "VENUE"]
