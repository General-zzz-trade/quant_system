"""Conservative bar simulator and USDT-linear perpetual accounting.

No exchange client is imported. OHLC cannot determine queue position: maker
orders require strict price penetration, have a volume cap and expire.

Optional isolated-margin mode (``isolated_leverage`` > 0, off by default):
gross exposure up to ``MAX_TOTAL_LEVERAGE`` x equity, a protective stop per
position (``set_stop``), and liquidation at the isolated-margin price. Within a
bar the order of events is unknown, so the model is deliberately adverse:
- an open beyond the liquidation price liquidates (loss = posted margin);
- a stop crossed intrabar fills at the stop, or at the open when the open has
  already gapped through it, minus ``stop_slippage_bps``;
- a stop fill that would land beyond the liquidation price liquidates instead;
- without a stop, a bar that touches the liquidation price liquidates.
Liquidation realizes exactly the posted isolated margin (the exchange's
clearance fee is taken out of that margin). With every option off the broker
behaves, and records its configuration, exactly as before.

Optional cross-margin stop (``stop_pct`` > 0, off by default; tech lead 2026-11-16): the live runtime's exchange
stop, a STOP_MARKET closePosition at the position's entry price x (1 - stop_pct) for a long, x (1 + stop_pct) for a
short (trading/application/execution/stops.py). Market entries fill at the open, so the stop guards the rest of
that bar and every later one, at the current average entry; it fills by the isolated-mode stop rule above (and
paper's BarStopModel). Market entries only: a maker fill lands at the close, after the bar's range.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, cast

from trading.domain.bars import Bar
from trading.domain.orders import LIMIT_OFFSET_BPS
from trading.domain.risk.exposure import MAX_TOTAL_LEVERAGE
from trading.domain.position import StrategyPosition

if TYPE_CHECKING:
    from trading.domain.strategy.overlay import StrategySpec

D = Decimal
Row = dict[str, Any]     # an order, fill or log row: the broker keeps its state as plain dicts


class BrokerOrder(Protocol):
    """What the broker reads of an order; it also reads ``reduce_only`` and ``reason`` when present."""

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


@dataclass(frozen=True)
class ReplayConfig:
    starting_balance: str = "10000"
    maker_bps: str = "2"
    taker_bps: str = "5"
    slippage_bps: str = "3"
    limit_offset_bps: str = str(LIMIT_OFFSET_BPS)
    participation: str = "0.01"
    max_exposure: str = "1"
    qty_step: str = "0.001"
    min_notional: str = "5"
    ttl_bars: int = 3
    daily_loss_pct: float = 5.0
    weekly_loss_pct: float = 10.0
    entry_mode: str = "maker"   # maker: post-only limit with TTL; market: next open + slippage + taker fee
    deadzone: float = 1.0
    zscore_window: int = 180
    zscore_warmup: int = 45
    min_hold: int = 6
    max_hold: int = 36
    # Isolated-margin mode; "0" keeps the original unleveraged accounting (max_exposure <= 1).
    isolated_leverage: str = "0"
    maintenance_margin_rate: str = "0.005"
    stop_slippage_bps: str = "10"
    # Cross-margin protective stop as a fraction of the entry price; "0" keeps the broker without one.
    stop_pct: str = "0"

    def __post_init__(self) -> None:
        for key in ("starting_balance", "maker_bps", "taker_bps", "slippage_bps", "limit_offset_bps",
                    "participation", "max_exposure", "qty_step", "min_notional"):
            value = D(str(getattr(self, key)))
            if not value.is_finite() or value < 0:
                raise ValueError(f"invalid {key}")
        if any(D(getattr(self, k)) <= 0 for k in ("starting_balance", "qty_step", "min_notional")):
            raise ValueError("balance, step and notional must be positive")
        leverage, mmr = D(str(self.isolated_leverage)), D(str(self.maintenance_margin_rate))
        if not leverage.is_finite() or leverage < 0 or not mmr.is_finite() or mmr < 0:
            raise ValueError("invalid isolated leverage or maintenance margin rate")
        if not 0 < D(self.participation) <= 1:
            raise ValueError("participation must be in (0, 1]")
        if leverage == 0:
            if not 0 < D(self.max_exposure) <= 1:
                raise ValueError("exposure must be in (0, 1] unless isolated_leverage models liquidation")
        else:
            if not 1 <= leverage <= 20:
                raise ValueError("isolated_leverage must be in [1, 20]")
            if not 0 < D(self.max_exposure) <= min(MAX_TOTAL_LEVERAGE, leverage):
                raise ValueError(f"max_exposure must be in (0, min({MAX_TOTAL_LEVERAGE}, isolated_leverage)]")
            if not 0 < mmr < 1 / leverage:
                raise ValueError("maintenance_margin_rate must be in (0, 1/isolated_leverage)")
        if not D(str(self.stop_slippage_bps)).is_finite() or not 0 <= D(str(self.stop_slippage_bps)) < 10000:
            raise ValueError("stop_slippage_bps must be in [0, 10000)")
        stop_pct = D(str(self.stop_pct))
        if not stop_pct.is_finite() or not 0 <= stop_pct < 1:
            raise ValueError("stop_pct must be in [0, 1)")
        if stop_pct > 0 and leverage > 0:
            raise ValueError("stop_pct is the cross-margin stop; isolated mode places its own with set_stop")
        if stop_pct > 0 and self.entry_mode != "market":
            raise ValueError("stop_pct needs market entries (a maker fill lands after the bar's range)")
        if any(D(getattr(self, k)) >= 10000 for k in ("maker_bps", "taker_bps", "slippage_bps", "limit_offset_bps")):
            raise ValueError("cost/offset bps must be below 10000")
        if self.entry_mode not in ("maker", "market"):
            raise ValueError("entry_mode must be maker or market")
        if self.ttl_bars < 1 or not 2 <= self.zscore_warmup <= self.zscore_window:
            raise ValueError("invalid expiry or zscore window")
        if not 0 < self.deadzone < 10 or not 1 <= self.min_hold <= self.max_hold:
            raise ValueError("invalid signal constraints")

    def record(self) -> dict[str, Any]:
        """Without isolated mode or a stop the new fields are omitted, so earlier records stay identical."""
        out = asdict(self)
        stop = D(str(self.stop_pct)) > 0
        if D(str(self.isolated_leverage)) == 0:
            for key in ("isolated_leverage", "maintenance_margin_rate") + (() if stop else ("stop_slippage_bps",)):
                out.pop(key)
        if not stop:
            out.pop("stop_pct")
        return out


@dataclass(frozen=True)
class BrokerFill:
    """One simulated fill, passed to ``BarBroker.emit`` (a hook; nothing in the repository sets it). It replaced a
    Rust-backed FillEvent whose header nobody read (ADR-0006, 2026-11-15)."""

    fill_id: str
    order_id: str
    symbol: str
    side: str
    qty: Decimal
    price: Decimal
    at: int


class BarBroker:
    def __init__(self, config: ReplayConfig) -> None:
        self.config = config
        self.position = StrategyPosition()
        self.fees = D(0)
        self.funding = D(0)
        self.mark = D(0)
        self.now = 0
        self.orders: list[Row] = []
        self.fills: list[Row] = []
        self.funding_log: list[Row] = []
        self.pending: list[Row] = []
        self.emit: Callable[[BrokerFill], None] = lambda event: None
        self.status: Callable[[str, str], None] = lambda oid, status: None
        self.leverage = D(str(config.isolated_leverage))
        self.margin = D(0)           # isolated margin posted for the open position
        self.stop_price: Decimal | None = None   # protective stop for the open position (isolated mode only)
        self.liquidations: list[Row] = []

    @property
    def cash(self) -> Decimal:
        return D(self.config.starting_balance) + self.position.realized_pnl - self.fees + self.funding

    @property
    def equity(self) -> Decimal:
        return self.cash + self.position.qty * (self.mark - self.position.entry_price)

    def round_qty(self, qty: Decimal) -> Decimal:
        step = D(self.config.qty_step)
        return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step

    def register(self, order: BrokerOrder) -> Row:
        existing = next((r for r in self.orders if r["internal_id"] == order.order_id), None)
        if existing:
            return existing
        reduce = order.qty == 0 or getattr(order, "reduce_only", False)
        row = {"id": len(self.orders) + 1, "internal_id": order.order_id, "submitted_at": self.now,
               "side": order.side.lower(), "qty": str(order.qty), "reference": str(order.price),
               "reduce_only": reduce, "reason": getattr(order, "reason", None), "status": "created"}
        self.orders.append(row)
        return row

    def status_update(self, oid: str, status: str) -> None:
        row = next((r for r in self.orders if r["internal_id"] == oid), None)
        if row is not None:
            row["status"] = status
            row["updated_at"] = self.now
        self.status(oid, status)

    def send_order(self, order: BrokerOrder) -> list[BrokerFill]:
        row = self.register(order)
        if row["status"] != "created":
            return []  # a retry never creates a second order
        if row["reduce_only"]:
            # Do not leave an old maker remainder able to re-open after a stop.
            for pending in list(self.pending):
                if not pending["reduce_only"]:
                    pending["rejection"] = "entry remainder canceled before reduction"
                    self._finish(pending, "canceled")
        requested = D(str(order.qty))
        qty = (abs(self.position.qty) if requested == 0 else min(requested, abs(self.position.qty))) \
            if row["reduce_only"] else requested
        qty = self.round_qty(qty)
        if qty <= 0 or (not row["reduce_only"] and qty * D(str(order.price)) < D(self.config.min_notional)):
            row["rejection"] = "below trading minimum"
            self.status_update(order.order_id, "rejected")
            return []
        sign = D(1) if row["side"] == "buy" else D(-1)
        limit = D(str(order.price)) * (1 - sign * D(self.config.limit_offset_bps) / 10000)
        row.update(remaining=str(qty), limit=str(limit), age=0, status="pending", symbol=order.symbol)
        self.pending.append(row)
        return []

    def cancel(self, order_id: str) -> bool:
        """Take a resting order off the book (the simulated exchange's cancel). False when it is not resting:
        already filled, canceled or rejected, or never sent. A retry of a canceled order does not revive it."""
        row = next((r for r in self.pending if r["internal_id"] == order_id), None)
        if row is None:
            return False
        row["rejection"] = "canceled by request"
        self._finish(row, "canceled")
        return True

    def process_bar(self, bar: Bar) -> None:
        self.mark = bar.open
        budget = self.round_qty(bar.volume * D(self.config.participation))
        # Market reductions occur at open, maker fills at close. Preserve time order.
        for row in sorted(self.pending, key=lambda r: not r["reduce_only"]):
            if row["submitted_at"] > bar.open_time:
                continue
            self.now = bar.open_time
            remaining = D(row["remaining"])
            sign = D(1) if row["side"] == "buy" else D(-1)
            if row["reduce_only"]:
                if self.position.qty * sign >= 0:
                    self._finish(row, "canceled")
                    continue
                qty = min(remaining, abs(self.position.qty))
                price = bar.open * (1 + sign * D(self.config.slippage_bps) / 10000)
                fee_bps = D(self.config.taker_bps)
            elif self.config.entry_mode == "market":
                qty = min(remaining, budget)
                price = bar.open * (1 + sign * D(self.config.slippage_bps) / 10000)
                fee_bps = D(self.config.taker_bps)
                if qty > 0 and self._exceeds_cap(row, bar, qty, sign, price, fee_bps):
                    row["rejection"] = "fill exceeds exposure cap"
                    self._finish(row, "rejected")
                    continue
                row["age"] += 1
            else:
                limit = D(row["limit"])
                if row["age"] == 0 and ((sign > 0 and limit >= bar.open) or (sign < 0 and limit <= bar.open)):
                    row["rejection"] = "post-only would cross next open"
                    self._finish(row, "rejected")
                    continue
                touched = bar.low < limit if sign > 0 else bar.high > limit
                row["age"] += 1
                qty = min(remaining, budget) if touched else D(0)
                price, fee_bps = limit, D(self.config.maker_bps)
                if qty > 0 and self._exceeds_cap(row, bar, qty, sign, price, fee_bps):
                    row["rejection"] = "fill exceeds exposure cap"
                    self._finish(row, "rejected")
                    continue
                self.now = bar.available_at
            if qty > 0:
                fee = qty * price * fee_bps / 10000
                fact = SimpleNamespace(side=row["side"], qty=qty, price=price, fee=fee, fee_asset="USDT")
                before = self.position.qty
                self.position.apply(fact)
                self._update_margin(before, price)
                self.fees += fee
                fill_id = f"offline-fill-{len(self.fills) + 1}"
                self.fills.append({"id": fill_id, "order": row["id"], "at": self.now, "side": row["side"],
                                   "qty": str(qty), "price": str(price), "fee": str(fee)})
                self.emit(BrokerFill(fill_id=fill_id, order_id=row["internal_id"], symbol=row["symbol"],
                                     qty=qty, price=price, side=row["side"], at=self.now))
                remaining -= qty
                row["remaining"] = str(remaining)
                if not row["reduce_only"]:
                    budget -= qty
            if remaining == 0:
                self._finish(row, "filled")
            elif row["age"] >= self.config.ttl_bars:
                self._finish(row, "canceled")
        if self.leverage > 0:
            self._protect(bar)
        elif D(self.config.stop_pct) > 0:
            self._cross_stop(bar)
        self.now, self.mark = bar.available_at, bar.close

    def _exceeds_cap(self, row: Row, bar: Bar, qty: Decimal, sign: Decimal, price: Decimal,
                     fee_bps: Decimal) -> bool:
        """Whether an entry fill would breach ``max_exposure``.

        Unlevered mode (the default): adverse bar extremes also bound exposure, since the
        OHLC ordering is unknown. Isolated-margin mode follows the live AccountExposureGate
        instead (notional at the order's reference price against current equity); adverse
        moves after the fill are what the protective stop and liquidation model handle.
        """
        proposed = self.position.qty + sign * qty
        if self.leverage > 0:
            return abs(proposed) * D(row["reference"]) > max(D(0), self.equity) * D(self.config.max_exposure)
        worst_mark = bar.low if sign > 0 else bar.high
        worst_equity = self.equity + self.position.qty * (worst_mark - bar.open)
        worst_equity += sign * qty * (worst_mark - price) - qty * price * fee_bps / 10000
        return abs(proposed) * bar.high > max(D(0), worst_equity) * D(self.config.max_exposure)

    # -- isolated-margin mode -----------------------------------------------------------------
    def set_stop(self, price: Decimal | float | str) -> None:
        """Protective stop for the current position (isolated mode only; cleared when flat)."""
        if self.leverage <= 0:
            raise ValueError("protective stops are modelled only in isolated-margin mode")
        price = D(str(price))
        if not price.is_finite() or price <= 0:
            raise ValueError("invalid stop price")
        self.stop_price = price

    def clear_stop(self) -> None:
        self.stop_price = None

    def liquidation_price(self) -> Decimal | None:
        """Isolated-margin liquidation price of the open position (None when flat or not isolated)."""
        qty = self.position.qty
        if self.leverage <= 0 or qty == 0:
            return None
        mmr, entry = D(self.config.maintenance_margin_rate), self.position.entry_price
        return entry * (1 - 1 / self.leverage + mmr) if qty > 0 else entry * (1 + 1 / self.leverage - mmr)

    def _update_margin(self, before: Decimal, price: Decimal) -> None:
        if self.leverage <= 0:
            return
        after = self.position.qty
        if after == 0:
            self.margin = D(0)
        elif before == 0 or (before > 0) != (after > 0):
            self.margin = abs(after) * D(str(price)) / self.leverage
        elif abs(after) > abs(before):
            self.margin += (abs(after) - abs(before)) * D(str(price)) / self.leverage
        else:
            self.margin = self.margin * abs(after) / abs(before)

    def _record_exit(self, kind: str, side: str, qty: Decimal, price: Decimal, fee: Decimal) -> None:
        fill_id = f"offline-fill-{len(self.fills) + 1}"
        self.fills.append({"id": fill_id, "order": kind, "at": self.now, "side": side,
                           "qty": str(qty), "price": str(price), "fee": str(fee)})

    def _protect(self, bar: Bar) -> None:
        qty = self.position.qty
        if qty == 0:
            self.stop_price = None
            return
        long = qty > 0
        liq = cast(Decimal, self.liquidation_price())   # isolated mode with a position: never None
        beyond = (lambda p: p <= liq) if long else (lambda p: p >= liq)
        self.now = bar.available_at
        if beyond(bar.open):
            return self._liquidate(bar)
        if self.stop_price is not None:
            stop = self.stop_price
            if (bar.low <= stop) if long else (bar.high >= stop):
                base = min(bar.open, stop) if long else max(bar.open, stop)
                slip = D(self.config.stop_slippage_bps) / 10000
                price = base * (1 - slip) if long else base * (1 + slip)
                if beyond(price):
                    return self._liquidate(bar)
                side = "sell" if long else "buy"
                fee = abs(qty) * price * D(self.config.taker_bps) / 10000
                self.position.apply(SimpleNamespace(side=side, qty=abs(qty), price=price, fee=fee, fee_asset="USDT"))
                self.fees += fee
                self.margin, self.stop_price = D(0), None
                self._record_exit("protective_stop", side, abs(qty), price, fee)
                return
        if (bar.low <= liq) if long else (bar.high >= liq):
            self._liquidate(bar)

    def _cross_stop(self, bar: Bar) -> None:
        """The live exchange stop (``stop_pct``) on the position held after this bar's open fills."""
        qty = self.position.qty
        if qty == 0:
            return
        long, pct = qty > 0, D(self.config.stop_pct)
        stop = self.position.entry_price * (1 - pct if long else 1 + pct)
        if not ((bar.low <= stop) if long else (bar.high >= stop)):
            return
        self.now = bar.available_at
        base = min(bar.open, stop) if long else max(bar.open, stop)
        slip = D(self.config.stop_slippage_bps) / 10000
        price = base * (1 - slip) if long else base * (1 + slip)
        side = "sell" if long else "buy"
        fee = abs(qty) * price * D(self.config.taker_bps) / 10000
        self.position.apply(SimpleNamespace(side=side, qty=abs(qty), price=price, fee=fee, fee_asset="USDT"))
        self.fees += fee
        self._record_exit("protective_stop", side, abs(qty), price, fee)

    def _liquidate(self, bar: Bar) -> None:
        """Close at the price that realizes exactly the posted isolated margin as the loss."""
        qty, margin = self.position.qty, self.margin
        side = "sell" if qty > 0 else "buy"
        price = self.position.entry_price - margin / qty   # qty * (price - entry) == -margin
        self.position.apply(SimpleNamespace(side=side, qty=abs(qty), price=price, fee=D(0), fee_asset="USDT"))
        self.liquidations.append({"at": self.now, "qty": str(qty), "margin_lost": str(margin)})
        self._record_exit("liquidation", side, abs(qty), price, D(0))
        self.margin, self.stop_price = D(0), None

    def _finish(self, row: Row, status: str) -> None:
        self.status_update(row["internal_id"], status)
        self.pending.remove(row)

    def settle_funding(self, at: int, rate: Decimal, mark: Decimal) -> None:
        if not rate.is_finite() or not mark.is_finite() or mark <= 0:
            raise ValueError("invalid funding event")
        payment = -self.position.qty * mark * rate  # positive rate: longs pay, shorts receive
        self.funding += payment
        self.funding_log.append({"at": at, "rate": str(rate), "mark": str(mark), "cashflow": str(payment)})

    def snapshot(self) -> dict[str, Any]:
        return {"at": self.now, "equity": str(self.equity), "cash": str(self.cash),
                "qty": str(self.position.qty), "mark": str(self.mark),
                "realized_pnl": str(self.position.realized_pnl), "fees": str(self.fees),
                "funding": str(self.funding),
                "unrealized_pnl": str(self.position.qty * (self.mark - self.position.entry_price))}


# -- research settings: lot constants and broker costs (moved from lab/simulate.py; research and paper use them) --
@dataclass(frozen=True)
class SymbolRules:
    step: D
    min_qty: D
    min_notional: D


SYMBOL_RULES = {
    "BTCUSDT": SymbolRules(D("0.001"), D("0.001"), D("100")),
    "ETHUSDT": SymbolRules(D("0.001"), D("0.001"), D("20")),
}


def broker_config(spec: StrategySpec, rules: SymbolRules, *, starting_balance: str = "5000",
                  cost_multiplier: int = 1, stop_pct: str = "0") -> ReplayConfig:
    m = D(cost_multiplier)
    return ReplayConfig(starting_balance=starting_balance, maker_bps=str(D(2) * m), taker_bps=str(D(5) * m),
                        slippage_bps=str(D(3) * m), limit_offset_bps=str(LIMIT_OFFSET_BPS), participation="0.01",
                        max_exposure="1", qty_step=str(rules.step), min_notional=str(rules.min_notional),
                        ttl_bars=spec.ttl_bars, entry_mode=spec.execution, stop_pct=stop_pct)
