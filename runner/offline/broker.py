"""Conservative bar simulator and USDT-linear perpetual accounting.

No exchange client is imported. OHLC cannot determine queue position: maker
orders require strict price penetration, have a volume cap and expire.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from types import SimpleNamespace

from data.quality.causal_bars import Bar
from event.header import EventHeader
from event.types import EventType, FillEvent
from shared.strategy_position import StrategyPosition

D = Decimal


@dataclass(frozen=True)
class ReplayConfig:
    starting_balance: str = "10000"
    maker_bps: str = "2"
    taker_bps: str = "5"
    slippage_bps: str = "3"
    limit_offset_bps: str = "2"
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

    def __post_init__(self):
        for key in ("starting_balance", "maker_bps", "taker_bps", "slippage_bps", "limit_offset_bps",
                    "participation", "max_exposure", "qty_step", "min_notional"):
            value = D(str(getattr(self, key)))
            if not value.is_finite() or value < 0:
                raise ValueError(f"invalid {key}")
        if any(D(getattr(self, k)) <= 0 for k in ("starting_balance", "qty_step", "min_notional")):
            raise ValueError("balance, step and notional must be positive")
        if not 0 < D(self.participation) <= 1 or not 0 < D(self.max_exposure) <= 1:
            raise ValueError("participation/exposure must be in (0, 1]; leveraged liquidation is unsupported")
        if any(D(getattr(self, k)) >= 10000 for k in ("maker_bps", "taker_bps", "slippage_bps", "limit_offset_bps")):
            raise ValueError("cost/offset bps must be below 10000")
        if self.entry_mode not in ("maker", "market"):
            raise ValueError("entry_mode must be maker or market")
        if self.ttl_bars < 1 or not 2 <= self.zscore_warmup <= self.zscore_window:
            raise ValueError("invalid expiry or zscore window")
        if not 0 < self.deadzone < 10 or not 1 <= self.min_hold <= self.max_hold:
            raise ValueError("invalid signal constraints")

    def record(self):
        return asdict(self)


class BarBroker:
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.position = StrategyPosition()
        self.fees = D(0)
        self.funding = D(0)
        self.mark = D(0)
        self.now = 0
        self.orders: list[dict] = []
        self.fills: list[dict] = []
        self.funding_log: list[dict] = []
        self.pending: list[dict] = []
        self.emit = lambda event: None
        self.status = lambda oid, status: None

    @property
    def cash(self):
        return D(self.config.starting_balance) + self.position.realized_pnl - self.fees + self.funding

    @property
    def equity(self):
        return self.cash + self.position.qty * (self.mark - self.position.entry_price)

    def round_qty(self, qty):
        step = D(self.config.qty_step)
        return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step

    def register(self, order):
        existing = next((r for r in self.orders if r["internal_id"] == order.order_id), None)
        if existing:
            return existing
        reduce = order.qty == 0 or getattr(order, "reduce_only", False)
        row = {"id": len(self.orders) + 1, "internal_id": order.order_id, "submitted_at": self.now,
               "side": order.side.lower(), "qty": str(order.qty), "reference": str(order.price),
               "reduce_only": reduce, "reason": getattr(order, "reason", None), "status": "created"}
        self.orders.append(row)
        return row

    def status_update(self, oid, status):
        row = next((r for r in self.orders if r["internal_id"] == oid), None)
        if row is not None:
            row["status"] = status
            row["updated_at"] = self.now
        self.status(oid, status)

    def send_order(self, order):
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

    def process_bar(self, bar: Bar):
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
                proposed = self.position.qty + sign * qty
                worst_mark = bar.low if sign > 0 else bar.high
                worst_equity = self.equity + self.position.qty * (worst_mark - bar.open)
                worst_equity += sign * qty * (worst_mark - price) - qty * price * fee_bps / 10000
                if qty > 0 and abs(proposed) * bar.high > max(D(0), worst_equity) * D(self.config.max_exposure):
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
                if qty > 0:
                    # Adverse bar extremes also bound exposure (OHLC ordering is unknown).
                    worst_mark = bar.low if sign > 0 else bar.high
                    proposed = self.position.qty + sign * qty
                    worst_equity = self.equity + self.position.qty * (worst_mark - bar.open)
                    worst_equity += sign * qty * (worst_mark - price) - qty * price * fee_bps / 10000
                    if abs(proposed) * bar.high > max(D(0), worst_equity) * D(self.config.max_exposure):
                        row["rejection"] = "fill exceeds exposure cap"
                        self._finish(row, "rejected")
                        continue
                self.now = bar.available_at
            if qty > 0:
                fee = qty * price * fee_bps / 10000
                fact = SimpleNamespace(side=row["side"], qty=qty, price=price, fee=fee, fee_asset="USDT")
                self.position.apply(fact)
                self.fees += fee
                fill_id = f"offline-fill-{len(self.fills) + 1}"
                self.fills.append({"id": fill_id, "order": row["id"], "at": self.now, "side": row["side"],
                                   "qty": str(qty), "price": str(price), "fee": str(fee)})
                event = FillEvent(header=EventHeader(fill_id, EventType.FILL, 1, self.now * 1_000_000, "offline"),
                                  fill_id=fill_id, order_id=row["internal_id"], symbol=row["symbol"],
                                  qty=qty, price=price, side=row["side"])
                self.emit(event)
                remaining -= qty
                row["remaining"] = str(remaining)
                if not row["reduce_only"]:
                    budget -= qty
            if remaining == 0:
                self._finish(row, "filled")
            elif row["age"] >= self.config.ttl_bars:
                self._finish(row, "canceled")
        self.now, self.mark = bar.available_at, bar.close

    def _finish(self, row, status):
        self.status_update(row["internal_id"], status)
        self.pending.remove(row)

    def settle_funding(self, at: int, rate: D, mark: D):
        if not rate.is_finite() or not mark.is_finite() or mark <= 0:
            raise ValueError("invalid funding event")
        payment = -self.position.qty * mark * rate  # positive rate: longs pay, shorts receive
        self.funding += payment
        self.funding_log.append({"at": at, "rate": str(rate), "mark": str(mark), "cashflow": str(payment)})

    def snapshot(self):
        return {"at": self.now, "equity": str(self.equity), "cash": str(self.cash),
                "qty": str(self.position.qty), "mark": str(self.mark),
                "realized_pnl": str(self.position.realized_pnl), "fees": str(self.fees),
                "funding": str(self.funding),
                "unrealized_pnl": str(self.position.qty * (self.mark - self.position.entry_price))}

    def order_records(self):
        return [{k: v for k, v in row.items() if k != "internal_id"} for row in self.orders]
