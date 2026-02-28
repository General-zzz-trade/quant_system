from __future__ import annotations

import uuid
from decimal import Decimal
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from event.header import EventHeader
from event.types import EventType


def _sign(side: str) -> int:
    s = str(side).strip().lower()
    if s in ("buy", "long"):
        return 1
    if s in ("sell", "short"):
        return -1
    raise ValueError(f"unsupported side: {side!r}")


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class BacktestExecutionAdapter:
    """Minimal paper execution adapter.

    Takes OrderEvent, produces fill events immediately at a chosen price.
    Maintains its own lightweight position state to compute realized PnL.
    """

    def __init__(
        self,
        *,
        price_source: Callable[[str], Optional[Decimal]],
        ts_source: Callable[[], Optional[datetime]],
        fee_bps: Decimal = Decimal("0"),
        slippage_bps: Decimal = Decimal("0"),
        source: str = "paper",
        on_fill: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._price_source = price_source
        self._ts_source = ts_source
        self._fee_bps = Decimal(str(fee_bps))
        self._slippage_bps = Decimal(str(slippage_bps))
        self._source = source
        self._on_fill = on_fill

        self._pos_qty: Dict[str, Decimal] = {}
        self._avg_px: Dict[str, Optional[Decimal]] = {}

    def send_order(self, order_event: Any) -> List[Any]:
        sym = str(getattr(order_event, "symbol")).upper()
        side = str(getattr(order_event, "side"))
        qty = Decimal(str(getattr(order_event, "qty")))
        if qty <= 0:
            return []

        px: Optional[Decimal]
        raw_price = getattr(order_event, "price", None)
        if raw_price is not None:
            px = Decimal(str(raw_price))
        else:
            px = self._price_source(sym)

        if px is None:
            raise RuntimeError(f"no price available for {sym}")

        # Apply slippage: buy fills at a higher price, sell fills at a lower price
        if self._slippage_bps > 0:
            slippage_mult = self._slippage_bps / Decimal("10000")
            if _sign(side) > 0:  # buy
                px = px * (Decimal("1") + slippage_mult)
            else:  # sell
                px = px * (Decimal("1") - slippage_mult)

        signed = qty * Decimal(_sign(side))
        prev_qty = self._pos_qty.get(sym, Decimal("0"))
        prev_avg = self._avg_px.get(sym, None)

        fee = (px * qty) * (self._fee_bps / Decimal("10000"))
        realized = Decimal("0")

        if prev_qty != 0 and prev_avg is not None and (prev_qty > 0) != (signed > 0):
            closed = min(abs(prev_qty), abs(signed))
            sign_prev = Decimal("1") if prev_qty > 0 else Decimal("-1")
            realized = (px - prev_avg) * closed * sign_prev

        new_qty = prev_qty + signed
        new_avg: Optional[Decimal]

        if new_qty == 0:
            new_avg = None
        else:
            if prev_qty == 0 or (prev_qty > 0) == (signed > 0):
                base_qty = abs(prev_qty)
                add_qty = abs(signed)
                base_avg = prev_avg if prev_avg is not None else px
                new_avg = (base_avg * base_qty + px * add_qty) / (base_qty + add_qty)
            else:
                if (prev_qty > 0 and new_qty > 0) or (prev_qty < 0 and new_qty < 0):
                    new_avg = prev_avg
                else:
                    new_avg = px

        self._pos_qty[sym] = new_qty
        self._avg_px[sym] = new_avg

        parent = getattr(order_event, "header", None)
        if isinstance(parent, EventHeader):
            h = EventHeader.from_parent(parent=parent, event_type=EventType.FILL, version=1, source=self._source)
        else:
            h = EventHeader.new_root(event_type=EventType.FILL, version=1, source=self._source)

        fill = SimpleNamespace(
            header=h,
            event_type=EventType.FILL,
            ts=self._ts_source(),
            symbol=sym,
            side=side,
            qty=qty,
            price=px,
            fee=fee,
            realized_pnl=realized,
            cash_delta=0.0,
            margin_change=0.0,
        )

        if self._on_fill is not None:
            self._on_fill(fill)

        return [fill]
