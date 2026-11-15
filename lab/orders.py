"""Orders of the research paper broker: the same ``OrderIntent`` the execution path sends (ADR-0006).

Until 2026-11-15 this built a Rust-backed ``OrderEvent`` with an event header nobody read; the broker, the
paper session and the risk gates read only the fields below, which ``OrderIntent`` has under the same names.
"""
from __future__ import annotations

from decimal import Decimal

from trading.domain.orders import OrderIntent


def make_order(*, order_id: str, source: str, symbol: str, side: str, qty: Decimal, price: Decimal,
               reduce_only: bool, reason: str) -> OrderIntent:
    return OrderIntent(order_id=order_id, source=source, symbol=symbol, side=side, qty=qty, price=price,
                       reduce_only=reduce_only, reason=reason)
