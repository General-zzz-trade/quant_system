"""Explicit exchange fills for strategy tests; deciding alone never fills an order."""
from types import SimpleNamespace
from event.types import OrderEvent


def confirm_orders(module, events):
    for order in events:
        if not isinstance(order, OrderEvent):
            continue
        qty = order.qty if order.qty else module._current_qty
        module.on_fill(SimpleNamespace(order_id=order.order_id, fill_id=f"test:{order.order_id}",
                                      qty=qty, price=order.price, side=order.side))
        module.on_order_status(order.order_id, "filled")
