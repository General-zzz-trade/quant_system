# execution/adapters/bybit/mapper.py
"""Map Bybit V5 API responses to canonical execution models."""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from execution.models.balances import BalanceSnapshot, CanonicalBalance
from execution.models.fills import CanonicalFill
from execution.models.instruments import InstrumentInfo
from execution.models.orders import CanonicalOrder
from execution.models.positions import VenuePosition
from execution.models.digest import fill_key, fill_digest

_VENUE = "bybit"


def _d(v: Any) -> Decimal:
    if v is None or v == "":
        return Decimal("0")
    return Decimal(str(v))


# -- Order status mapping --
_ORDER_STATUS = {
    "New": "new",
    "PartiallyFilled": "partially_filled",
    "Filled": "filled",
    "Cancelled": "canceled",
    "PartiallyFilledCanceled": "canceled",
    "Rejected": "rejected",
    "Deactivated": "expired",
    "Untriggered": "new",
    "Triggered": "new",
}


def map_instrument(info: dict) -> InstrumentInfo:
    """Convert Bybit instrument info to InstrumentInfo."""
    lot_filter = info.get("lotSizeFilter", {})
    price_filter = info.get("priceFilter", {})
    return InstrumentInfo(
        venue=_VENUE,
        symbol=info.get("symbol", ""),
        base_asset=info.get("baseCoin", ""),
        quote_asset=info.get("quoteCoin", ""),
        price_precision=len(price_filter.get("tickSize", "0.01").rstrip("0").split(".")[-1]),
        qty_precision=len(lot_filter.get("qtyStep", "0.001").rstrip("0").split(".")[-1]),
        tick_size=_d(price_filter.get("tickSize", "0.01")),
        lot_size=_d(lot_filter.get("qtyStep", "0.001")),
        min_qty=_d(lot_filter.get("minOrderQty", "0")),
        max_qty=_d(lot_filter.get("maxOrderQty")) if lot_filter.get("maxOrderQty") else None,
        min_notional=_d(lot_filter.get("minNotionalValue", "0")),
        contract_type="perpetual" if info.get("contractType") == "LinearPerpetual" else None,
        margin_asset=info.get("settleCoin", "USDT"),
        trading_enabled=info.get("status") == "Trading",
    )


def map_position(pos: dict) -> VenuePosition:
    """Convert Bybit position to VenuePosition."""
    side = pos.get("side", "")
    qty = _d(pos.get("size", "0"))
    if side == "Sell":
        qty = -qty
    return VenuePosition(
        venue=_VENUE,
        symbol=pos.get("symbol", ""),
        qty=qty,
        entry_price=_d(pos.get("avgPrice")),
        mark_price=_d(pos.get("markPrice")),
        liquidation_price=_d(pos.get("liqPrice")) if pos.get("liqPrice") else None,
        unrealized_pnl=_d(pos.get("unrealisedPnl")),
        leverage=int(pos["leverage"]) if pos.get("leverage") else None,
        margin_type="cross" if pos.get("tradeMode") == "0" else "isolated",
        ts_ms=int(pos.get("updatedTime", 0)),
        raw=pos,
    )


def map_order(order: dict) -> CanonicalOrder:
    """Convert Bybit order to CanonicalOrder."""
    side = "buy" if order.get("side") == "Buy" else "sell"
    order_type = order.get("orderType", "").lower()
    status = _ORDER_STATUS.get(order.get("orderStatus", ""), "new")
    return CanonicalOrder(
        venue=_VENUE,
        symbol=order.get("symbol", ""),
        order_id=order.get("orderId", ""),
        client_order_id=order.get("orderLinkId") or None,
        status=status,
        side=side,
        order_type=order_type if order_type in ("market", "limit") else "limit",
        tif=(order.get("timeInForce", "GTC") or "GTC").lower(),
        qty=_d(order.get("qty")),
        price=_d(order.get("price")) if order.get("price") and order["price"] != "0" else None,
        filled_qty=_d(order.get("cumExecQty")),
        avg_price=_d(order.get("avgPrice")) if order.get("avgPrice") and order["avgPrice"] != "0" else None,
        ts_ms=int(order.get("createdTime", 0)),
        raw=order,
    )


def map_fill(fill: dict) -> CanonicalFill:
    """Convert Bybit execution/trade to CanonicalFill."""
    side = "buy" if fill.get("side") == "Buy" else "sell"
    trade_id = fill.get("execId", "")
    qty = _d(fill.get("execQty"))
    price = _d(fill.get("execPrice"))
    fee = _d(fill.get("execFee"))
    symbol = fill.get("symbol", "")
    order_id = fill.get("orderId", "")

    return CanonicalFill(
        venue=_VENUE,
        symbol=symbol,
        order_id=order_id,
        trade_id=trade_id,
        fill_id=fill_key(venue=_VENUE, symbol=symbol, trade_id=trade_id),
        side=side,
        qty=qty,
        price=price,
        fee=abs(fee),
        fee_asset=fill.get("feeCurrency"),
        liquidity="maker" if fill.get("isMaker") == "true" else "taker",
        ts_ms=int(fill.get("execTime", 0)),
        payload_digest=fill_digest(
            symbol=symbol, order_id=order_id, trade_id=trade_id,
            side=side, qty=qty, price=price, fee=abs(fee),
            ts_ms=int(fill.get("execTime", 0)),
        ),
        raw=fill,
    )


def map_balance(data: dict) -> BalanceSnapshot:
    """Convert Bybit wallet balance to BalanceSnapshot."""
    now_ms = int(time.time() * 1000)
    balances = []
    for account in data.get("list", []):
        for coin in account.get("coin", []):
            wallet_bal = float(coin.get("walletBalance", 0))
            if wallet_bal == 0:
                continue
            locked = float(coin.get("locked", 0))
            balances.append(CanonicalBalance.from_free_locked(
                venue=_VENUE,
                asset=coin.get("coin", ""),
                free=_d(wallet_bal - locked),
                locked=_d(locked),
                ts_ms=now_ms,
                raw=coin,
            ))
    return BalanceSnapshot(venue=_VENUE, balances=tuple(balances), ts_ms=now_ms)
