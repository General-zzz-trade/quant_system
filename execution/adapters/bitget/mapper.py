"""Map Bitget V2 API responses to canonical execution models."""
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

_VENUE = "bitget"


def _d(v: Any) -> Decimal:
    if v is None or v == "":
        return Decimal("0")
    return Decimal(str(v))


# ── Instruments ──────────────────────────────────────────────────

def map_instrument(info: dict) -> InstrumentInfo:
    """Convert Bitget contract info → InstrumentInfo."""
    symbol = info.get("symbol", "")
    # Bitget symbol format: "ETHUSDT" for USDT-M
    base = info.get("baseCoin", symbol.replace("USDT", ""))
    quote = info.get("quoteCoin", "USDT")
    price_prec = int(info.get("pricePlace", "2"))
    qty_prec = int(info.get("volumePlace", "2"))
    tick_size = _d(info.get("priceEndStep", "0.01"))
    lot_size = _d(info.get("minTradeNum", "0.01"))
    min_qty = _d(info.get("minTradeNum", "0.01"))
    min_notional = _d(info.get("minTradeUSDT", "5"))

    return InstrumentInfo(
        venue=_VENUE,
        symbol=symbol,
        base_asset=base,
        quote_asset=quote,
        price_precision=price_prec,
        qty_precision=qty_prec,
        tick_size=tick_size,
        lot_size=lot_size,
        min_qty=min_qty,
        max_qty=_d(info.get("maxTradeNum", "10000")),
        min_notional=min_notional,
        contract_type="perpetual",
        margin_asset=quote,
        trading_enabled=info.get("symbolStatus", "normal") == "normal",
    )


# ── Balance ──────────────────────────────────────────────────────

def map_balance(data: dict) -> BalanceSnapshot:
    """Convert Bitget account list → BalanceSnapshot."""
    balances = []
    for acct in data.get("list", data if isinstance(data, list) else [data]):
        if not isinstance(acct, dict):
            continue
        asset = acct.get("marginCoin", "USDT")
        available = _d(acct.get("available", acct.get("crossedMaxAvailable", "0")))
        locked = _d(acct.get("locked", "0"))
        equity = _d(acct.get("usdtEquity", acct.get("equity", "0")))
        total = equity if equity > 0 else available + locked
        balances.append(CanonicalBalance(
            venue=_VENUE,
            asset=asset,
            free=available,
            locked=locked,
            total=total,
            ts_ms=int(time.time() * 1000),
            raw=acct,
        ))
    return BalanceSnapshot(
        venue=_VENUE,
        balances=tuple(balances),
        ts_ms=int(time.time() * 1000),
    )


# ── Position ─────────────────────────────────────────────────────

def map_position(pos: dict) -> VenuePosition:
    """Convert Bitget position → VenuePosition (signed qty)."""
    symbol = pos.get("symbol", "")
    hold_side = pos.get("holdSide", "long").lower()
    size = _d(pos.get("total", pos.get("available", "0")))
    if hold_side == "short":
        size = -size

    return VenuePosition(
        venue=_VENUE,
        symbol=symbol,
        qty=size,
        entry_price=_d(pos.get("openPriceAvg", pos.get("averageOpenPrice", "0"))),
        mark_price=_d(pos.get("markPrice", "0")),
        liquidation_price=_d(pos.get("liquidationPrice", "0")),
        unrealized_pnl=_d(pos.get("unrealizedPL", "0")),
        leverage=_d(pos.get("leverage", "1")),
        margin_type=pos.get("marginMode", "crossed"),
        ts_ms=int(pos.get("cTime", str(int(time.time() * 1000)))),
        raw=pos,
    )


# ── Order ────────────────────────────────────────────────────────

_STATUS_MAP = {
    "live": "new",
    "new": "new",
    "init": "new",
    "partially_filled": "partially_filled",
    "partial-fill": "partially_filled",
    "filled": "filled",
    "full-fill": "filled",
    "cancelled": "canceled",
    "canceled": "canceled",
}


def map_order(order: dict) -> CanonicalOrder:
    """Convert Bitget order → CanonicalOrder."""
    raw_status = order.get("status", order.get("state", "")).lower().replace("_", "-")
    status = _STATUS_MAP.get(raw_status, raw_status)
    side = order.get("side", "").lower()

    return CanonicalOrder(
        venue=_VENUE,
        symbol=order.get("symbol", ""),
        order_id=order.get("orderId", ""),
        client_order_id=order.get("clientOid", ""),
        status=status,
        side=side.replace("open_", "").replace("close_", ""),
        order_type=order.get("orderType", "market").lower(),
        tif=order.get("force", "gtc").lower(),
        qty=_d(order.get("size", "0")),
        price=_d(order.get("price", "0")),
        filled_qty=_d(order.get("baseVolume", order.get("filledQty", "0"))),
        avg_price=_d(order.get("priceAvg", "0")),
        ts_ms=int(order.get("cTime", str(int(time.time() * 1000)))),
        raw=order,
    )


# ── Fill ─────────────────────────────────────────────────────────

def map_fill(fill: dict) -> CanonicalFill:
    """Convert Bitget trade/fill → CanonicalFill."""
    symbol = fill.get("symbol", "")
    trade_id = fill.get("tradeId", fill.get("fillId", ""))
    fid = fill_key(venue=_VENUE, symbol=symbol, trade_id=trade_id)
    side = fill.get("side", "").lower()
    qty = _d(fill.get("size", fill.get("amount", "0")))
    price = _d(fill.get("price", fill.get("priceAvg", "0")))
    fee = _d(fill.get("fee", "0"))
    fee_asset = fill.get("feeDetail", {}).get("feeCoin", "USDT") if isinstance(fill.get("feeDetail"), dict) else "USDT"
    ts_ms = int(fill.get("cTime", str(int(time.time() * 1000))))

    return CanonicalFill(
        venue=_VENUE,
        symbol=symbol,
        order_id=fill.get("orderId", ""),
        trade_id=trade_id,
        fill_id=fid,
        side=side.replace("open_", "").replace("close_", ""),
        qty=qty,
        price=price,
        fee=fee,
        fee_asset=fee_asset,
        liquidity="maker" if fill.get("tradeScope") == "maker" else "taker",
        ts_ms=ts_ms,
        payload_digest=fill_digest(
            symbol=symbol, order_id=fill.get("orderId", ""),
            trade_id=trade_id, side=side, qty=qty, price=price,
            fee=fee, fee_asset=fee_asset, ts_ms=ts_ms,
        ),
        raw=fill,
    )
