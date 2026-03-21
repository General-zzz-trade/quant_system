# execution/adapters/hyperliquid/mapper.py
"""Map Hyperliquid API responses to canonical execution models."""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Optional

from execution.models.balances import BalanceSnapshot, CanonicalBalance
from execution.models.fills import CanonicalFill
from execution.models.instruments import InstrumentInfo
from execution.models.orders import CanonicalOrder
from execution.models.positions import VenuePosition
from execution.models.digest import fill_key, fill_digest

_VENUE = "hyperliquid"


def _d(v: Any) -> Decimal:
    """Convert value to Decimal, defaulting to 0."""
    if v is None or v == "":
        return Decimal("0")
    return Decimal(str(v))


def _sz_decimals_to_lot(sz_decimals: int) -> Decimal:
    """Convert szDecimals (number of decimal places) to lot_size step."""
    if sz_decimals <= 0:
        return Decimal("1")
    return Decimal("1") / Decimal(10 ** sz_decimals)


# -- Order status mapping --
_ORDER_STATUS = {
    "open": "new",
    "filled": "filled",
    "canceled": "canceled",
    "triggered": "new",
    "rejected": "rejected",
    "marginCanceled": "canceled",
}


def normalize_coin(symbol: str) -> str:
    """Normalize symbol to Hyperliquid coin name (no USDT suffix).

    Accepts: "BTC", "BTCUSDT", "btcusdt", "ETH", "ETHUSDT"
    Returns: "BTC", "ETH", etc.
    """
    s = symbol.upper().strip()
    # Remove common suffixes
    for suffix in ("USDT", "USD", "PERP"):
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[:-len(suffix)]
    return s


def coin_to_symbol(coin: str) -> str:
    """Convert Hyperliquid coin name to system symbol (with USDT suffix).

    "BTC" -> "BTCUSDT", "ETH" -> "ETHUSDT"
    """
    c = coin.upper().strip()
    if c.endswith("USDT"):
        return c
    return f"{c}USDT"


def map_instrument(
    asset_info: dict, asset_index: int, asset_ctx: Optional[dict] = None,
) -> InstrumentInfo:
    """Convert Hyperliquid universe entry to InstrumentInfo.

    Args:
        asset_info: Entry from meta.universe, e.g. {"name": "BTC", "szDecimals": 5, ...}
        asset_index: Index in the universe array (used as asset ID for orders).
        asset_ctx: Optional asset context from metaAndAssetCtxs for additional data.
    """
    coin = asset_info.get("name", "")
    sz_decimals = int(asset_info.get("szDecimals", 0))
    lot_size = _sz_decimals_to_lot(sz_decimals)

    # Price precision from asset context if available
    price_precision = 8  # default
    tick_size = Decimal("0.01")  # default
    if asset_ctx:
        mark_px = asset_ctx.get("markPx", "")
        if mark_px:
            # Infer tick size from the mark price decimal places
            parts = str(mark_px).split(".")
            if len(parts) == 2:
                price_precision = len(parts[1])
                tick_size = Decimal("1") / Decimal(10 ** price_precision)

    return InstrumentInfo(
        venue=_VENUE,
        symbol=coin_to_symbol(coin),
        base_asset=coin,
        quote_asset="USDC",
        price_precision=price_precision,
        qty_precision=sz_decimals,
        tick_size=tick_size,
        lot_size=lot_size,
        min_qty=lot_size,  # Minimum is one lot
        max_qty=None,
        min_notional=Decimal("10"),  # Hyperliquid min ~$10
        contract_type="perpetual",
        margin_asset="USDC",
        trading_enabled=True,
    )


def map_position(pos: dict, coin_to_symbol_map: dict[str, str] | None = None) -> VenuePosition:
    """Convert Hyperliquid position to VenuePosition.

    Args:
        pos: Position dict from clearinghouseState.assetPositions[].position
        coin_to_symbol_map: Optional mapping of coin -> symbol
    """
    coin = pos.get("coin", "")
    symbol = coin_to_symbol(coin)
    if coin_to_symbol_map and coin in coin_to_symbol_map:
        symbol = coin_to_symbol_map[coin]

    szi = pos.get("szi", "0")
    qty = _d(szi)  # Already signed: positive=long, negative=short

    entry_px = pos.get("entryPx")
    liquidation_px = pos.get("liquidationPx")
    unrealized_pnl = _d(pos.get("unrealizedPnl", "0"))
    leverage_info = pos.get("leverage", {})
    leverage_val = None
    margin_type = None
    if isinstance(leverage_info, dict):
        leverage_val = int(leverage_info.get("value", 0)) if leverage_info.get("value") else None
        lev_type = leverage_info.get("type", "")
        margin_type = "cross" if lev_type == "cross" else "isolated" if lev_type == "isolated" else None
    elif isinstance(leverage_info, (int, float)):
        leverage_val = int(leverage_info)

    return VenuePosition(
        venue=_VENUE,
        symbol=symbol,
        qty=qty,
        entry_price=_d(entry_px) if entry_px else None,
        mark_price=None,  # Not in position data; available from asset context
        liquidation_price=_d(liquidation_px) if liquidation_px else None,
        unrealized_pnl=unrealized_pnl,
        leverage=leverage_val,
        margin_type=margin_type,
        ts_ms=int(time.time() * 1000),
        raw=pos,
    )


def map_order(order: dict) -> CanonicalOrder:
    """Convert Hyperliquid open order to CanonicalOrder."""
    coin = order.get("coin", "")
    symbol = coin_to_symbol(coin)
    side = "buy" if order.get("side", "").upper() == "B" or order.get("side", "").lower() == "buy" else "sell"

    oid = str(order.get("oid", ""))
    status_raw = order.get("orderStatus", order.get("status", "open"))
    status = _ORDER_STATUS.get(status_raw, "new")

    limit_px = order.get("limitPx", order.get("price"))
    sz = order.get("sz", order.get("origSz", "0"))

    order_type = "limit"
    tif = "gtc"
    order_type_info = order.get("orderType")
    if isinstance(order_type_info, str):
        if order_type_info.lower() == "market":
            order_type = "market"
    elif isinstance(order_type_info, dict):
        if "limit" in order_type_info:
            tif_val = order_type_info["limit"].get("tif", "Gtc")
            tif = tif_val.lower() if tif_val else "gtc"

    return CanonicalOrder(
        venue=_VENUE,
        symbol=symbol,
        order_id=oid,
        client_order_id=str(order.get("cloid")) if order.get("cloid") else None,
        status=status,
        side=side,
        order_type=order_type,
        tif=tif,
        qty=_d(sz),
        price=_d(limit_px) if limit_px else None,
        filled_qty=_d(order.get("filledSz", "0")),
        avg_price=_d(order.get("avgPx")) if order.get("avgPx") else None,
        ts_ms=int(order.get("timestamp", 0)),
        raw=order,
    )


def map_fill(fill: dict) -> CanonicalFill:
    """Convert Hyperliquid fill to CanonicalFill."""
    coin = fill.get("coin", "")
    symbol = coin_to_symbol(coin)
    side = "buy" if fill.get("side", "").upper() == "B" or fill.get("side", "").lower() == "buy" else "sell"
    dir_str = fill.get("dir", "")
    if dir_str:
        # dir can be "Open Long", "Close Short", "Open Short", "Close Long"
        if "long" in dir_str.lower() or "buy" in dir_str.lower():
            side = "buy"
        elif "short" in dir_str.lower() or "sell" in dir_str.lower():
            side = "sell"

    trade_id = str(fill.get("tid", fill.get("tradeId", "")))
    order_id = str(fill.get("oid", ""))
    qty = _d(fill.get("sz", "0"))
    price = _d(fill.get("px", "0"))
    fee = _d(fill.get("fee", "0"))
    ts_ms = int(fill.get("time", 0))

    # Determine liquidity from crossed flag or fee sign
    liquidity = None
    if fill.get("crossed") is True:
        liquidity = "taker"
    elif fill.get("crossed") is False:
        liquidity = "maker"
    elif fee > 0:
        liquidity = "taker"
    elif fee < 0:
        liquidity = "maker"  # Negative fee = rebate = maker

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
        fee_asset="USDC",
        liquidity=liquidity,
        ts_ms=ts_ms,
        payload_digest=fill_digest(
            symbol=symbol, order_id=order_id, trade_id=trade_id,
            side=side, qty=qty, price=price, fee=abs(fee),
            ts_ms=ts_ms,
        ),
        raw=fill,
    )


def map_balance(state: dict) -> BalanceSnapshot:
    """Convert Hyperliquid clearinghouseState to BalanceSnapshot.

    Args:
        state: Full clearinghouseState response.
    """
    now_ms = int(time.time() * 1000)
    balances = []

    margin_summary = state.get("marginSummary", state.get("crossMarginSummary", {}))
    if margin_summary:
        account_value = float(margin_summary.get("accountValue", 0))
        total_margin_used = float(margin_summary.get("totalMarginUsed", 0))
        if account_value > 0:
            free = account_value - total_margin_used
            balances.append(CanonicalBalance.from_free_locked(
                venue=_VENUE,
                asset="USDC",
                free=_d(max(free, 0)),
                locked=_d(total_margin_used),
                ts_ms=now_ms,
                raw=margin_summary,
            ))

    return BalanceSnapshot(venue=_VENUE, balances=tuple(balances), ts_ms=now_ms)


def map_orderbook(data: dict) -> dict[str, Any]:
    """Convert Hyperliquid L2 book to standard orderbook format.

    Args:
        data: Response from {"type": "l2Book", "coin": "BTC"}

    Returns:
        Dict with "bids", "asks" (list of [price, size] floats), "coin", "ts_ms".
    """
    levels = data.get("levels", [[], []])
    bids_raw = levels[0] if len(levels) > 0 else []
    asks_raw = levels[1] if len(levels) > 1 else []

    bids = [[float(b.get("px", 0)), float(b.get("sz", 0)), int(b.get("n", 0))]
            for b in bids_raw]
    asks = [[float(a.get("px", 0)), float(a.get("sz", 0)), int(a.get("n", 0))]
            for a in asks_raw]

    return {
        "bids": bids,
        "asks": asks,
        "coin": data.get("coin", ""),
        "ts_ms": int(time.time() * 1000),
    }
