# execution/adapters/ib/mapper.py
"""Map IB data structures to canonical execution models."""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from ib_insync import (
    CFD,
    Contract,
    Crypto,
    Forex,
    Future,
    Option,
    Stock,
)

from execution.models.balances import BalanceSnapshot, CanonicalBalance
from execution.models.fills import CanonicalFill
from execution.models.instruments import InstrumentInfo
from execution.models.orders import CanonicalOrder
from execution.models.positions import VenuePosition

_VENUE = "ib"


def _d(v: Any) -> Decimal:
    """Convert to Decimal safely."""
    if v is None:
        return Decimal("0")
    if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
        return Decimal("0")
    return Decimal(str(v))


# -- Contract builders --


def make_contract(symbol: str, sec_type: str = "STK", **kwargs: Any) -> Contract:
    """Build an IB Contract from simplified parameters.

    Examples:
        make_contract("AAPL")                         → Stock
        make_contract("EURUSD", "CASH")               → Forex
        make_contract("ES", "FUT", expiry="202412")   → Future
        make_contract("AAPL", "OPT", expiry="20241220", strike=200, right="C") → Option
        make_contract("XAUUSD", "CFD")                → CFD
        make_contract("BTC", "CRYPTO")                → Crypto
    """
    sec_type = sec_type.upper()
    if sec_type in ("STK", "STOCK"):
        return Stock(symbol, kwargs.get("exchange", "SMART"), kwargs.get("currency", "USD"))
    elif sec_type in ("CASH", "FX", "FOREX"):
        pair = symbol.replace("/", "")
        return Forex(pair[:3] + pair[3:])
    elif sec_type in ("FUT", "FUTURE"):
        return Future(
            symbol,
            kwargs.get("expiry", ""),
            kwargs.get("exchange", "CME"),
            currency=kwargs.get("currency", "USD"),
        )
    elif sec_type in ("OPT", "OPTION"):
        return Option(
            symbol,
            kwargs.get("expiry", ""),
            kwargs.get("strike", 0),
            kwargs.get("right", "C"),
            kwargs.get("exchange", "SMART"),
            currency=kwargs.get("currency", "USD"),
        )
    elif sec_type == "CFD":
        return CFD(symbol, currency=kwargs.get("currency", "USD"))
    elif sec_type == "CRYPTO":
        return Crypto(symbol, currency=kwargs.get("currency", "USD"))
    else:
        c = Contract()
        c.symbol = symbol
        c.secType = sec_type
        c.exchange = kwargs.get("exchange", "SMART")
        c.currency = kwargs.get("currency", "USD")
        return c


# -- Instrument mapping --


def map_contract_details(cd: Any) -> InstrumentInfo:
    """Convert IB ContractDetails to InstrumentInfo."""
    c = cd.contract
    symbol = c.localSymbol or c.symbol

    # Determine contract type
    sec_type_map = {
        "STK": None,
        "CASH": "forex",
        "FUT": "futures",
        "OPT": "options",
        "CFD": "cfd",
        "CRYPTO": "perpetual",
    }
    contract_type = sec_type_map.get(c.secType)

    tick_size = Decimal(str(cd.minTick)) if cd.minTick else Decimal("0.01")

    return InstrumentInfo(
        venue=_VENUE,
        symbol=symbol,
        base_asset=c.symbol,
        quote_asset=c.currency,
        price_precision=max(0, -tick_size.as_tuple().exponent) if tick_size > 0 else 2,
        qty_precision=0 if c.secType == "STK" else 2,
        tick_size=tick_size,
        lot_size=Decimal("1") if c.secType == "STK" else Decimal("0.01"),
        min_qty=Decimal("1") if c.secType == "STK" else Decimal("0.01"),
        min_notional=Decimal("0"),
        contract_type=contract_type,
        margin_asset=c.currency,
        trading_enabled=True,
    )


# -- Position mapping --


def map_position(pos: Any) -> VenuePosition:
    """Convert IB Position to VenuePosition."""
    qty = _d(pos.position)
    avg_cost = _d(pos.avgCost)
    contract = pos.contract
    symbol = contract.localSymbol or contract.symbol

    # For forex, avgCost is per unit; for stocks it's per share
    entry_price = avg_cost
    if contract.secType == "CASH":
        entry_price = avg_cost  # already price per unit for forex

    return VenuePosition(
        venue=_VENUE,
        symbol=symbol,
        qty=qty,
        entry_price=entry_price,
        mark_price=None,  # filled by ticker
        unrealized_pnl=Decimal("0"),  # filled by pnl subscription
        leverage=None,
        margin_type="cross",
        ts_ms=int(time.time() * 1000),
        raw={
            "account": pos.account,
            "secType": contract.secType,
            "exchange": contract.exchange,
            "currency": contract.currency,
            "conId": contract.conId,
        },
    )


# -- Order mapping --

_IB_ORDER_STATUS = {
    "PendingSubmit": "new",
    "PendingCancel": "new",
    "PreSubmitted": "new",
    "Submitted": "new",
    "ApiPending": "new",
    "ApiCancelled": "canceled",
    "Cancelled": "canceled",
    "Filled": "filled",
    "Inactive": "rejected",
}


def map_trade(trade: Any) -> CanonicalOrder:
    """Convert IB Trade to CanonicalOrder."""
    order = trade.order
    contract = trade.contract
    symbol = contract.localSymbol or contract.symbol
    status_str = trade.orderStatus.status if trade.orderStatus else "new"

    side = "buy" if order.action == "BUY" else "sell"
    order_type = order.orderType.lower().replace(" ", "_")
    # Normalize: MKT→market, LMT→limit, STP→stop, STP LMT→stop_limit
    type_map = {"mkt": "market", "lmt": "limit", "stp": "stop", "stp_lmt": "stop_limit"}
    order_type = type_map.get(order_type, order_type)

    tif_map = {"GTC": "gtc", "IOC": "ioc", "FOK": "fok", "DAY": "day"}
    tif = tif_map.get(order.tif, order.tif.lower() if order.tif else "day")

    filled_qty = _d(trade.orderStatus.filled) if trade.orderStatus else Decimal("0")
    avg_price = _d(trade.orderStatus.avgFillPrice) if trade.orderStatus and trade.orderStatus.avgFillPrice else None

    return CanonicalOrder(
        venue=_VENUE,
        symbol=symbol,
        order_id=str(order.orderId),
        client_order_id=str(order.permId) if order.permId else None,
        status=_IB_ORDER_STATUS.get(status_str, "new"),
        side=side,
        order_type=order_type,
        tif=tif,
        qty=_d(order.totalQuantity),
        price=_d(order.lmtPrice) if order.lmtPrice and order.lmtPrice > 0 else None,
        filled_qty=filled_qty,
        avg_price=avg_price,
        ts_ms=int(time.time() * 1000),
        raw={
            "orderId": order.orderId,
            "permId": order.permId,
            "secType": contract.secType,
            "exchange": contract.exchange,
            "ib_status": status_str,
        },
    )


# -- Fill mapping --


def map_fill(fill: Any) -> CanonicalFill:
    """Convert IB Fill to CanonicalFill."""
    exe = fill.execution
    contract = fill.contract
    symbol = contract.localSymbol or contract.symbol

    side = "buy" if exe.side == "BOT" else "sell"

    # Commission from CommissionReport
    fee = Decimal("0")
    if fill.commissionReport and fill.commissionReport.commission:
        commission = fill.commissionReport.commission
        if commission < 1e9:  # IB uses 1e10 for "not yet reported"
            fee = _d(commission)

    return CanonicalFill(
        venue=_VENUE,
        symbol=symbol,
        order_id=str(exe.orderId),
        trade_id=str(exe.execId),
        fill_id=f"ib:{exe.execId}",
        side=side,
        qty=_d(exe.shares),
        price=_d(exe.price),
        fee=fee,
        fee_asset=fill.commissionReport.currency if fill.commissionReport else None,
        liquidity="maker" if exe.liquidation else None,
        ts_ms=int(time.time() * 1000),
        raw={
            "execId": exe.execId,
            "orderId": exe.orderId,
            "acctNumber": exe.acctNumber,
            "exchange": exe.exchange,
        },
    )


# -- Balance mapping --


def map_account_values(account_values: list, account: str = "") -> BalanceSnapshot:
    """Convert IB AccountValue list to BalanceSnapshot."""
    now_ms = int(time.time() * 1000)

    # Extract key values
    vals: dict[str, float] = {}
    for av in account_values:
        if av.currency == "BASE" or av.currency == "":
            continue
        if av.tag in ("TotalCashBalance", "NetLiquidation", "BuyingPower",
                       "AvailableFunds", "MaintMarginReq", "UnrealizedPnL"):
            key = f"{av.tag}_{av.currency}"
            try:
                vals[key] = float(av.value)
            except (ValueError, TypeError):
                pass
            if av.tag == "TotalCashBalance":
                pass  # currency tracked via currencies_seen below

    # Group by currency
    currencies_seen: set[str] = set()
    balances: list[CanonicalBalance] = []
    for av in account_values:
        if av.tag == "TotalCashBalance" and av.currency not in ("BASE", "") and av.currency not in currencies_seen:
            currencies_seen.add(av.currency)
            try:
                cash = float(av.value)
            except (ValueError, TypeError):
                continue
            margin_key = f"MaintMarginReq_{av.currency}"
            locked = vals.get(margin_key, 0)
            balances.append(CanonicalBalance.from_free_locked(
                venue=_VENUE,
                asset=av.currency,
                free=_d(cash - locked),
                locked=_d(locked),
                ts_ms=now_ms,
            ))

    # Fallback: use base currency summary
    if not balances:
        net_liq = vals.get("NetLiquidation_USD", 0)
        avail = vals.get("AvailableFunds_USD", 0)
        balances.append(CanonicalBalance.from_free_locked(
            venue=_VENUE,
            asset="USD",
            free=_d(avail),
            locked=_d(net_liq - avail),
            ts_ms=now_ms,
        ))

    return BalanceSnapshot(venue=_VENUE, balances=tuple(balances), ts_ms=now_ms)
