# execution/adapters/binance/schemas.py
"""Binance API response schemas — field name constants."""
from __future__ import annotations

# Order response fields
ORDER_ID = "orderId"
CLIENT_ORDER_ID = "clientOrderId"
SYMBOL = "symbol"
SIDE = "side"
TYPE = "type"
STATUS = "status"
PRICE = "price"
ORIG_QTY = "origQty"
EXECUTED_QTY = "executedQty"
CUM_QUOTE = "cumQuote"
TIME_IN_FORCE = "timeInForce"
REDUCE_ONLY = "reduceOnly"
UPDATE_TIME = "updateTime"

# Fill/trade fields
TRADE_ID = "id"
TRADE_PRICE = "price"
TRADE_QTY = "qty"
TRADE_FEE = "commission"
TRADE_FEE_ASSET = "commissionAsset"
TRADE_TIME = "time"
TRADE_BUYER = "buyer"

# Position fields
POS_SYMBOL = "symbol"
POS_AMT = "positionAmt"
POS_ENTRY_PRICE = "entryPrice"
POS_UNREALIZED = "unRealizedProfit"
POS_LEVERAGE = "leverage"
POS_SIDE = "positionSide"

# Balance fields
BAL_ASSET = "asset"
BAL_BALANCE = "balance"
BAL_AVAILABLE = "availableBalance"
BAL_CROSS_UNREALIZED = "crossUnPnl"

# User data stream event types
EVENT_TYPE = "e"
EVENT_ACCOUNT_UPDATE = "ACCOUNT_UPDATE"
EVENT_ORDER_TRADE_UPDATE = "ORDER_TRADE_UPDATE"
EVENT_MARGIN_CALL = "MARGIN_CALL"
