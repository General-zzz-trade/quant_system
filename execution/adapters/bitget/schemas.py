# execution/adapters/bitget/schemas.py
"""Bitget API response schemas — field name constants."""
from __future__ import annotations

# Order response fields
ORDER_ID = "orderId"
CLIENT_OID = "clientOid"
SYMBOL = "symbol"
SIDE = "side"
ORDER_TYPE = "orderType"
STATUS = "status"
PRICE = "price"
SIZE = "size"
PRICE_AVG = "priceAvg"
FORCE = "force"
TRADE_SIDE = "tradeSide"
CREATE_TIME = "cTime"
UPDATE_TIME = "uTime"
BASE_VOLUME = "baseVolume"

# Fill/trade fields
TRADE_ID = "tradeId"
FILL_PRICE = "price"
FILL_QTY = "baseVolume"
FILL_FEE = "fee"
FILL_FEE_COIN = "feeCoin"
FILL_TIME = "cTime"
FILL_SIDE = "side"
FILL_FEE_DETAIL = "feeDetail"

# Position fields
POS_SYMBOL = "symbol"
POS_TOTAL = "total"
POS_AVAILABLE = "available"
POS_HOLD_SIDE = "holdSide"
POS_ENTRY_PRICE = "openPriceAvg"
POS_UNREALIZED = "unrealizedPL"
POS_LEVERAGE = "leverage"
POS_MARGIN_MODE = "marginMode"
POS_MARGIN_COIN = "marginCoin"

# Balance/account fields
BAL_MARGIN_COIN = "marginCoin"
BAL_AVAILABLE = "available"
BAL_LOCKED = "locked"
BAL_EQUITY = "accountEquity"
BAL_CROSS_MAX_AVAILABLE = "crossMaxAvailable"

# Product type constant
PRODUCT_TYPE = "USDT-FUTURES"
