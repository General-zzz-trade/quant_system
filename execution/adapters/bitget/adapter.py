"""Bitget V2 venue adapter for USDT-M futures."""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from execution.adapters.bitget.client import BitgetRestClient
from execution.adapters.bitget.config import BitgetConfig
from execution.adapters.bitget.mapper import (
    map_balance, map_fill, map_instrument, map_order, map_position,
)
from execution.models.balances import BalanceSnapshot
from execution.models.fills import CanonicalFill
from execution.models.instruments import InstrumentInfo
from execution.models.orders import CanonicalOrder
from execution.models.positions import VenuePosition

logger = logging.getLogger(__name__)


class BitgetAdapter:
    """Bitget V2 USDT-M futures adapter.

    Implements VenueAdapter protocol for integration with the quant system.
    Endpoints reference: https://www.bitget.com/api-doc/contract/intro

    Key differences from Bybit:
    - Signing: base64(HMAC-SHA256) with passphrase header
    - Product type: "USDT-FUTURES" (vs Bybit "linear")
    - Success code: "00000" (vs Bybit retCode=0)
    - Symbol format: same ("ETHUSDT")
    - Position: holdSide="long"/"short" (vs Bybit side="Buy"/"Sell")
    """

    venue: str = "bitget"

    def __init__(self, config: BitgetConfig) -> None:
        self._config = config
        self._client = BitgetRestClient(config)
        self._connected = False

    # ── Connection ────────────────────────────────────────────────

    def connect(self) -> bool:
        """Test connection by fetching account info."""
        data = self._client.get("/api/v2/mix/account/accounts", {
            "productType": self._config.product_type,
        })
        if str(data.get("code")) == "00000":
            self._connected = True
            n_accts = len(data.get("data", []))
            logger.info("Bitget connected: %s, %d assets with balance",
                        self._config.base_url, n_accts)
            return True
        logger.error("Bitget connection failed: code=%s msg=%s",
                      data.get("code"), data.get("msg"))
        return False

    def is_connected(self) -> bool:
        return self._connected

    # ── VenueAdapter Protocol ─────────────────────────────────────

    def list_instruments(self, symbols: list[str] | None = None) -> Tuple[InstrumentInfo, ...]:
        data = self._client.get("/api/v2/mix/market/contracts", {
            "productType": self._config.product_type,
        })
        items = data.get("data", [])
        instruments = tuple(map_instrument(item) for item in items)
        if symbols:
            instruments = tuple(i for i in instruments if i.symbol in symbols)
        return instruments

    def get_balances(self) -> BalanceSnapshot:
        data = self._client.get("/api/v2/mix/account/accounts", {
            "productType": self._config.product_type,
        })
        return map_balance({"list": data.get("data", [])})

    def get_positions(self, symbol: str = "") -> Tuple[VenuePosition, ...]:
        params: dict[str, str] = {"productType": self._config.product_type}
        if symbol:
            params["symbol"] = symbol
        data = self._client.get("/api/v2/mix/position/all-position", params)
        return tuple(
            map_position(p) for p in data.get("data", [])
            if float(p.get("total", p.get("available", "0"))) != 0
        )

    def get_open_orders(self, *, symbol: Optional[str] = None) -> Tuple[CanonicalOrder, ...]:
        params: dict[str, str] = {"productType": self._config.product_type}
        if symbol:
            params["symbol"] = symbol
        data = self._client.get("/api/v2/mix/order/orders-pending", params)
        return tuple(map_order(o) for o in data.get("data", {}).get("entrustedList", []))

    def get_recent_fills(self, *, symbol: Optional[str] = None,
                         since_ms: int = 0) -> Tuple[CanonicalFill, ...]:
        params: dict[str, str] = {"productType": self._config.product_type}
        if symbol:
            params["symbol"] = symbol
        data = self._client.get("/api/v2/mix/order/fills", params)
        return tuple(map_fill(f) for f in data.get("data", {}).get("fillList", []))

    # ── Order Methods ─────────────────────────────────────────────

    def send_market_order(self, symbol: str, side: str, qty: float,
                          *, reduce_only: bool = False) -> dict:
        """Place a market order."""
        trade_side = "close" if reduce_only else "open"
        # Bitget side: "buy" or "sell"
        order_side = "buy" if side.lower() == "buy" else "sell"

        body: dict[str, Any] = {
            "symbol": symbol,
            "productType": self._config.product_type,
            "marginMode": "crossed",
            "marginCoin": self._config.margin_coin,
            "side": order_side,
            "tradeSide": trade_side,
            "orderType": "market",
            "size": str(qty),
        }
        data = self._client.post("/api/v2/mix/order/place-order", body)
        code = str(data.get("code", ""))
        result_data = data.get("data", {})
        if code == "00000":
            return {
                "status": "ok",
                "orderId": result_data.get("orderId", ""),
                "clientOid": result_data.get("clientOid", ""),
            }
        return {"status": "error", "code": code, "msg": data.get("msg", "")}

    def send_limit_order(self, symbol: str, side: str, qty: float, price: float,
                         *, tif: str = "gtc", reduce_only: bool = False) -> dict:
        """Place a limit order."""
        trade_side = "close" if reduce_only else "open"
        body: dict[str, Any] = {
            "symbol": symbol,
            "productType": self._config.product_type,
            "marginMode": "crossed",
            "marginCoin": self._config.margin_coin,
            "side": "buy" if side.lower() == "buy" else "sell",
            "tradeSide": trade_side,
            "orderType": "limit",
            "size": str(qty),
            "price": str(price),
            "force": tif,
        }
        data = self._client.post("/api/v2/mix/order/place-order", body)
        code = str(data.get("code", ""))
        result_data = data.get("data", {})
        if code == "00000":
            return {
                "status": "ok",
                "orderId": result_data.get("orderId", ""),
                "clientOid": result_data.get("clientOid", ""),
            }
        return {"status": "error", "code": code, "msg": data.get("msg", "")}

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        body = {
            "symbol": symbol,
            "productType": self._config.product_type,
            "orderId": order_id,
        }
        data = self._client.post("/api/v2/mix/order/cancel-order", body)
        return {"status": "ok" if str(data.get("code")) == "00000" else "error",
                "msg": data.get("msg", "")}

    def close_position(self, symbol: str) -> dict:
        """Close position by reading current position and sending opposite market order."""
        positions = self.get_positions(symbol=symbol)
        for pos in positions:
            if pos.symbol == symbol and pos.qty != 0:
                side = "sell" if pos.qty > 0 else "buy"
                qty = float(abs(pos.qty))
                return self.send_market_order(symbol, side, qty, reduce_only=True)
        return {"status": "no_position"}

    # ── Market Data ───────────────────────────────────────────────

    def get_ticker(self, symbol: str) -> dict:
        """Get current ticker for symbol."""
        data = self._client.get("/api/v2/mix/market/ticker", {
            "symbol": symbol,
            "productType": self._config.product_type,
        })
        items = data.get("data", [])
        if items:
            t = items[0] if isinstance(items, list) else items
            return {
                "lastPrice": float(t.get("lastPr", t.get("last", 0))),
                "bid1Price": float(t.get("bidPr", t.get("bestBid", 0))),
                "ask1Price": float(t.get("askPr", t.get("bestAsk", 0))),
                "volume24h": float(t.get("baseVolume", 0)),
                "fundingRate": float(t.get("fundingRate", 0)),
            }
        return {}

    def get_klines(self, symbol: str, interval: str = "1H",
                   limit: int = 200) -> list[dict]:
        """Get historical klines.

        Bitget granularity: "1m","5m","15m","30m","1H","4H","6H","12H","1D","3D","1W","1M"
        """
        data = self._client.get("/api/v2/mix/market/candles", {
            "symbol": symbol,
            "productType": self._config.product_type,
            "granularity": interval,
            "limit": str(limit),
        })
        # Bitget returns: [[ts, open, high, low, close, volume, quoteVolume], ...]
        bars = []
        for k in data.get("data", []):
            if not isinstance(k, list) or len(k) < 6:
                continue
            bars.append({
                "time": int(k[0]) // 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        # Bitget returns newest first — reverse to chronological
        bars.reverse()
        return bars

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for symbol."""
        body = {
            "symbol": symbol,
            "productType": self._config.product_type,
            "marginCoin": self._config.margin_coin,
            "leverage": str(leverage),
        }
        data = self._client.post("/api/v2/mix/account/set-leverage", body)
        return {"status": "ok" if str(data.get("code")) == "00000" else "error",
                "msg": data.get("msg", "")}

    def get_funding_rate(self, symbol: str) -> dict:
        """Get current funding rate."""
        data = self._client.get("/api/v2/mix/market/current-fund-rate", {
            "symbol": symbol,
            "productType": self._config.product_type,
        })
        items = data.get("data", [])
        if items:
            r = items[0] if isinstance(items, list) else items
            return {
                "symbol": r.get("symbol", symbol),
                "fundingRate": float(r.get("fundingRate", 0)),
            }
        return {"symbol": symbol, "fundingRate": 0.0}
