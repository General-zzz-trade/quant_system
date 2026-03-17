# execution/adapters/bybit/adapter.py
"""Bybit venue adapter — USDT perpetual futures via V5 REST API.

Implements VenueAdapter protocol for Bybit linear perpetuals.
Supports demo trading (api-demo.bybit.com) and testnet.

Usage:
    from execution.adapters.bybit import BybitAdapter, BybitConfig

    config = BybitConfig.demo(api_key="...", api_secret="...")
    adapter = BybitAdapter(config)
    adapter.connect()

    print(adapter.get_balances())
    print(adapter.get_positions())
    adapter.send_market_order("BTCUSDT", "buy", 0.001)
"""
from __future__ import annotations

import logging
import os as _os
import time as _time
from typing import Any, Optional, Tuple

from execution.adapters.bybit.client import BybitRestClient
from execution.adapters.bybit.config import BybitConfig
from execution.adapters.bybit.mapper import (
    map_balance,
    map_fill,
    map_instrument,
    map_order,
    map_position,
)
from execution.models.balances import BalanceSnapshot
from execution.models.fills import CanonicalFill
from execution.models.instruments import InstrumentInfo
from execution.models.orders import CanonicalOrder
from execution.models.positions import VenuePosition

logger = logging.getLogger(__name__)


def _make_order_link_id(symbol: str, side: str) -> str:
    """Generate unique orderLinkId for Bybit dedup (max 36 chars)."""
    ts_ms = int(_time.time() * 1000)
    rand = _os.urandom(2).hex()  # 4 hex chars
    return f"qs_{symbol}_{side[0]}_{ts_ms}_{rand}"


class BybitAdapter:
    """Bybit venue adapter implementing VenueAdapter protocol."""

    venue: str = "bybit"

    def __init__(self, config: BybitConfig) -> None:
        self._config = config
        self._client = BybitRestClient(config)
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Test connection by fetching account balance."""
        data = self._client.get("/v5/account/wallet-balance",
                                {"accountType": self._config.account_type})
        if data.get("retCode") == 0:
            self._connected = True
            coins = data.get("result", {}).get("list", [{}])[0].get("coin", [])
            nonzero = [c for c in coins if float(c.get("walletBalance", 0)) > 0]
            logger.info(
                "Bybit connected: %s, %d assets with balance",
                self._config.base_url, len(nonzero),
            )
            return True
        logger.error("Bybit connection failed: %s", data.get("retMsg"))
        return False

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # VenueAdapter protocol
    # ------------------------------------------------------------------

    def list_instruments(self, symbols: list[str] | None = None) -> Tuple[InstrumentInfo, ...]:
        """Get instrument metadata for linear perpetuals."""
        params = {"category": self._config.category}
        if symbols and len(symbols) == 1:
            params["symbol"] = symbols[0]
        data = self._client.get("/v5/market/instruments-info", params)
        if data.get("retCode") != 0:
            return ()
        items = data.get("result", {}).get("list", [])
        instruments = [map_instrument(item) for item in items]
        if symbols:
            instruments = [i for i in instruments if i.symbol in symbols]
        return tuple(instruments)

    def get_balances(self) -> BalanceSnapshot:
        """Get current account balances."""
        data = self._client.get("/v5/account/wallet-balance",
                                {"accountType": self._config.account_type})
        if data.get("retCode") != 0:
            return BalanceSnapshot(venue="bybit", balances=(), ts_ms=0)
        return map_balance(data.get("result", {}))

    def get_positions(self, symbol: str = "") -> Tuple[VenuePosition, ...]:
        """Get open positions."""
        params: dict[str, str] = {"category": self._config.category}
        if symbol:
            params["symbol"] = symbol
        else:
            params["settleCoin"] = "USDT"  # required when no symbol specified
        data = self._client.get("/v5/position/list", params)
        if data.get("retCode") != 0:
            return ()
        items = data.get("result", {}).get("list", [])
        return tuple(
            map_position(p) for p in items
            if float(p.get("size", "0")) != 0
        )

    def get_open_orders(
        self, *, symbol: Optional[str] = None,
    ) -> Tuple[CanonicalOrder, ...]:
        """Get active (unfilled) orders."""
        params: dict[str, str] = {"category": self._config.category}
        if symbol:
            params["symbol"] = symbol
        data = self._client.get("/v5/order/realtime", params)
        if data.get("retCode") != 0:
            return ()
        items = data.get("result", {}).get("list", [])
        return tuple(map_order(o) for o in items)

    def get_recent_fills(
        self, *, symbol: Optional[str] = None, since_ms: int = 0,
    ) -> Tuple[CanonicalFill, ...]:
        """Get recent execution history."""
        params: dict[str, str] = {"category": self._config.category}
        if symbol:
            params["symbol"] = symbol
        data = self._client.get("/v5/execution/list", params)
        if data.get("retCode") != 0:
            return ()
        items = data.get("result", {}).get("list", [])
        return tuple(map_fill(f) for f in items)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def send_market_order(
        self, symbol: str, side: str, qty: float,
        *, reduce_only: bool = False,
    ) -> dict:
        """Send a market order with orderLinkId for deduplication."""
        # Generate unique orderLinkId: prevents duplicate orders on retry/timeout
        # Bybit rejects duplicate orderLinkId within 3 minutes
        order_link_id = _make_order_link_id(symbol, side)

        body = {
            "category": self._config.category,
            "symbol": symbol,
            "side": "Buy" if side.lower() == "buy" else "Sell",
            "orderType": "Market",
            "qty": str(qty),
            "orderLinkId": order_link_id,
        }
        if reduce_only:
            body["reduceOnly"] = True
        data = self._client.post("/v5/order/create", body)
        if data.get("retCode") == 0:
            result = data.get("result", {})
            logger.info(
                "Bybit %s %s %.4f orderId=%s",
                side, symbol, qty, result.get("orderId"),
            )
            return {
                "orderId": result.get("orderId"),
                "orderLinkId": result.get("orderLinkId"),
                "status": "submitted",
            }
        return {"status": "error", "retCode": data.get("retCode"),
                "retMsg": data.get("retMsg")}

    def send_limit_order(
        self, symbol: str, side: str, qty: float, price: float,
        *, tif: str = "GTC", reduce_only: bool = False,
    ) -> dict:
        """Send a limit order."""
        body = {
            "category": self._config.category,
            "symbol": symbol,
            "side": "Buy" if side.lower() == "buy" else "Sell",
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "timeInForce": tif.upper(),
        }
        if reduce_only:
            body["reduceOnly"] = True
        data = self._client.post("/v5/order/create", body)
        if data.get("retCode") == 0:
            result = data.get("result", {})
            return {
                "orderId": result.get("orderId"),
                "status": "submitted",
            }
        return {"status": "error", "retCode": data.get("retCode"),
                "retMsg": data.get("retMsg")}

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an order."""
        body = {
            "category": self._config.category,
            "symbol": symbol,
            "orderId": order_id,
        }
        data = self._client.post("/v5/order/cancel", body)
        return {
            "status": "canceled" if data.get("retCode") == 0 else "error",
            "retCode": data.get("retCode"),
            "retMsg": data.get("retMsg"),
        }

    def cancel_all(self, symbol: str = "") -> dict:
        """Cancel all open orders."""
        body: dict[str, Any] = {"category": self._config.category}
        if symbol:
            body["symbol"] = symbol
        data = self._client.post("/v5/order/cancel-all", body)
        return {
            "status": "canceled" if data.get("retCode") == 0 else "error",
            "retCode": data.get("retCode"),
        }

    def close_position(self, symbol: str) -> dict:
        """Close an open position with a market order."""
        positions = self.get_positions(symbol=symbol)
        for pos in positions:
            if pos.symbol == symbol and not pos.is_flat:
                side = "sell" if pos.is_long else "buy"
                qty = float(pos.abs_qty)
                return self.send_market_order(
                    symbol, side, qty, reduce_only=True,
                )
        return {"status": "no_position"}

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> dict:
        """Get latest ticker (last price, bid, ask, volume)."""
        data = self._client.get("/v5/market/tickers", {
            "category": self._config.category, "symbol": symbol,
        })
        if data.get("retCode") != 0:
            return {}
        items = data.get("result", {}).get("list", [])
        if not items:
            return {}
        t = items[0]
        return {
            "symbol": t.get("symbol"),
            "lastPrice": float(t.get("lastPrice", 0)),
            "bid1Price": float(t.get("bid1Price", 0)),
            "ask1Price": float(t.get("ask1Price", 0)),
            "volume24h": float(t.get("volume24h", 0)),
            "turnover24h": float(t.get("turnover24h", 0)),
            "fundingRate": float(t.get("fundingRate", 0)),
        }

    def get_klines(
        self, symbol: str, interval: str = "60", limit: int = 200,
    ) -> list[dict]:
        """Get historical klines. interval: "1"=1m, "5"=5m, "60"=1h, "D"=1d."""
        data = self._client.get("/v5/market/kline", {
            "category": self._config.category,
            "symbol": symbol,
            "interval": interval,
            "limit": str(limit),
        })
        if data.get("retCode") != 0:
            return []
        items = data.get("result", {}).get("list", [])
        return [
            {
                "time": int(k[0]) // 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            for k in items
        ]

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_info(self) -> dict:
        """Get detailed account information."""
        bal = self.get_balances()
        result: dict[str, Any] = {"venue": "bybit", "is_demo": self._config.is_demo}
        for b in bal.balances:
            result[f"{b.asset}_free"] = float(b.free)
            result[f"{b.asset}_locked"] = float(b.locked)
            result[f"{b.asset}_total"] = float(b.total)
        return result
