# execution/adapters/binance/adapter.py
"""Binance Futures venue adapter -- USDT-M perpetual futures via REST API.

Implements the same interface as BybitAdapter for Binance USDT-M Futures.
Supports testnet (testnet.binancefuture.com) and production (fapi.binance.com).

Usage:
    from execution.adapters.binance import BinanceAdapter, BinanceConfig

    config = BinanceConfig(api_key="...", api_secret="...", testnet=True)
    adapter = BinanceAdapter(config)
    adapter.connect()

    print(adapter.get_balances())
    print(adapter.get_positions())
    adapter.send_market_order("BTCUSDT", "buy", 0.001)
"""
from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Optional, Tuple

from execution.adapters.binance.config import BinanceConfig
from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
from execution.adapters.binance.rate_limit_policy import make_rate_limit_policy
from execution.models.balances import BalanceSnapshot, CanonicalBalance
from execution.models.fills import CanonicalFill
from execution.models.positions import VenuePosition

logger = logging.getLogger(__name__)

# Binance Futures interval mapping (our internal -> Binance API)
_INTERVAL_MAP = {
    "1": "1m",
    "5": "5m",
    "15": "15m",
    "60": "1h",
    "240": "4h",
    "D": "1d",
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


class BinanceAdapter:
    """Binance Futures venue adapter implementing the same interface as BybitAdapter."""

    venue: str = "binance"

    def __init__(self, config: BinanceConfig) -> None:
        self._config = config
        rest_cfg = BinanceRestConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            api_secret=config.api_secret,
        )
        self._rate_policy = make_rate_limit_policy()
        self._client = BinanceRestClient(rest_cfg, rate_policy=self._rate_policy)
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Test connection by fetching account balance."""
        try:
            data = self._client.request_signed(
                method="GET",
                path="/fapi/v2/account",
            )
            # Binance returns a dict with "assets" and "positions" on success.
            # On error it returns {"code": ..., "msg": ...}
            if "code" in data and data["code"] != 200:
                logger.error(
                    "Binance connection failed: code=%s msg=%s",
                    data.get("code"), data.get("msg"),
                )
                return False
            self._connected = True
            # Count non-zero balances
            assets = data.get("assets", [])
            nonzero = [
                a for a in assets
                if float(a.get("walletBalance", "0")) > 0
            ]
            logger.info(
                "Binance connected: %s, %d assets with balance",
                self._config.base_url, len(nonzero),
            )
            return True
        except Exception as e:
            logger.error("Binance connection failed: %s", e)
            return False

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    def get_balances(self) -> BalanceSnapshot:
        """Get current account balances."""
        try:
            data = self._client.request_signed(
                method="GET",
                path="/fapi/v2/account",
            )
            if "code" in data and data["code"] != 200:
                return BalanceSnapshot(venue="binance", balances=(), ts_ms=0)
            assets = data.get("assets", [])
            ts_ms = int(time.time() * 1000)
            balances = []
            for a in assets:
                wallet = Decimal(str(a.get("walletBalance", "0")))
                if wallet == 0:
                    continue
                available = Decimal(str(a.get("availableBalance", "0")))
                locked = wallet - available
                if locked < 0:
                    locked = Decimal("0")
                balances.append(
                    CanonicalBalance.from_free_locked(
                        venue="binance",
                        asset=a.get("asset", ""),
                        free=available,
                        locked=locked,
                        ts_ms=ts_ms,
                    )
                )
            return BalanceSnapshot(
                venue="binance",
                balances=tuple(balances),
                ts_ms=ts_ms,
            )
        except Exception as e:
            logger.error("get_balances failed: %s", e)
            return BalanceSnapshot(venue="binance", balances=(), ts_ms=0)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self, symbol: str = "") -> Tuple[VenuePosition, ...]:
        """Get open positions."""
        try:
            params: dict[str, Any] = {}
            if symbol:
                params["symbol"] = symbol
            data = self._client.request_signed(
                method="GET",
                path="/fapi/v2/positionRisk",
                params=params,
            )
            # Error response
            if isinstance(data, dict) and "code" in data:
                logger.error("get_positions error: %s", data.get("msg"))
                return ()
            # data is a list of position dicts
            if not isinstance(data, list):
                return ()
            positions = []
            for p in data:
                amt = float(p.get("positionAmt", "0"))
                if amt == 0:
                    continue
                positions.append(VenuePosition(
                    venue="binance",
                    symbol=p.get("symbol", ""),
                    qty=Decimal(str(amt)),
                    entry_price=Decimal(str(p.get("entryPrice", "0"))),
                    mark_price=Decimal(str(p.get("markPrice", "0"))),
                    liquidation_price=Decimal(str(p.get("liquidationPrice", "0"))),
                    unrealized_pnl=Decimal(str(p.get("unRealizedProfit", "0"))),
                    leverage=int(p.get("leverage", "1")),
                    margin_type=p.get("marginType", "").lower(),
                    ts_ms=int(time.time() * 1000),
                ))
            return tuple(positions)
        except Exception as e:
            logger.error("get_positions failed: %s", e)
            return ()

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def send_market_order(
        self, symbol: str, side: str, qty: float,
        *, reduce_only: bool = False,
    ) -> dict:
        """Send a market order."""
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": "BUY" if side.lower() == "buy" else "SELL",
            "type": "MARKET",
            "quantity": str(qty),
        }
        if reduce_only:
            params["reduceOnly"] = True
        try:
            data = self._client.request_signed(
                method="POST",
                path="/fapi/v1/order",
                params=params,
            )
            if "code" in data and data["code"] != 200:
                return {
                    "status": "error",
                    "code": data.get("code"),
                    "msg": data.get("msg"),
                }
            order_id = str(data.get("orderId", ""))
            logger.info(
                "Binance %s %s %.4f orderId=%s",
                side, symbol, qty, order_id,
            )
            return {
                "orderId": order_id,
                "clientOrderId": data.get("clientOrderId", ""),
                "status": "submitted",
            }
        except Exception as e:
            logger.error("send_market_order failed: %s", e)
            return {"status": "error", "msg": str(e)}

    def send_limit_order(
        self, symbol: str, side: str, qty: float, price: float,
        *, tif: str = "GTC", reduce_only: bool = False, post_only: bool = False,
    ) -> dict:
        """Send a limit order."""
        time_in_force = "GTX" if post_only else tif.upper()
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": "BUY" if side.lower() == "buy" else "SELL",
            "type": "LIMIT",
            "quantity": str(qty),
            "price": str(price),
            "timeInForce": time_in_force,
        }
        if reduce_only:
            params["reduceOnly"] = True
        try:
            data = self._client.request_signed(
                method="POST",
                path="/fapi/v1/order",
                params=params,
            )
            if "code" in data and data["code"] != 200:
                return {
                    "status": "error",
                    "code": data.get("code"),
                    "msg": data.get("msg"),
                }
            return {
                "orderId": str(data.get("orderId", "")),
                "status": "submitted",
            }
        except Exception as e:
            logger.error("send_limit_order failed: %s", e)
            return {"status": "error", "msg": str(e)}

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an order."""
        try:
            data = self._client.request_signed(
                method="DELETE",
                path="/fapi/v1/order",
                params={"symbol": symbol, "orderId": order_id},
            )
            if "code" in data and data["code"] != 200:
                return {
                    "status": "error",
                    "code": data.get("code"),
                    "msg": data.get("msg"),
                }
            return {"status": "canceled"}
        except Exception as e:
            logger.error("cancel_order failed: %s", e)
            return {"status": "error", "msg": str(e)}

    def cancel_all(self, symbol: str = "") -> dict:
        """Cancel all open orders for a symbol."""
        if not symbol:
            logger.warning("Binance cancel_all requires a symbol")
            return {"status": "error", "msg": "symbol required"}
        try:
            data = self._client.request_signed(
                method="DELETE",
                path="/fapi/v1/allOpenOrders",
                params={"symbol": symbol},
            )
            if isinstance(data, dict) and "code" in data and data["code"] != 200:
                return {"status": "error", "code": data.get("code")}
            return {"status": "canceled"}
        except Exception as e:
            logger.error("cancel_all failed: %s", e)
            return {"status": "error", "msg": str(e)}

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
    # Fills / trades
    # ------------------------------------------------------------------

    def get_recent_fills(
        self, *, symbol: Optional[str] = None, since_ms: int = 0,
    ) -> Tuple[CanonicalFill, ...]:
        """Get recent trade history (fills)."""
        try:
            params: dict[str, Any] = {}
            if symbol:
                params["symbol"] = symbol
            if since_ms:
                params["startTime"] = since_ms
            params["limit"] = 50
            data = self._client.request_signed(
                method="GET",
                path="/fapi/v1/userTrades",
                params=params,
            )
            if isinstance(data, dict) and "code" in data:
                return ()
            if not isinstance(data, list):
                return ()
            fills = []
            for f in data:
                fills.append(CanonicalFill(
                    venue="binance",
                    symbol=f.get("symbol", ""),
                    order_id=str(f.get("orderId", "")),
                    trade_id=str(f.get("id", "")),
                    fill_id=str(f.get("id", "")),
                    side="buy" if f.get("side", "").upper() == "BUY" else "sell",
                    qty=Decimal(str(f.get("qty", "0"))),
                    price=Decimal(str(f.get("price", "0"))),
                    fee=Decimal(str(f.get("commission", "0"))),
                    fee_asset=f.get("commissionAsset"),
                    liquidity="maker" if f.get("maker") else "taker",
                    ts_ms=int(f.get("time", 0)),
                ))
            return tuple(fills)
        except Exception as e:
            logger.error("get_recent_fills failed: %s", e)
            return ()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> dict:
        """Get latest ticker (last price, bid, ask, volume)."""
        try:
            # Use bookTicker for bid/ask
            book = self._client.request_public(
                method="GET",
                path="/fapi/v1/ticker/bookTicker",
                params={"symbol": symbol},
            )
            # Use 24hr ticker for price/volume
            price_data = self._client.request_public(
                method="GET",
                path="/fapi/v1/ticker/24hr",
                params={"symbol": symbol},
            )
            last_price = float(price_data.get("lastPrice", 0))
            bid = float(book.get("bidPrice", 0))
            ask = float(book.get("askPrice", 0))
            return {
                "symbol": symbol,
                "lastPrice": last_price,
                "bid1Price": bid,
                "ask1Price": ask,
                "volume24h": float(price_data.get("volume", 0)),
                "turnover24h": float(price_data.get("quoteVolume", 0)),
                "fundingRate": float(price_data.get("lastFundingRate", 0)),
            }
        except Exception as e:
            logger.error("get_ticker failed: %s", e)
            return {}

    def get_klines(
        self, symbol: str, interval: str = "60", limit: int = 200,
    ) -> list[dict]:
        """Get historical klines.

        interval: "1"=1m, "5"=5m, "60"=1h, "240"=4h, "D"=1d
        (also accepts Binance native: "1m", "1h", etc.)

        Paginates automatically when limit > 1000 (Binance API max per request).
        """
        _API_MAX = 1000
        bn_interval = _INTERVAL_MAP.get(interval, interval)
        all_bars: list[dict] = []
        end_ts: int | None = None

        remaining = limit
        while remaining > 0:
            batch = min(remaining, _API_MAX)
            params: dict[str, Any] = {
                "symbol": symbol,
                "interval": bn_interval,
                "limit": batch,
            }
            if end_ts is not None:
                params["endTime"] = end_ts

            try:
                data = self._client.request_public(
                    method="GET",
                    path="/fapi/v1/klines",
                    params=params,
                )
            except Exception as e:
                logger.error("get_klines failed: %s", e)
                break

            if isinstance(data, dict) and "code" in data:
                break
            if not isinstance(data, list) or not data:
                break

            # Binance kline: [open_time, o, h, l, c, vol, close_time, ...]
            bars = [
                {
                    "time": int(k[0]) // 1000,
                    "start": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
                for k in data
            ]
            all_bars.extend(bars)
            remaining -= len(bars)

            if len(data) < batch:
                break

            # Binance returns oldest first; oldest item is first.
            # Next page ends before the oldest we already have.
            oldest_ts = int(data[0][0])
            end_ts = oldest_ts - 1

        # Deduplicate by start timestamp
        seen: set[int] = set()
        unique: list[dict] = []
        for b in all_bars:
            ts = b.get("start", b["time"] * 1000)
            if ts not in seen:
                seen.add(ts)
                unique.append(b)

        return unique

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_info(self) -> dict:
        """Get detailed account information."""
        bal = self.get_balances()
        result: dict[str, Any] = {
            "venue": "binance",
            "is_demo": self._config.is_demo,
        }
        for b in bal.balances:
            result[f"{b.asset}_free"] = float(b.free)
            result[f"{b.asset}_locked"] = float(b.locked)
            result[f"{b.asset}_total"] = float(b.total)
        return result
