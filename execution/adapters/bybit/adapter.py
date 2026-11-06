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
import threading as _threading
import time as _time
from decimal import Decimal, InvalidOperation
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

_ORDER_LINK_ID_LOCK = _threading.Lock()
_ORDER_LINK_ID_SEQ = 0
_ORDER_LINK_ID_SEQ_MOD = 36 ** 4


def _to_base36(value: int) -> str:
    if value < 0:
        raise ValueError("base36 input must be non-negative")
    if value == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    parts = []
    while value:
        value, rem = divmod(value, 36)
        parts.append(digits[rem])
    return "".join(reversed(parts))


def _next_order_link_suffix() -> str:
    global _ORDER_LINK_ID_SEQ
    with _ORDER_LINK_ID_LOCK:
        _ORDER_LINK_ID_SEQ = (_ORDER_LINK_ID_SEQ + 1) % _ORDER_LINK_ID_SEQ_MOD
        return _to_base36(_ORDER_LINK_ID_SEQ).rjust(4, "0")


def _make_order_link_id(symbol: str, side: str) -> str:
    """Generate unique orderLinkId for Bybit dedup (max 36 chars)."""
    symbol_token = str(symbol).upper()
    if not symbol_token:
        raise ValueError("symbol is required for orderLinkId")
    side_token = (str(side).strip().lower()[:1] or "u")
    ts_token = _to_base36(int(_time.time() * 1000))
    suffix = _next_order_link_suffix()
    max_symbol_len = 36 - len(f"qs__{side_token}_{ts_token}_{suffix}")
    if max_symbol_len <= 0:
        raise ValueError("orderLinkId budget exhausted by timestamp/suffix layout")
    symbol_token = symbol_token[:max_symbol_len]
    return f"qs_{symbol_token}_{side_token}_{ts_token}_{suffix}"


_ORDER_TTL_SECONDS = 30
_MAX_QUERY_PAGES = 100
_CONFIRMABLE_ORDER_STATUSES = {
    "New", "PartiallyFilled", "Filled", "Cancelled", "PartiallyFilledCanceled",
    "Rejected", "Deactivated", "Untriggered", "Triggered",
}


class BybitAdapter:
    """Bybit venue adapter implementing VenueAdapter protocol."""

    venue: str = "bybit"

    def __init__(self, config: BybitConfig) -> None:
        self._config = config
        self._client = BybitRestClient(config)
        self._connected = False
        self._pending_orders: dict[str, float] = {}  # orderLinkId -> submit_time
        self._pending_lock = _threading.Lock()

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

    def get_account_equity(self) -> float:
        """Unified account equity including unrealized PnL; fail on absent data."""
        data = self._client.get("/v5/account/wallet-balance", {"accountType": self._config.account_type})
        if data.get("retCode") != 0:
            raise RuntimeError("Bybit equity request failed")
        return float(data["result"]["list"][0]["totalEquity"])

    def get_balances(self) -> BalanceSnapshot:
        """Get current account balances."""
        data = self._client.get("/v5/account/wallet-balance",
                                {"accountType": self._config.account_type})
        if data.get("retCode") != 0:
            return BalanceSnapshot(venue="bybit", balances=(), ts_ms=0)
        return map_balance(data.get("result", {}))

    def _query_pages(
        self, path: str, params: dict[str, str], *, legacy_empty_on_error: bool = False,
    ) -> list[dict]:
        """Read a complete REST snapshot or raise; never return a partial page set."""
        request_params = dict(params)
        seen_cursors: set[str] = set()
        rows: list[dict] = []
        for page_index in range(_MAX_QUERY_PAGES):
            data = self._client.get(path, dict(request_params))
            if not isinstance(data, dict) or data.get("retCode") != 0:
                if legacy_empty_on_error and page_index == 0 and isinstance(data, dict):
                    return []
                raise RuntimeError(f"Bybit query failed: {path}")
            result = data.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("list"), list):
                raise RuntimeError(f"Malformed Bybit query response: {path}")
            items = result["list"]
            if any(not isinstance(item, dict) for item in items):
                raise RuntimeError(f"Malformed Bybit query item: {path}")
            rows.extend(items)
            cursor = result.get("nextPageCursor", "")
            if not isinstance(cursor, str):
                raise RuntimeError(f"Malformed Bybit pagination cursor: {path}")
            if not cursor:
                return rows
            if cursor in seen_cursors:
                raise RuntimeError(f"Repeated Bybit pagination cursor: {path}")
            seen_cursors.add(cursor)
            request_params["cursor"] = cursor
        raise RuntimeError(f"Incomplete Bybit pagination: {path}")

    @staticmethod
    def _query_number(item: dict, name: str, *, positive: bool = False) -> Decimal:
        try:
            value = Decimal(str(item[name]))
        except (KeyError, InvalidOperation, ValueError) as exc:
            raise RuntimeError(f"Malformed Bybit {name}") from exc
        if not value.is_finite() or value < 0 or (positive and value == 0):
            raise RuntimeError(f"Invalid Bybit {name}")
        return value

    @staticmethod
    def _query_symbol(item: dict, symbol: Optional[str]) -> None:
        actual = item.get("symbol")
        if not isinstance(actual, str) or not actual or (symbol and actual != symbol):
            raise RuntimeError("Bybit query symbol mismatch or missing symbol")

    def _confirmed_order(self, item: dict, symbol: Optional[str]) -> CanonicalOrder:
        self._query_symbol(item, symbol)
        if not isinstance(item.get("orderId"), str) or not item["orderId"]:
            raise RuntimeError("Bybit order missing orderId")
        if item.get("orderLinkId") is not None and not isinstance(item["orderLinkId"], str):
            raise RuntimeError("Bybit order has invalid orderLinkId")
        if item.get("side") not in ("Buy", "Sell"):
            raise RuntimeError("Bybit order has invalid side")
        if item.get("orderStatus") not in _CONFIRMABLE_ORDER_STATUSES:
            raise RuntimeError("Bybit order has unknown status")
        if item.get("orderType") not in ("Market", "Limit"):
            raise RuntimeError("Bybit order has invalid orderType")
        qty = self._query_number(item, "qty", positive=True)
        filled_qty = self._query_number(item, "cumExecQty")
        if filled_qty > qty:
            raise RuntimeError("Bybit order filled quantity exceeds order quantity")
        order = map_order(item)
        for value in (order.price, order.avg_price):
            if value is not None and (not value.is_finite() or value < 0):
                raise RuntimeError("Bybit order has invalid price")
        return order

    def get_positions(
        self, symbol: str = "", *, strict: bool = False,
    ) -> Tuple[VenuePosition, ...]:
        """Get open positions; strict snapshots raise on incomplete account data."""
        params: dict[str, str] = {"category": self._config.category}
        if symbol:
            params["symbol"] = symbol
        else:
            params["settleCoin"] = "USDT"  # required when no symbol specified
        if strict:
            items = self._query_pages("/v5/position/list", params)
            positions = []
            for item in items:
                self._query_symbol(item, symbol)
                size = self._query_number(item, "size")
                if item.get("side") not in ("Buy", "Sell", "") or (
                    size and item.get("side") not in ("Buy", "Sell")
                ):
                    raise RuntimeError("Bybit position has invalid side")
                if size:
                    position = map_position(item)
                    if any(value is not None and not value.is_finite() for value in (
                        position.entry_price, position.mark_price, position.unrealized_pnl,
                    )):
                        raise RuntimeError("Bybit position has invalid numeric values")
                    positions.append(position)
            return tuple(positions)
        data = self._client.get("/v5/position/list", params)
        if data.get("retCode") != 0:
            return ()
        items = data.get("result", {}).get("list", [])
        return tuple(
            map_position(p) for p in items
            if float(p.get("size", "0")) != 0
        )

    def get_open_orders(
        self, *, symbol: Optional[str] = None, strict: bool = False,
    ) -> Tuple[CanonicalOrder, ...]:
        """Get active (unfilled) orders."""
        params: dict[str, str] = {"category": self._config.category}
        if symbol:
            params["symbol"] = symbol
        elif strict:
            params["settleCoin"] = "USDT"
        if strict:
            return tuple(
                self._confirmed_order(item, symbol)
                for item in self._query_pages("/v5/order/realtime", params)
            )
        data = self._client.get("/v5/order/realtime", params)
        if data.get("retCode") != 0:
            return ()
        items = data.get("result", {}).get("list", [])
        return tuple(map_order(o) for o in items)

    def get_order(
        self, symbol: str, order_id: str = "", client_order_id: str = "",
    ) -> CanonicalOrder | None:
        """Confirm one identity via realtime then history; failures stay unknown.

        An empty successful lookup in both endpoints returns None. Callers must
        not treat that as proof that a timed-out submission was never accepted.
        """
        if not symbol or not (order_id or client_order_id):
            raise ValueError("symbol and an order identity are required")
        params = {"category": self._config.category, "symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["orderLinkId"] = client_order_id
        for path in ("/v5/order/realtime", "/v5/order/history"):
            items = self._query_pages(path, params)
            if not items:
                continue
            if len(items) != 1:
                raise RuntimeError("Bybit order lookup returned multiple identities")
            item = items[0]
            if ((order_id and item.get("orderId") != order_id)
                    or (client_order_id and item.get("orderLinkId") != client_order_id)):
                raise RuntimeError("Bybit order lookup identity mismatch")
            return self._confirmed_order(item, symbol)
        return None

    def get_recent_fills(
        self, *, symbol: Optional[str] = None, since_ms: int = 0,
        order_id: str = "", limit: int = 100,
    ) -> Tuple[CanonicalFill, ...]:
        """Read all matching Trade executions, deduplicated by exchange execId.

        ``limit`` controls each REST page, not the maximum number of returned
        fills. Any failed or incomplete page set raises instead of returning
        evidence that could incorrectly settle an order. Legacy calls without
        order_id retain their initial API-error empty result; confirmation must
        always provide order_id.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("execution page limit must be between 1 and 100")
        if not isinstance(since_ms, int) or isinstance(since_ms, bool) or since_ms < 0:
            raise ValueError("since_ms must be a non-negative integer")
        params: dict[str, str] = {"category": self._config.category, "limit": str(limit)}
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        if since_ms:
            params["startTime"] = str(since_ms)
        fills: dict[str, CanonicalFill] = {}
        for item in self._query_pages(
            "/v5/execution/list", params, legacy_empty_on_error=not order_id,
        ):
            if item.get("execType") != "Trade":
                continue
            self._query_symbol(item, symbol)
            if not isinstance(item.get("orderId"), str) or not item["orderId"]:
                raise RuntimeError("Bybit execution missing orderId")
            if order_id and item["orderId"] != order_id:
                raise RuntimeError("Bybit execution order identity mismatch")
            exec_id = item.get("execId")
            if not isinstance(exec_id, str) or not exec_id:
                raise RuntimeError("Bybit execution missing execId")
            if item.get("side") not in ("Buy", "Sell"):
                raise RuntimeError("Bybit execution has invalid side")
            timestamp = self._query_number(item, "execTime", positive=True)
            if timestamp != timestamp.to_integral_value():
                raise RuntimeError("Bybit execution has invalid timestamp")
            self._query_number(item, "execQty", positive=True)
            self._query_number(item, "execPrice", positive=True)
            if timestamp < since_ms:
                continue
            if "execFee" not in item or item["execFee"] in (None, ""):
                raise RuntimeError("Bybit execution missing actual fee")
            fill = map_fill(item)
            if not fill.fee.is_finite():
                raise RuntimeError("Bybit execution has invalid fee")
            previous = fills.get(exec_id)
            if previous is not None and (
                previous.payload_digest != fill.payload_digest or previous.fee_asset != fill.fee_asset
            ):
                raise RuntimeError("Bybit execution ID has conflicting payloads")
            fills[exec_id] = fill
        return tuple(fills.values())

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def _check_order_ttls(self) -> list[str]:
        """Cancel orders that have been in SUBMITTED state longer than TTL.

        Returns list of cancelled orderLinkIds.
        """
        now = _time.time()
        cancelled: list[str] = []
        with self._pending_lock:
            stale = [
                oid for oid, ts in self._pending_orders.items()
                if now - ts > _ORDER_TTL_SECONDS
            ]
        for oid in stale:
            logger.warning(
                "Order TTL expired (>%ds): orderLinkId=%s — cancelling",
                _ORDER_TTL_SECONDS, oid,
            )
            # Extract symbol from orderLinkId format: qs_{SYMBOL}_{side}_{ts}_{seq}
            body: dict[str, str] = {
                "category": self._config.category,
                "orderLinkId": oid,
            }
            parts = oid.split("_")
            if len(parts) >= 3:
                body["symbol"] = parts[1].upper()
            data = self._client.post("/v5/order/cancel", body)
            if data.get("retCode") == 0:
                logger.info("Stale order cancelled: %s", oid)
            else:
                logger.error(
                    "Failed to cancel stale order %s: %s",
                    oid, data.get("retMsg"),
                )
            with self._pending_lock:
                self._pending_orders.pop(oid, None)
            cancelled.append(oid)
        return cancelled

    def send_market_order(
        self, symbol: str, side: str, qty: float,
        *, reduce_only: bool = False, client_order_id: Optional[str] = None,
    ) -> dict:
        """Send a market order with orderLinkId for deduplication."""
        # Durable identities belong to the caller's persisted lifecycle. Only
        # legacy submissions participate in this adapter's automatic TTL logic.
        durable = client_order_id is not None
        if durable:
            if not isinstance(client_order_id, str) or not client_order_id or len(client_order_id) > 36:
                raise ValueError("client_order_id must contain 1 to 36 characters")
            order_link_id = client_order_id
        else:
            self._check_order_ttls()
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
            returned_link_id = result.get("orderLinkId", order_link_id)
            if durable:
                if returned_link_id != order_link_id:
                    raise RuntimeError("Bybit acknowledgement orderLinkId mismatch")
                if not isinstance(result.get("orderId"), str) or not result["orderId"]:
                    raise RuntimeError("Bybit acknowledgement missing orderId")
            else:
                with self._pending_lock:
                    self._pending_orders[returned_link_id] = _time.time()
            logger.info(
                "Bybit %s %s %.4f orderId=%s",
                side, symbol, qty, result.get("orderId"),
            )
            return {
                "orderId": result.get("orderId"),
                "orderLinkId": returned_link_id,
                "status": "submitted",
            }
        return {"status": "error", "retCode": data.get("retCode"),
                "retMsg": data.get("retMsg")}

    def send_limit_order(
        self, symbol: str, side: str, qty: float, price: float,
        *, tif: str = "GTC", reduce_only: bool = False, post_only: bool = False,
    ) -> dict:
        """Send a limit order with orderLinkId for deduplication."""
        order_link_id = _make_order_link_id(symbol, side)
        body = {
            "category": self._config.category,
            "symbol": symbol,
            "side": "Buy" if side.lower() == "buy" else "Sell",
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "timeInForce": "PostOnly" if post_only else tif.upper(),
            "orderLinkId": order_link_id,
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
        """Get historical klines. interval: "1"=1m, "5"=5m, "60"=1h, "D"=1d.

        Paginates automatically when limit > 200 (Bybit API max per request).
        """
        _API_MAX = 200
        all_bars: list[dict] = []
        end_ts: int | None = None  # None = fetch latest

        remaining = limit
        while remaining > 0:
            batch = min(remaining, _API_MAX)
            params: dict[str, str] = {
                "category": self._config.category,
                "symbol": symbol,
                "interval": interval,
                "limit": str(batch),
            }
            if end_ts is not None:
                params["end"] = str(end_ts)

            data = self._client.get("/v5/market/kline", params)
            if data.get("retCode") != 0:
                break
            items = data.get("result", {}).get("list", [])
            if not items:
                break

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
                for k in items
            ]
            all_bars.extend(bars)
            remaining -= len(bars)

            if len(items) < batch:
                break  # no more data

            # Bybit returns newest first; oldest item is last. Next page ends before it.
            oldest_ts = int(items[-1][0])
            end_ts = oldest_ts - 1

        # Deduplicate by start timestamp (pagination overlap)
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
        result: dict[str, Any] = {"venue": "bybit", "is_demo": self._config.is_demo}
        for b in bal.balances:
            result[f"{b.asset}_free"] = float(b.free)
            result[f"{b.asset}_locked"] = float(b.locked)
            result[f"{b.asset}_total"] = float(b.total)
        return result
