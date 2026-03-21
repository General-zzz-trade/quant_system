# execution/adapters/hyperliquid/adapter.py
"""Hyperliquid venue adapter — perpetual futures via REST API.

Implements VenueAdapter protocol for Hyperliquid perpetuals.
Info (read) endpoints are public; exchange (write) endpoints require EIP-712 signing.

Usage:
    from execution.adapters.hyperliquid import HyperliquidAdapter, HyperliquidConfig

    # Read-only (no private key needed):
    config = HyperliquidConfig.mainnet()
    adapter = HyperliquidAdapter(config)
    print(adapter.list_instruments())

    # Full access:
    config = HyperliquidConfig.mainnet(private_key="0x...")
    adapter = HyperliquidAdapter(config)
    adapter.connect()
    print(adapter.get_balances())
    adapter.send_market_order("BTC", "buy", 0.001)
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from execution.adapters.hyperliquid.client import HyperliquidRestClient
from execution.adapters.hyperliquid.config import HyperliquidConfig
from execution.adapters.hyperliquid.mapper import (
    coin_to_symbol,
    map_balance,
    map_fill,
    map_instrument,
    map_order,
    map_orderbook,
    map_position,
    normalize_coin,
)
from execution.models.balances import BalanceSnapshot
from execution.models.fills import CanonicalFill
from execution.models.instruments import InstrumentInfo
from execution.models.orders import CanonicalOrder
from execution.models.positions import VenuePosition

logger = logging.getLogger(__name__)


class HyperliquidAdapter:
    """Hyperliquid venue adapter implementing VenueAdapter protocol.

    Read operations (list_instruments, get_ticker, get_orderbook) work without
    authentication. Write operations (send_market_order, send_limit_order, etc.)
    require a private key and the eth_account library.
    """

    venue: str = "hyperliquid"

    def __init__(self, config: HyperliquidConfig) -> None:
        self._config = config
        self._client = HyperliquidRestClient(config)
        self._connected = False
        # Cache: coin name -> asset index (e.g. "BTC" -> 0, "ETH" -> 1)
        self._asset_index: dict[str, int] = {}
        # Cache: coin name -> universe entry
        self._universe: list[dict] = []
        # Cache: asset contexts
        self._asset_ctxs: list[dict] = []

    # ------------------------------------------------------------------
    # Connection & meta cache
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialize adapter by fetching metadata and optionally testing auth.

        Always fetches the asset universe (public). If a private key is configured,
        also tests account state.
        """
        if not self._refresh_meta():
            return False

        if self._config.has_private_key and self._config.wallet_address:
            state = self._client.info_request({
                "type": "clearinghouseState",
                "user": self._config.wallet_address,
            })
            if isinstance(state, dict) and "marginSummary" in state:
                acct_value = state.get("marginSummary", {}).get("accountValue", "0")
                logger.info(
                    "Hyperliquid connected: %s, account_value=%s, %d assets",
                    self._config.base_url, acct_value, len(self._universe),
                )
                self._connected = True
                return True
            elif isinstance(state, dict) and "crossMarginSummary" in state:
                acct_value = state.get("crossMarginSummary", {}).get("accountValue", "0")
                logger.info(
                    "Hyperliquid connected: %s, account_value=%s, %d assets",
                    self._config.base_url, acct_value, len(self._universe),
                )
                self._connected = True
                return True
            else:
                logger.error("Hyperliquid auth check failed: %s", state)
                return False
        else:
            logger.info(
                "Hyperliquid connected (read-only): %s, %d assets",
                self._config.base_url, len(self._universe),
            )
            self._connected = True
            return True

    def is_connected(self) -> bool:
        return self._connected

    def _refresh_meta(self) -> bool:
        """Fetch and cache asset universe + contexts."""
        data = self._client.info_request({"type": "metaAndAssetCtxs"})
        if not isinstance(data, list) or len(data) < 2:
            # Fallback: try just meta
            meta = self._client.info_request({"type": "meta"})
            if isinstance(meta, dict) and "universe" in meta:
                self._universe = meta["universe"]
                self._asset_ctxs = []
            else:
                logger.error("Failed to fetch Hyperliquid metadata: %s", data)
                return False
        else:
            meta = data[0]
            self._universe = meta.get("universe", [])
            self._asset_ctxs = data[1] if len(data) > 1 else []

        # Build asset index
        self._asset_index = {}
        for i, asset in enumerate(self._universe):
            name = asset.get("name", "")
            self._asset_index[name.upper()] = i
        return True

    def _get_asset_index(self, coin_or_symbol: str) -> int:
        """Get the integer asset index for a coin.

        Accepts: "BTC", "BTCUSDT", "ETH", "ETHUSDT"
        Returns: integer index.
        Raises: ValueError if coin not found.
        """
        coin = normalize_coin(coin_or_symbol)
        if coin in self._asset_index:
            return self._asset_index[coin]
        # Refresh cache and retry
        self._refresh_meta()
        if coin in self._asset_index:
            return self._asset_index[coin]
        raise ValueError(f"Unknown coin: {coin_or_symbol} (normalized: {coin})")

    # ------------------------------------------------------------------
    # VenueAdapter protocol
    # ------------------------------------------------------------------

    def list_instruments(self, symbols: list[str] | None = None) -> Tuple[InstrumentInfo, ...]:
        """Get instrument metadata for all perpetuals."""
        if not self._universe:
            self._refresh_meta()

        instruments = []
        for i, asset in enumerate(self._universe):
            ctx = self._asset_ctxs[i] if i < len(self._asset_ctxs) else None
            instruments.append(map_instrument(asset, i, ctx))

        if symbols:
            normalized = {normalize_coin(s) for s in symbols}
            instruments = [
                inst for inst in instruments
                if inst.base_asset.upper() in normalized
                or inst.symbol in {s.upper() for s in symbols}
            ]

        return tuple(instruments)

    def get_balances(self) -> BalanceSnapshot:
        """Get current account balances."""
        if not self._config.wallet_address:
            return BalanceSnapshot(venue="hyperliquid", balances=(), ts_ms=0)
        state = self._client.info_request({
            "type": "clearinghouseState",
            "user": self._config.wallet_address,
        })
        if isinstance(state, dict) and ("marginSummary" in state or "crossMarginSummary" in state):
            return map_balance(state)
        return BalanceSnapshot(venue="hyperliquid", balances=(), ts_ms=0)

    def get_positions(self, symbol: str = "") -> Tuple[VenuePosition, ...]:
        """Get open positions."""
        if not self._config.wallet_address:
            return ()
        state = self._client.info_request({
            "type": "clearinghouseState",
            "user": self._config.wallet_address,
        })
        if not isinstance(state, dict):
            return ()

        asset_positions = state.get("assetPositions", [])
        positions = []
        for ap in asset_positions:
            pos = ap.get("position", ap)
            szi = pos.get("szi", "0")
            if float(szi) == 0:
                continue
            vp = map_position(pos)
            positions.append(vp)

        if symbol:
            coin = normalize_coin(symbol)
            positions = [
                p for p in positions
                if normalize_coin(p.symbol) == coin
            ]

        return tuple(positions)

    def get_open_orders(
        self, *, symbol: Optional[str] = None,
    ) -> Tuple[CanonicalOrder, ...]:
        """Get active (unfilled) orders."""
        if not self._config.wallet_address:
            return ()
        data = self._client.info_request({
            "type": "openOrders",
            "user": self._config.wallet_address,
        })
        if not isinstance(data, list):
            return ()

        orders = [map_order(o) for o in data]

        if symbol:
            coin = normalize_coin(symbol)
            orders = [o for o in orders if normalize_coin(o.symbol) == coin]

        return tuple(orders)

    def get_recent_fills(
        self, *, symbol: Optional[str] = None, since_ms: int = 0,
    ) -> Tuple[CanonicalFill, ...]:
        """Get recent execution history."""
        if not self._config.wallet_address:
            return ()
        data = self._client.info_request({
            "type": "userFills",
            "user": self._config.wallet_address,
        })
        if not isinstance(data, list):
            return ()

        fills = [map_fill(f) for f in data]

        if symbol:
            coin = normalize_coin(symbol)
            fills = [f for f in fills if normalize_coin(f.symbol) == coin]

        if since_ms > 0:
            fills = [f for f in fills if f.ts_ms >= since_ms]

        return tuple(fills)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def send_market_order(
        self, symbol: str, side: str, qty: float,
        *, reduce_only: bool = False,
    ) -> dict:
        """Send a market order.

        Hyperliquid doesn't have a native market order type. We simulate it
        using a limit order with aggressive price (slippage tolerance).
        """
        coin = normalize_coin(symbol)
        asset_idx = self._get_asset_index(coin)
        is_buy = side.lower() == "buy"

        # Get current price for aggressive limit
        ticker = self.get_ticker(symbol)
        if not ticker:
            return {"status": "error", "msg": f"cannot get ticker for {symbol}"}

        # Use 1% slippage for market-like execution
        mid = ticker.get("lastPrice", 0)
        if mid <= 0:
            return {"status": "error", "msg": "invalid last price"}

        slippage = 0.01
        price = mid * (1 + slippage) if is_buy else mid * (1 - slippage)

        # Round price to reasonable precision
        if mid > 1000:
            price = round(price, 1)
        elif mid > 1:
            price = round(price, 4)
        else:
            price = round(price, 6)

        action = {
            "type": "order",
            "orders": [{
                "a": asset_idx,
                "b": is_buy,
                "p": str(price),
                "s": str(qty),
                "r": reduce_only,
                "t": {"limit": {"tif": "Ioc"}},  # IOC for market-like behavior
            }],
            "grouping": "na",
        }

        try:
            data = self._client.exchange_request(action)
        except (NotImplementedError, ValueError) as e:
            return {"status": "error", "msg": str(e)}

        return self._parse_order_response(data, coin, side, qty)

    def send_limit_order(
        self, symbol: str, side: str, qty: float, price: float,
        *, tif: str = "Gtc", reduce_only: bool = False,
    ) -> dict:
        """Send a limit order."""
        coin = normalize_coin(symbol)
        asset_idx = self._get_asset_index(coin)
        is_buy = side.lower() == "buy"

        action = {
            "type": "order",
            "orders": [{
                "a": asset_idx,
                "b": is_buy,
                "p": str(price),
                "s": str(qty),
                "r": reduce_only,
                "t": {"limit": {"tif": tif}},
            }],
            "grouping": "na",
        }

        try:
            data = self._client.exchange_request(action)
        except (NotImplementedError, ValueError) as e:
            return {"status": "error", "msg": str(e)}

        return self._parse_order_response(data, coin, side, qty)

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an order by ID."""
        coin = normalize_coin(symbol)
        asset_idx = self._get_asset_index(coin)

        action = {
            "type": "cancel",
            "cancels": [{"a": asset_idx, "o": int(order_id)}],
        }

        try:
            data = self._client.exchange_request(action)
        except (NotImplementedError, ValueError) as e:
            return {"status": "error", "msg": str(e)}

        if isinstance(data, dict) and data.get("status") == "ok":
            return {"status": "canceled"}
        return {"status": "error", "data": data}

    def cancel_all(self, symbol: str = "") -> dict:
        """Cancel all open orders, optionally for a specific symbol."""
        orders = self.get_open_orders(symbol=symbol or None)
        if not orders:
            return {"status": "ok", "canceled": 0}

        # Group cancels by asset
        cancels = []
        for order in orders:
            coin = normalize_coin(order.symbol)
            try:
                asset_idx = self._get_asset_index(coin)
                cancels.append({"a": asset_idx, "o": int(order.order_id)})
            except (ValueError, TypeError):
                logger.warning("Cannot cancel order %s: unknown asset", order.order_id)

        if not cancels:
            return {"status": "ok", "canceled": 0}

        action = {"type": "cancel", "cancels": cancels}

        try:
            data = self._client.exchange_request(action)
        except (NotImplementedError, ValueError) as e:
            return {"status": "error", "msg": str(e)}

        return {"status": "ok", "canceled": len(cancels), "data": data}

    def close_position(self, symbol: str) -> dict:
        """Close an open position with a market order."""
        positions = self.get_positions(symbol=symbol)
        for pos in positions:
            if normalize_coin(pos.symbol) == normalize_coin(symbol) and not pos.is_flat:
                side = "sell" if pos.is_long else "buy"
                qty = float(pos.abs_qty)
                return self.send_market_order(
                    symbol, side, qty, reduce_only=True,
                )
        return {"status": "no_position"}

    def set_leverage(self, symbol: str, leverage: int, *, is_cross: bool = True) -> dict:
        """Set leverage for a symbol."""
        asset_idx = self._get_asset_index(symbol)

        action = {
            "type": "updateLeverage",
            "asset": asset_idx,
            "isCross": is_cross,
            "leverage": leverage,
        }

        try:
            data = self._client.exchange_request(action)
        except (NotImplementedError, ValueError) as e:
            return {"status": "error", "msg": str(e)}

        if isinstance(data, dict) and data.get("status") == "ok":
            return {"status": "ok", "leverage": leverage}
        return {"status": "error", "data": data}

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> dict:
        """Get latest ticker (mid price, bid, ask, funding rate)."""
        coin = normalize_coin(symbol)

        # Get L2 book for bid/ask
        book_data = self._client.info_request({"type": "l2Book", "coin": coin})
        if not isinstance(book_data, dict) or "levels" not in book_data:
            return {}

        book = map_orderbook(book_data)
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else best_bid or best_ask

        # Get funding rate from asset contexts
        funding_rate = 0.0
        if not self._asset_ctxs:
            self._refresh_meta()
        if coin.upper() in self._asset_index:
            idx = self._asset_index[coin.upper()]
            if idx < len(self._asset_ctxs):
                ctx = self._asset_ctxs[idx]
                funding_rate = float(ctx.get("funding", 0))

        return {
            "symbol": coin_to_symbol(coin),
            "lastPrice": mid,
            "bid1Price": best_bid,
            "ask1Price": best_ask,
            "volume24h": 0.0,  # Not directly available from L2 book
            "fundingRate": funding_rate,
        }

    def get_orderbook(self, symbol: str) -> dict:
        """Get L2 orderbook.

        Returns:
            Dict with "bids", "asks" (list of [price, size, n_orders]),
            "coin", "ts_ms".
        """
        coin = normalize_coin(symbol)
        data = self._client.info_request({"type": "l2Book", "coin": coin})
        if not isinstance(data, dict) or "levels" not in data:
            return {"bids": [], "asks": [], "coin": coin, "ts_ms": 0}
        return map_orderbook(data)

    def get_funding_rates(self) -> list[dict]:
        """Get current funding rates for all assets."""
        if not self._asset_ctxs:
            self._refresh_meta()

        rates = []
        for i, ctx in enumerate(self._asset_ctxs):
            if i < len(self._universe):
                coin = self._universe[i].get("name", "")
                rates.append({
                    "symbol": coin_to_symbol(coin),
                    "fundingRate": float(ctx.get("funding", 0)),
                    "markPx": float(ctx.get("markPx", 0)),
                    "openInterest": float(ctx.get("openInterest", 0)),
                    "volume24h": float(ctx.get("dayNtlVlm", 0)),
                })
        return rates

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_info(self) -> dict:
        """Get detailed account information."""
        bal = self.get_balances()
        result: dict[str, Any] = {
            "venue": "hyperliquid",
            "is_testnet": self._config.is_testnet,
        }
        for b in bal.balances:
            result[f"{b.asset}_free"] = float(b.free)
            result[f"{b.asset}_locked"] = float(b.locked)
            result[f"{b.asset}_total"] = float(b.total)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_order_response(
        self, data: Any, coin: str, side: str, qty: float,
    ) -> dict:
        """Parse exchange response for order submission."""
        if isinstance(data, dict):
            if data.get("status") == "ok":
                response = data.get("response", {})
                if response.get("type") == "order":
                    statuses = response.get("data", {}).get("statuses", [])
                    if statuses:
                        s = statuses[0]
                        if "resting" in s:
                            order_info = s["resting"]
                            return {
                                "orderId": str(order_info.get("oid", "")),
                                "status": "submitted",
                            }
                        elif "filled" in s:
                            fill_info = s["filled"]
                            return {
                                "orderId": str(fill_info.get("oid", "")),
                                "status": "filled",
                                "avgPrice": fill_info.get("avgPx"),
                                "filledQty": fill_info.get("totalSz"),
                            }
                        elif "error" in s:
                            return {"status": "error", "msg": s["error"]}
                logger.info(
                    "Hyperliquid %s %s %.6f submitted", side, coin, qty,
                )
                return {"status": "submitted", "data": data}
            elif data.get("status") == "error":
                return {"status": "error", "msg": data.get("msg", str(data))}
        return {"status": "error", "msg": str(data)}
