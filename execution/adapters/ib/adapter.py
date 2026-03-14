# execution/adapters/ib/adapter.py
"""IB venue adapter — connects quant_system to Interactive Brokers.

Implements the VenueAdapter protocol for multi-asset trading via
IB Gateway (headless, Linux native) or TWS.

Usage:
    from execution.adapters.ib import IBAdapter, IBConfig

    config = IBConfig.paper()  # port 4002
    adapter = IBAdapter(config)
    adapter.connect()

    # Register with engine
    registry.register("ib", adapter)

    # Trade any asset class
    adapter.send_market_order("AAPL", "buy", 100)                          # stock
    adapter.send_market_order("EURUSD", "buy", 20000, sec_type="CASH")     # forex
    adapter.send_market_order("ES", "buy", 1, sec_type="FUT", expiry="202412")  # futures
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from execution.adapters.ib.config import IBConfig
from execution.adapters.ib.mapper import (
    make_contract,
    map_account_values,
    map_contract_details,
    map_fill,
    map_position,
    map_trade,
)
from execution.models.balances import BalanceSnapshot
from execution.models.fills import CanonicalFill
from execution.models.instruments import InstrumentInfo
from execution.models.orders import CanonicalOrder
from execution.models.positions import VenuePosition

logger = logging.getLogger(__name__)


class IBAdapter:
    """Interactive Brokers venue adapter implementing VenueAdapter protocol.

    Wraps ib_insync.IB to provide:
    - Multi-asset instrument listing (stocks, forex, futures, options, CFDs, crypto)
    - Account balance and position queries
    - Order placement (market, limit, stop, bracket)
    - Fill/execution history
    - Real-time and historical market data
    """

    venue: str = "ib"

    def __init__(self, config: IBConfig | None = None) -> None:
        self._config = config or IBConfig()
        self._ib: Any = None  # ib_insync.IB instance
        self._connected = False
        self._contract_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to IB Gateway/TWS.

        Returns True if connected successfully.
        """
        from ib_insync import IB

        self._ib = IB()
        try:
            self._ib.connect(
                self._config.host,
                self._config.port,
                clientId=self._config.client_id,
                timeout=self._config.timeout,
                readonly=self._config.readonly,
            )
            self._connected = True

            accounts = self._ib.managedAccounts()
            account = self._config.account or (accounts[0] if accounts else "?")
            logger.info(
                "IB connected: host=%s:%d account=%s paper=%s",
                self._config.host, self._config.port, account, self._config.is_paper,
            )
            return True
        except Exception as e:
            logger.error("IB connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect from IB."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("IB disconnected")

    def is_connected(self) -> bool:
        return self._connected and self._ib is not None and self._ib.isConnected()

    # ------------------------------------------------------------------
    # VenueAdapter protocol
    # ------------------------------------------------------------------

    def list_instruments(self, symbols: list[tuple[str, str]] | None = None) -> Tuple[InstrumentInfo, ...]:
        """Get instrument metadata.

        Args:
            symbols: List of (symbol, secType) tuples.
                     e.g. [("AAPL", "STK"), ("EURUSD", "CASH"), ("ES", "FUT")]
                     If None, returns details for configured symbols.
        """
        self._ensure_connected()
        instruments: list[InstrumentInfo] = []

        if symbols is None:
            # Use configured symbols, default to STK
            symbols = [(s, "STK") for s in self._config.symbols]

        for sym, sec_type in symbols:
            contract = make_contract(sym, sec_type)
            try:
                self._ib.qualifyContracts(contract)
                details_list = self._ib.reqContractDetails(contract)
                for cd in details_list[:1]:  # take first match
                    inst = map_contract_details(cd)
                    self._contract_cache[inst.symbol] = cd.contract
                    instruments.append(inst)
            except Exception as e:
                logger.warning("Failed to get details for %s/%s: %s", sym, sec_type, e)

        return tuple(instruments)

    def get_balances(self) -> BalanceSnapshot:
        """Get current account balances."""
        self._ensure_connected()
        account = self._config.account or self._ib.managedAccounts()[0]
        values = self._ib.accountValues(account)
        return map_account_values(values, account)

    def get_positions(self) -> Tuple[VenuePosition, ...]:
        """Get all open positions."""
        self._ensure_connected()
        positions = self._ib.positions()
        account = self._config.account
        if account:
            positions = [p for p in positions if p.account == account]
        return tuple(map_position(p) for p in positions if p.position != 0)

    def get_open_orders(
        self, *, symbol: Optional[str] = None,
    ) -> Tuple[CanonicalOrder, ...]:
        """Get all open/pending orders."""
        self._ensure_connected()
        trades = self._ib.openTrades()
        if symbol:
            trades = [t for t in trades
                      if (t.contract.localSymbol or t.contract.symbol) == symbol]
        return tuple(map_trade(t) for t in trades)

    def get_recent_fills(
        self, *, symbol: Optional[str] = None, since_ms: int = 0,
    ) -> Tuple[CanonicalFill, ...]:
        """Get recent fills/executions."""
        self._ensure_connected()
        fills = self._ib.fills()
        if symbol:
            fills = [f for f in fills
                     if (f.contract.localSymbol or f.contract.symbol) == symbol]
        return tuple(map_fill(f) for f in fills)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def _resolve_contract(self, symbol: str, sec_type: str = "STK", **kwargs: Any) -> Any:
        """Get or create a qualified IB Contract."""
        cache_key = f"{symbol}:{sec_type}"
        if cache_key in self._contract_cache:
            return self._contract_cache[cache_key]

        contract = make_contract(symbol, sec_type, **kwargs)
        qualified = self._ib.qualifyContracts(contract)
        if qualified:
            self._contract_cache[cache_key] = qualified[0]
            return qualified[0]
        raise ValueError(f"Cannot qualify contract: {symbol}/{sec_type}")

    def send_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        *,
        sec_type: str = "STK",
        tif: str = "GTC",
        **kwargs: Any,
    ) -> dict:
        """Send a market order.

        Args:
            symbol: Trading symbol.
            side: "buy" or "sell".
            qty: Quantity (shares/lots/contracts).
            sec_type: Security type — "STK", "CASH", "FUT", "OPT", "CFD", "CRYPTO".
            tif: Time in force — "GTC", "IOC", "DAY".

        Returns:
            Dict with order result.
        """
        self._ensure_connected()
        from ib_insync import MarketOrder

        contract = self._resolve_contract(symbol, sec_type, **kwargs)
        action = "BUY" if side.lower() == "buy" else "SELL"
        order = MarketOrder(action, qty, tif=tif)

        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.1)  # allow status update

        logger.info(
            "IB %s %s %s qty=%.2f orderId=%d",
            action, sec_type, symbol, qty, trade.order.orderId,
        )
        return {
            "orderId": trade.order.orderId,
            "status": trade.orderStatus.status if trade.orderStatus else "submitted",
            "symbol": symbol,
            "side": side,
            "qty": qty,
        }

    def send_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        *,
        sec_type: str = "STK",
        tif: str = "GTC",
        **kwargs: Any,
    ) -> dict:
        """Send a limit order."""
        self._ensure_connected()
        from ib_insync import LimitOrder

        contract = self._resolve_contract(symbol, sec_type, **kwargs)
        action = "BUY" if side.lower() == "buy" else "SELL"
        order = LimitOrder(action, qty, price, tif=tif)

        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.1)

        logger.info(
            "IB LIMIT %s %s %s qty=%.2f @ %.5f orderId=%d",
            action, sec_type, symbol, qty, price, trade.order.orderId,
        )
        return {
            "orderId": trade.order.orderId,
            "status": trade.orderStatus.status if trade.orderStatus else "submitted",
            "symbol": symbol,
            "price": price,
        }

    def send_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        *,
        sec_type: str = "STK",
        tif: str = "GTC",
        **kwargs: Any,
    ) -> dict:
        """Send a stop order."""
        self._ensure_connected()
        from ib_insync import StopOrder

        contract = self._resolve_contract(symbol, sec_type, **kwargs)
        action = "BUY" if side.lower() == "buy" else "SELL"
        order = StopOrder(action, qty, stop_price, tif=tif)

        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.1)

        return {
            "orderId": trade.order.orderId,
            "status": trade.orderStatus.status if trade.orderStatus else "submitted",
        }

    def send_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        *,
        sec_type: str = "STK",
        **kwargs: Any,
    ) -> dict:
        """Send a bracket order (entry + TP + SL).

        Returns dict with parent and child order IDs.
        """
        self._ensure_connected()
        contract = self._resolve_contract(symbol, sec_type, **kwargs)
        action = "BUY" if side.lower() == "buy" else "SELL"

        bracket = self._ib.bracketOrder(action, qty, entry_price, take_profit, stop_loss)
        trades = []
        for o in bracket:
            trade = self._ib.placeOrder(contract, o)
            trades.append(trade)
        self._ib.sleep(0.1)

        return {
            "parent_orderId": trades[0].order.orderId if trades else None,
            "tp_orderId": trades[1].order.orderId if len(trades) > 1 else None,
            "sl_orderId": trades[2].order.orderId if len(trades) > 2 else None,
            "status": "submitted",
        }

    def cancel_order(self, order_id: int) -> dict:
        """Cancel an order by orderId."""
        self._ensure_connected()
        for trade in self._ib.openTrades():
            if trade.order.orderId == order_id:
                self._ib.cancelOrder(trade.order)
                self._ib.sleep(0.1)
                return {"orderId": order_id, "status": "cancel_submitted"}
        return {"orderId": order_id, "status": "not_found"}

    def cancel_all(self) -> dict:
        """Cancel all open orders."""
        self._ensure_connected()
        self._ib.reqGlobalCancel()
        self._ib.sleep(0.5)
        return {"status": "global_cancel_submitted"}

    def close_position(self, symbol: str, sec_type: str = "STK", **kwargs: Any) -> dict:
        """Close an open position by sending an offsetting market order."""
        self._ensure_connected()
        positions = self._ib.positions()
        for pos in positions:
            pos_symbol = pos.contract.localSymbol or pos.contract.symbol
            if pos_symbol == symbol and pos.position != 0:
                side = "sell" if pos.position > 0 else "buy"
                qty = abs(pos.position)
                return self.send_market_order(symbol, side, qty, sec_type=sec_type, **kwargs)
        return {"status": "no_position", "symbol": symbol}

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str, sec_type: str = "STK", **kwargs: Any) -> Optional[dict]:
        """Get real-time snapshot (bid/ask/last/volume)."""
        self._ensure_connected()
        contract = self._resolve_contract(symbol, sec_type, **kwargs)
        ticker = self._ib.reqMktData(contract, snapshot=True)
        self._ib.sleep(2)  # wait for data

        result = {
            "symbol": symbol,
            "bid": ticker.bid if ticker.bid == ticker.bid else None,
            "ask": ticker.ask if ticker.ask == ticker.ask else None,
            "last": ticker.last if ticker.last == ticker.last else None,
            "volume": ticker.volume if ticker.volume == ticker.volume else None,
            "high": ticker.high if ticker.high == ticker.high else None,
            "low": ticker.low if ticker.low == ticker.low else None,
            "close": ticker.close if ticker.close == ticker.close else None,
        }
        self._ib.cancelMktData(contract)
        return result

    def get_bars(
        self,
        symbol: str,
        sec_type: str = "STK",
        timeframe: str = "5 mins",
        duration: str = "1 D",
        **kwargs: Any,
    ) -> list[dict]:
        """Get historical OHLCV bars.

        Args:
            symbol: Trading symbol.
            sec_type: Security type.
            timeframe: Bar size — "1 min", "5 mins", "15 mins", "1 hour", "1 day".
            duration: How far back — "1 D", "1 W", "1 M", "1 Y".

        Returns:
            List of bar dicts.
        """
        self._ensure_connected()
        contract = self._resolve_contract(symbol, sec_type, **kwargs)

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=timeframe,
            whatToShow="MIDPOINT" if sec_type in ("CASH", "FX", "FOREX") else "TRADES",
            useRTH=False,
        )
        return [
            {
                "time": int(b.date.timestamp()) if hasattr(b.date, "timestamp") else 0,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_summary(self) -> dict:
        """Get detailed account summary."""
        self._ensure_connected()
        account = self._config.account or self._ib.managedAccounts()[0]
        values = self._ib.accountValues(account)

        summary: dict[str, Any] = {"account": account}
        for av in values:
            if av.currency in ("BASE", ""):
                continue
            if av.tag in ("NetLiquidation", "TotalCashBalance", "BuyingPower",
                          "AvailableFunds", "MaintMarginReq", "UnrealizedPnL",
                          "RealizedPnL", "GrossPositionValue"):
                key = f"{av.tag}_{av.currency}"
                try:
                    summary[key] = float(av.value)
                except (ValueError, TypeError):
                    summary[key] = av.value
        return summary

    def get_portfolio(self) -> list[dict]:
        """Get portfolio items with market values."""
        self._ensure_connected()
        items = self._ib.portfolio()
        return [
            {
                "symbol": item.contract.localSymbol or item.contract.symbol,
                "secType": item.contract.secType,
                "position": item.position,
                "marketPrice": item.marketPrice,
                "marketValue": item.marketValue,
                "averageCost": item.averageCost,
                "unrealizedPNL": item.unrealizedPNL,
                "realizedPNL": item.realizedPNL,
            }
            for item in items
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if not self._connected or not self._ib or not self._ib.isConnected():
            raise RuntimeError("IB adapter not connected. Call connect() first.")
