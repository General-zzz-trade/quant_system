# execution/adapters/ib/data_bridge.py
"""Bridge IB real-time market data into quant_system engine events.

Uses IB's streaming bar/tick subscriptions and converts them to
MarketEvent dicts compatible with the pipeline/coordinator.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)


class IBDataBridge:
    """Subscribes to IB real-time bars and emits MarketEvent dicts.

    Usage:
        bridge = IBDataBridge(adapter)
        bridge.on_bar = my_callback  # receives (symbol, bar_dict)
        bridge.subscribe("EURUSD", "CASH", "5 secs")
        bridge.subscribe("AAPL", "STK", "5 secs")
        adapter._ib.run()  # blocking event loop
    """

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._ib = adapter._ib
        self._subscriptions: dict[str, Any] = {}

        # Callbacks
        self.on_bar: Optional[Callable[[str, dict], None]] = None
        self.on_tick: Optional[Callable[[str, dict], None]] = None

    def subscribe(
        self,
        symbol: str,
        sec_type: str = "STK",
        bar_size: str = "5 secs",
        what_to_show: str = "",
        **kwargs: Any,
    ) -> None:
        """Subscribe to real-time bars for a symbol.

        Args:
            symbol: Trading symbol.
            sec_type: Security type.
            bar_size: "5 secs", "10 secs", "15 secs", "30 secs",
                      "1 min", "2 mins", "3 mins", "5 mins", "10 mins", "15 mins".
            what_to_show: "TRADES", "MIDPOINT", "BID", "ASK" (auto-detected if empty).
        """
        contract = self._adapter._resolve_contract(symbol, sec_type, **kwargs)

        if not what_to_show:
            what_to_show = "MIDPOINT" if sec_type in ("CASH", "FX", "FOREX") else "TRADES"

        bars = self._ib.reqRealTimeBars(contract, 5, what_to_show, useRTH=False)
        bars.updateEvent += lambda bars, hasNewBar: self._on_realtime_bar(symbol, bars, hasNewBar)
        self._subscriptions[symbol] = bars

        logger.info("IBDataBridge subscribed: %s/%s bar_size=%s", symbol, sec_type, bar_size)

    def subscribe_ticks(self, symbol: str, sec_type: str = "STK", **kwargs: Any) -> None:
        """Subscribe to tick-by-tick data (bid/ask)."""
        contract = self._adapter._resolve_contract(symbol, sec_type, **kwargs)
        ticker = self._ib.reqMktData(contract)
        ticker.updateEvent += lambda t: self._on_tick_update(symbol, t)
        logger.info("IBDataBridge tick subscription: %s/%s", symbol, sec_type)

    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from a symbol."""
        bars = self._subscriptions.pop(symbol, None)
        if bars:
            self._ib.cancelRealTimeBars(bars)
            logger.info("IBDataBridge unsubscribed: %s", symbol)

    def unsubscribe_all(self) -> None:
        """Unsubscribe all."""
        for sym in list(self._subscriptions):
            self.unsubscribe(sym)

    def _on_realtime_bar(self, symbol: str, bars: Any, has_new: bool) -> None:
        """Handle new real-time bar."""
        if not has_new or not bars:
            return
        bar = bars[-1]
        event = {
            "symbol": symbol,
            "timestamp_ms": int(bar.time.timestamp() * 1000) if hasattr(bar.time, "timestamp") else 0,
            "open": bar.open_,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "venue": "ib",
            "funding_rate": None,
            "liquidation_buy_volume": None,
            "liquidation_sell_volume": None,
            "open_interest": None,
        }
        if self.on_bar:
            self.on_bar(symbol, event)

    def _on_tick_update(self, symbol: str, ticker: Any) -> None:
        """Handle tick update."""
        if self.on_tick is None:
            return
        self.on_tick(symbol, {
            "symbol": symbol,
            "bid": ticker.bid if ticker.bid == ticker.bid else None,
            "ask": ticker.ask if ticker.ask == ticker.ask else None,
            "last": ticker.last if ticker.last == ticker.last else None,
            "volume": ticker.volume if ticker.volume == ticker.volume else None,
        })
