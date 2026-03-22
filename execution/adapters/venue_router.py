# execution/adapters/venue_router.py
"""Venue router — routes orders across multiple venue adapters for best execution.

Implements the VenueAdapter protocol so it can be used as a drop-in replacement
for a single adapter. Routes orders to the venue with the best fee structure,
with automatic fallback if the primary venue fails.

Usage:
    from execution.adapters.venue_router import VenueRouter

    router = VenueRouter([bybit_adapter, hl_adapter])
    router.send_market_order("BTCUSDT", "buy", 0.001)
    router.get_balances()  # aggregated across venues
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple

from execution.models.balances import BalanceSnapshot, CanonicalBalance
from execution.models.fills import CanonicalFill
from execution.models.instruments import InstrumentInfo
from execution.models.orders import CanonicalOrder
from execution.models.positions import VenuePosition

logger = logging.getLogger(__name__)


# Fee schedules (taker / maker) in basis points.
# Used for routing decisions. Update if venue fee tiers change.
_DEFAULT_FEES: dict[str, dict[str, float]] = {
    "bybit": {"taker": 0.055, "maker": 0.02},       # 5.5bps / 2bps
    "hyperliquid": {"taker": 0.035, "maker": 0.0},   # 3.5bps / 0bps
    "binance": {"taker": 0.05, "maker": 0.02},       # 5bps / 2bps
}


class VenueRouter:
    """Routes orders to the best venue from a list of adapters.

    Routing logic:
    - Limit orders: prefer venue with lowest maker fee (0% maker -> top choice).
    - Market orders: prefer venue with lowest taker fee.
    - Fallback: if primary venue fails, try the next venue in fee-ranked order.
    - Balances and positions: aggregated across all venues.
    """

    venue: str = "router"

    def __init__(
        self,
        adapters: Sequence[Any],
        *,
        fee_overrides: dict[str, dict[str, float]] | None = None,
    ) -> None:
        if not adapters:
            raise ValueError("VenueRouter requires at least one adapter")

        self._adapters: list[Any] = list(adapters)
        self._fees = dict(_DEFAULT_FEES)
        if fee_overrides:
            self._fees.update(fee_overrides)

        venue_names = [a.venue for a in self._adapters]
        logger.info("VenueRouter initialized with venues: %s", venue_names)

        # Pre-compute ranked orders for maker and taker
        self._maker_ranked = self._rank_by_fee("maker")
        self._taker_ranked = self._rank_by_fee("taker")

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _rank_by_fee(self, fee_type: str) -> list[Any]:
        """Return adapters sorted by fee (lowest first)."""
        def fee_key(adapter: Any) -> float:
            venue = adapter.venue.lower()
            return self._fees.get(venue, {}).get(fee_type, 999.0)
        return sorted(self._adapters, key=fee_key)

    def _get_primary_for_market(self) -> list[Any]:
        """Adapters ranked for market (taker) orders."""
        return list(self._taker_ranked)

    def _get_primary_for_limit(self) -> list[Any]:
        """Adapters ranked for limit (maker) orders."""
        return list(self._maker_ranked)

    def _venue_fee(self, adapter: Any, fee_type: str) -> float:
        venue = adapter.venue.lower()
        return self._fees.get(venue, {}).get(fee_type, 999.0)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect all adapters. Returns True if at least one succeeds."""
        any_ok = False
        for adapter in self._adapters:
            try:
                if hasattr(adapter, "connect") and adapter.connect():
                    any_ok = True
                    logger.info("VenueRouter: %s connected", adapter.venue)
                elif hasattr(adapter, "is_connected") and adapter.is_connected():
                    any_ok = True
            except Exception:
                logger.exception("VenueRouter: %s connect failed", adapter.venue)
        return any_ok

    def is_connected(self) -> bool:
        """True if at least one adapter is connected."""
        for adapter in self._adapters:
            if hasattr(adapter, "is_connected") and adapter.is_connected():
                return True
        return False

    # ------------------------------------------------------------------
    # VenueAdapter protocol — read operations (aggregated)
    # ------------------------------------------------------------------

    def list_instruments(self, symbols: list[str] | None = None) -> Tuple[InstrumentInfo, ...]:
        """List instruments from all venues (deduplicated by symbol)."""
        seen: set[str] = set()
        result: list[InstrumentInfo] = []
        for adapter in self._adapters:
            try:
                if symbols:
                    instruments = adapter.list_instruments(symbols)
                else:
                    instruments = adapter.list_instruments()
                for inst in instruments:
                    if inst.symbol not in seen:
                        seen.add(inst.symbol)
                        result.append(inst)
            except Exception:
                logger.warning(
                    "VenueRouter: list_instruments failed on %s", adapter.venue,
                    exc_info=True,
                )
        return tuple(result)

    def get_balances(self) -> BalanceSnapshot:
        """Aggregate balances across all venues."""
        all_balances: list[CanonicalBalance] = []
        ts_ms = 0
        for adapter in self._adapters:
            try:
                snap = adapter.get_balances()
                all_balances.extend(snap.balances)
                ts_ms = max(ts_ms, snap.ts_ms)
            except Exception:
                logger.warning(
                    "VenueRouter: get_balances failed on %s", adapter.venue,
                    exc_info=True,
                )
        return BalanceSnapshot(
            venue="router",
            balances=tuple(all_balances),
            ts_ms=ts_ms,
        )

    def get_positions(self, symbol: str = "") -> Tuple[VenuePosition, ...]:
        """Aggregate positions across all venues."""
        all_positions: list[VenuePosition] = []
        for adapter in self._adapters:
            try:
                if symbol:
                    positions = adapter.get_positions(symbol=symbol)
                else:
                    positions = adapter.get_positions()
                all_positions.extend(positions)
            except Exception:
                logger.warning(
                    "VenueRouter: get_positions failed on %s", adapter.venue,
                    exc_info=True,
                )
        return tuple(all_positions)

    def get_open_orders(
        self, *, symbol: Optional[str] = None,
    ) -> Tuple[CanonicalOrder, ...]:
        """Aggregate open orders across all venues."""
        all_orders: list[CanonicalOrder] = []
        for adapter in self._adapters:
            try:
                orders = adapter.get_open_orders(symbol=symbol)
                all_orders.extend(orders)
            except Exception:
                logger.warning(
                    "VenueRouter: get_open_orders failed on %s", adapter.venue,
                    exc_info=True,
                )
        return tuple(all_orders)

    def get_recent_fills(
        self, *, symbol: Optional[str] = None, since_ms: int = 0,
    ) -> Tuple[CanonicalFill, ...]:
        """Aggregate recent fills across all venues."""
        all_fills: list[CanonicalFill] = []
        for adapter in self._adapters:
            try:
                fills = adapter.get_recent_fills(symbol=symbol, since_ms=since_ms)
                all_fills.extend(fills)
            except Exception:
                logger.warning(
                    "VenueRouter: get_recent_fills failed on %s", adapter.venue,
                    exc_info=True,
                )
        return tuple(all_fills)

    # ------------------------------------------------------------------
    # Market data — use first adapter that succeeds
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> dict:
        """Get ticker from first available venue."""
        for adapter in self._adapters:
            try:
                if hasattr(adapter, "get_ticker"):
                    result = adapter.get_ticker(symbol)
                    if result:
                        return result
            except Exception:
                logger.warning(
                    "VenueRouter: get_ticker failed on %s", adapter.venue,
                    exc_info=True,
                )
        return {}

    def get_klines(self, symbol: str, **kwargs: Any) -> list:
        """Get klines from first available venue."""
        for adapter in self._adapters:
            try:
                if hasattr(adapter, "get_klines"):
                    result = adapter.get_klines(symbol, **kwargs)
                    if result:
                        return result
            except Exception:
                logger.warning(
                    "VenueRouter: get_klines failed on %s", adapter.venue,
                    exc_info=True,
                )
        return []

    def get_orderbook(self, symbol: str) -> dict:
        """Get orderbook from first available venue."""
        for adapter in self._adapters:
            try:
                if hasattr(adapter, "get_orderbook"):
                    result = adapter.get_orderbook(symbol)
                    if result:
                        return result
            except Exception:
                logger.warning(
                    "VenueRouter: get_orderbook failed on %s", adapter.venue,
                    exc_info=True,
                )
        return {"bids": [], "asks": []}

    # ------------------------------------------------------------------
    # Order submission — routed with fallback
    # ------------------------------------------------------------------

    def send_market_order(
        self, symbol: str, side: str, qty: float,
        *, reduce_only: bool = False,
    ) -> dict:
        """Send market order to venue with lowest taker fee, with fallback."""
        ranked = self._get_primary_for_market()
        return self._route_order(
            ranked, "send_market_order",
            symbol, side, qty,
            reduce_only=reduce_only,
            order_type="market",
        )

    def send_limit_order(
        self, symbol: str, side: str, qty: float, price: float,
        *, tif: str = "Gtc", reduce_only: bool = False,
    ) -> dict:
        """Send limit order to venue with lowest maker fee, with fallback."""
        ranked = self._get_primary_for_limit()
        return self._route_order(
            ranked, "send_limit_order",
            symbol, side, qty, price,
            tif=tif, reduce_only=reduce_only,
            order_type="limit",
        )

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel order — tries all venues since we may not know which has it."""
        for adapter in self._adapters:
            try:
                result = adapter.cancel_order(symbol, order_id)
                if result.get("status") != "error":
                    logger.info(
                        "VenueRouter: canceled order %s on %s",
                        order_id, adapter.venue,
                    )
                    return result
            except Exception:
                continue
        return {"status": "error", "msg": "order not found on any venue"}

    def cancel_all(self, symbol: str = "") -> dict:
        """Cancel all orders across all venues."""
        total_canceled = 0
        for adapter in self._adapters:
            try:
                result = adapter.cancel_all(symbol)
                total_canceled += result.get("canceled", 0)
            except Exception:
                logger.warning(
                    "VenueRouter: cancel_all failed on %s", adapter.venue,
                    exc_info=True,
                )
        return {"status": "ok", "canceled": total_canceled}

    def close_position(self, symbol: str) -> dict:
        """Close position on whichever venue holds it."""
        for adapter in self._adapters:
            try:
                positions = adapter.get_positions(symbol=symbol)
                for pos in positions:
                    if not pos.is_flat:
                        result = adapter.close_position(symbol)
                        logger.info(
                            "VenueRouter: closed position on %s for %s",
                            adapter.venue, symbol,
                        )
                        return result
            except Exception:
                logger.warning(
                    "VenueRouter: close_position failed on %s", adapter.venue,
                    exc_info=True,
                )
        return {"status": "no_position"}

    def set_leverage(self, symbol: str, leverage: int, **kwargs: Any) -> dict:
        """Set leverage on all venues."""
        results = {}
        for adapter in self._adapters:
            try:
                if hasattr(adapter, "set_leverage"):
                    result = adapter.set_leverage(symbol, leverage, **kwargs)
                    results[adapter.venue] = result
            except Exception:
                logger.warning(
                    "VenueRouter: set_leverage failed on %s", adapter.venue,
                    exc_info=True,
                )
        return {"status": "ok", "results": results}

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_info(self) -> dict:
        """Aggregate account info from all venues."""
        result: dict[str, Any] = {"venue": "router", "venues": {}}
        for adapter in self._adapters:
            try:
                if hasattr(adapter, "get_account_info"):
                    result["venues"][adapter.venue] = adapter.get_account_info()
            except Exception:
                logger.warning(
                    "VenueRouter: get_account_info failed on %s", adapter.venue,
                    exc_info=True,
                )
        return result

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route_order(
        self,
        ranked_adapters: list[Any],
        method_name: str,
        *args: Any,
        order_type: str = "unknown",
        **kwargs: Any,
    ) -> dict:
        """Try sending order to each adapter in ranked order until one succeeds."""
        last_error = "no adapters available"
        fee_type = "maker" if order_type == "limit" else "taker"

        for adapter in ranked_adapters:
            fee_pct = self._venue_fee(adapter, fee_type)
            try:
                method = getattr(adapter, method_name)
                # Remove our internal kwarg before calling the adapter
                clean_kwargs = {k: v for k, v in kwargs.items() if k != "order_type"}
                result = method(*args, **clean_kwargs)

                if result.get("status") != "error":
                    logger.info(
                        "VenueRouter: %s %s routed to %s (fee=%.3f%%)",
                        order_type,
                        args[0] if args else "?",  # symbol
                        adapter.venue,
                        fee_pct,
                    )
                    result["routed_venue"] = adapter.venue
                    return result
                else:
                    last_error = result.get("msg", str(result))
                    logger.warning(
                        "VenueRouter: %s failed on %s: %s — trying next",
                        method_name, adapter.venue, last_error,
                    )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "VenueRouter: %s exception on %s: %s — trying next",
                    method_name, adapter.venue, e,
                    exc_info=True,
                )

        logger.error(
            "VenueRouter: %s failed on all venues. Last error: %s",
            method_name, last_error,
        )
        return {"status": "error", "msg": f"all venues failed: {last_error}"}

    # ------------------------------------------------------------------
    # Adapter access
    # ------------------------------------------------------------------

    def get_adapter(self, venue: str) -> Any | None:
        """Get a specific adapter by venue name."""
        for adapter in self._adapters:
            if adapter.venue.lower() == venue.lower():
                return adapter
        return None

    @property
    def adapters(self) -> list[Any]:
        """All registered adapters."""
        return list(self._adapters)

    def __repr__(self) -> str:
        venues = [a.venue for a in self._adapters]
        return f"VenueRouter(venues={venues})"
