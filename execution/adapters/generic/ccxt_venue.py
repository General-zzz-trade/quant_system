"""CCXT unified venue adapter — provides a standard interface to 100+ exchanges.

Requires: pip install ccxt
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VenueBalance:
    currency: str
    free: Decimal
    used: Decimal
    total: Decimal


@dataclass(frozen=True, slots=True)
class VenuePosition:
    symbol: str
    side: str
    qty: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class VenueOrder:
    order_id: str
    symbol: str
    side: str
    qty: Decimal
    price: Optional[Decimal]
    status: str
    filled_qty: Decimal
    avg_fill_price: Optional[Decimal]


class CcxtVenueAdapter:
    """Unified exchange adapter using ccxt.

    Provides standardized access to balances, positions, and order placement
    across multiple exchanges.
    """

    def __init__(
        self,
        exchange_id: str,
        *,
        api_key: str = "",
        api_secret: str = "",
        sandbox: bool = False,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import ccxt  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("ccxt not installed: pip install ccxt") from e

        config: Dict[str, Any] = {
            "enableRateLimit": True,
        }
        if api_key:
            config["apiKey"] = api_key
        if api_secret:
            config["secret"] = api_secret
        if extra_config:
            config.update(extra_config)

        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {exchange_id}")

        self._exchange = exchange_class(config)
        if sandbox:
            self._exchange.set_sandbox_mode(True)

        self._exchange_id = exchange_id
        logger.info("CcxtVenueAdapter initialized: %s (sandbox=%s)", exchange_id, sandbox)

    @property
    def exchange_id(self) -> str:
        return self._exchange_id

    def get_balances(self) -> List[VenueBalance]:
        """Fetch account balances."""
        balance = self._exchange.fetch_balance()
        result = []
        for currency, data in balance.get("total", {}).items():
            if data and float(data) > 0:
                free = balance.get("free", {}).get(currency, 0)
                used = balance.get("used", {}).get(currency, 0)
                result.append(VenueBalance(
                    currency=currency,
                    free=Decimal(str(free or 0)),
                    used=Decimal(str(used or 0)),
                    total=Decimal(str(data)),
                ))
        return result

    def get_positions(self) -> List[VenuePosition]:
        """Fetch open positions (futures/perps only)."""
        try:
            positions = self._exchange.fetch_positions()
        except Exception:
            return []

        result = []
        for pos in positions:
            qty = abs(float(pos.get("contracts", 0) or 0))
            if qty == 0:
                continue
            result.append(VenuePosition(
                symbol=pos.get("symbol", ""),
                side=pos.get("side", ""),
                qty=Decimal(str(qty)),
                avg_price=Decimal(str(pos.get("entryPrice", 0) or 0)),
                unrealized_pnl=Decimal(str(pos.get("unrealizedPnl", 0) or 0)),
            ))
        return result

    def place_market_order(
        self, symbol: str, side: str, qty: Decimal,
    ) -> VenueOrder:
        """Place a market order."""
        order = self._exchange.create_order(
            symbol=symbol,
            type="market",
            side=side.lower(),
            amount=float(qty),
        )
        return self._parse_order(order)

    def place_limit_order(
        self, symbol: str, side: str, qty: Decimal, price: Decimal,
    ) -> VenueOrder:
        """Place a limit order."""
        order = self._exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side.lower(),
            amount=float(qty),
            price=float(price),
        )
        return self._parse_order(order)

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        try:
            self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning("Failed to cancel %s: %s", order_id, e)
            return False

    def _parse_order(self, raw: Dict[str, Any]) -> VenueOrder:
        return VenueOrder(
            order_id=str(raw.get("id", "")),
            symbol=raw.get("symbol", ""),
            side=raw.get("side", ""),
            qty=Decimal(str(raw.get("amount", 0))),
            price=Decimal(str(raw["price"])) if raw.get("price") else None,
            status=raw.get("status", ""),
            filled_qty=Decimal(str(raw.get("filled", 0) or 0)),
            avg_fill_price=Decimal(str(raw["average"])) if raw.get("average") else None,
        )
