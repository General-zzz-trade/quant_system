"""Smart order router — routes orders to the best venue based on price and liquidity.

Compares quotes across multiple venues and selects the optimal execution path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VenueQuote:
    """Price quote from a single venue."""
    venue_id: str
    bid: Decimal
    ask: Decimal
    bid_qty: Decimal
    ask_qty: Decimal
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Routing decision for an order."""
    venue_id: str
    price: Decimal
    available_qty: Decimal
    reason: str


@dataclass(frozen=True, slots=True)
class SplitRoute:
    """Multi-venue split for large orders."""
    legs: tuple[RouteDecision, ...]
    total_qty: Decimal
    estimated_avg_price: Decimal


class QuoteProvider(Protocol):
    """Protocol for venue quote retrieval."""
    def get_quote(self, symbol: str) -> Optional[VenueQuote]: ...


class SmartRouter:
    """Routes orders to the venue with the best price.

    Supports single-venue best-price routing and multi-venue splitting
    for large orders that exceed single-venue liquidity.
    """

    def __init__(
        self,
        *,
        max_latency_ms: float = 500.0,
        min_improvement_bps: float = 1.0,
    ) -> None:
        self._providers: Dict[str, QuoteProvider] = {}
        self._max_latency_ms = max_latency_ms
        self._min_improvement_bps = min_improvement_bps

    def register_venue(self, venue_id: str, provider: QuoteProvider) -> None:
        self._providers[venue_id] = provider

    def unregister_venue(self, venue_id: str) -> None:
        self._providers.pop(venue_id, None)

    @property
    def venue_ids(self) -> List[str]:
        return list(self._providers.keys())

    def collect_quotes(self, symbol: str) -> List[VenueQuote]:
        """Gather quotes from all registered venues."""
        quotes: List[VenueQuote] = []
        for venue_id, provider in self._providers.items():
            try:
                q = provider.get_quote(symbol)
                if q is not None and q.latency_ms <= self._max_latency_ms:
                    quotes.append(q)
            except Exception as e:
                logger.warning("Quote fetch failed for %s: %s", venue_id, e)
        return quotes

    def route_order(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
    ) -> Optional[RouteDecision]:
        """Route to the single best venue for this order."""
        quotes = self.collect_quotes(symbol)
        if not quotes:
            return None

        if side.lower() == "buy":
            # Best ask (lowest price with sufficient qty)
            candidates = [
                q for q in quotes if q.ask_qty >= qty and q.ask > 0
            ]
            if not candidates:
                # Fall back to best price even if insufficient qty
                candidates = [q for q in quotes if q.ask > 0]
            if not candidates:
                return None
            best = min(candidates, key=lambda q: q.ask)
            return RouteDecision(
                venue_id=best.venue_id,
                price=best.ask,
                available_qty=best.ask_qty,
                reason="best_ask",
            )
        else:
            # Best bid (highest price with sufficient qty)
            candidates = [
                q for q in quotes if q.bid_qty >= qty and q.bid > 0
            ]
            if not candidates:
                candidates = [q for q in quotes if q.bid > 0]
            if not candidates:
                return None
            best = max(candidates, key=lambda q: q.bid)
            return RouteDecision(
                venue_id=best.venue_id,
                price=best.bid,
                available_qty=best.bid_qty,
                reason="best_bid",
            )

    def split_order(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
    ) -> Optional[SplitRoute]:
        """Split a large order across multiple venues for better fill."""
        quotes = self.collect_quotes(symbol)
        if not quotes:
            return None

        is_buy = side.lower() == "buy"

        # Sort by price: ascending for buys (cheapest first), descending for sells
        if is_buy:
            sorted_quotes = sorted(
                [q for q in quotes if q.ask > 0],
                key=lambda q: q.ask,
            )
        else:
            sorted_quotes = sorted(
                [q for q in quotes if q.bid > 0],
                key=lambda q: q.bid,
                reverse=True,
            )

        if not sorted_quotes:
            return None

        legs: List[RouteDecision] = []
        remaining = qty
        total_cost = Decimal("0")

        for q in sorted_quotes:
            if remaining <= 0:
                break
            avail = q.ask_qty if is_buy else q.bid_qty
            fill_qty = min(remaining, avail)
            price = q.ask if is_buy else q.bid

            legs.append(RouteDecision(
                venue_id=q.venue_id,
                price=price,
                available_qty=fill_qty,
                reason="split_fill",
            ))
            total_cost += price * fill_qty
            remaining -= fill_qty

        filled = qty - remaining
        if filled <= 0:
            return None

        avg_price = total_cost / filled

        return SplitRoute(
            legs=tuple(legs),
            total_qty=filled,
            estimated_avg_price=avg_price,
        )
