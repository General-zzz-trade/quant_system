# execution/venue_manager.py
"""VenueManager + SmartRouter — multi-exchange order routing."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, Tuple

from execution.models.venue import VenueInfo

logger = logging.getLogger(__name__)


class VenueGateway(Protocol):
    """Minimal gateway protocol for order submission."""
    venue: str

    def submit_order(self, cmd: Any) -> Dict[str, Any]: ...
    def cancel_order(self, cmd: Any) -> Dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Result of smart routing decision."""
    venue: str
    reason: str
    score: float = 0.0


@dataclass
class VenueManager:
    """Manages multiple exchange venues with unified access.

    Provides:
    - Venue registration and lookup
    - Health tracking per venue
    - SmartRouter for best execution routing
    """

    _venues: Dict[str, VenueInfo] = field(default_factory=dict, init=False)
    _gateways: Dict[str, VenueGateway] = field(default_factory=dict, init=False)
    _health: Dict[str, bool] = field(default_factory=dict, init=False)
    _fees: Dict[str, float] = field(default_factory=dict, init=False)

    def register(
        self,
        info: VenueInfo,
        gateway: VenueGateway,
        *,
        fee_bps: float = 4.0,
    ) -> None:
        name = info.name
        self._venues[name] = info
        self._gateways[name] = gateway
        self._health[name] = True
        self._fees[name] = fee_bps
        logger.info("Registered venue: %s (fee=%.1f bps)", name, fee_bps)

    def unregister(self, venue: str) -> None:
        self._venues.pop(venue, None)
        self._gateways.pop(venue, None)
        self._health.pop(venue, None)
        self._fees.pop(venue, None)

    def get_gateway(self, venue: str) -> VenueGateway:
        gw = self._gateways.get(venue)
        if gw is None:
            raise KeyError(f"Venue not registered: {venue}")
        return gw

    def set_health(self, venue: str, healthy: bool) -> None:
        if venue in self._health:
            self._health[venue] = healthy

    def is_healthy(self, venue: str) -> bool:
        return self._health.get(venue, False)

    @property
    def venues(self) -> List[str]:
        return list(self._venues.keys())

    @property
    def healthy_venues(self) -> List[str]:
        return [v for v, h in self._health.items() if h]

    def get_info(self, venue: str) -> Optional[VenueInfo]:
        return self._venues.get(venue)

    def submit_order(self, venue: str, cmd: Any) -> Dict[str, Any]:
        gw = self.get_gateway(venue)
        if not self.is_healthy(venue):
            raise RuntimeError(f"Venue {venue} is unhealthy")
        return gw.submit_order(cmd)

    def cancel_order(self, venue: str, cmd: Any) -> Dict[str, Any]:
        gw = self.get_gateway(venue)
        return gw.cancel_order(cmd)


@dataclass
class SmartRouter:
    """Routes orders to the best venue based on configurable scoring.

    Factors considered:
    - Venue health
    - Fee tier
    - Latency (configurable)
    - Symbol availability
    """

    manager: VenueManager
    symbol_venues: Dict[str, List[str]] = field(default_factory=dict)
    latency_scores: Dict[str, float] = field(default_factory=dict)

    def register_symbol(self, symbol: str, venues: List[str]) -> None:
        self.symbol_venues[symbol] = venues

    def set_latency(self, venue: str, latency_ms: float) -> None:
        self.latency_scores[venue] = latency_ms

    def route(self, symbol: str, side: str, qty: Decimal) -> RouteDecision:
        """Select best venue for an order."""
        candidates = self.symbol_venues.get(symbol, self.manager.healthy_venues)
        candidates = [v for v in candidates if self.manager.is_healthy(v)]

        if not candidates:
            raise RuntimeError(f"No healthy venue for {symbol}")

        if len(candidates) == 1:
            return RouteDecision(
                venue=candidates[0],
                reason="only_venue",
                score=1.0,
            )

        # Score each venue
        best_venue = candidates[0]
        best_score = -1.0

        for venue in candidates:
            fee = self.manager._fees.get(venue, 10.0)
            latency = self.latency_scores.get(venue, 100.0)

            # Lower fee and latency = higher score
            score = 100.0 - fee * 2.0 - latency * 0.01

            if score > best_score:
                best_score = score
                best_venue = venue

        return RouteDecision(
            venue=best_venue,
            reason=f"best_score={best_score:.1f}",
            score=best_score,
        )

    def route_and_submit(self, cmd: Any) -> Tuple[RouteDecision, Dict[str, Any]]:
        """Route and execute in one call."""
        symbol = getattr(cmd, "symbol", "")
        side = getattr(cmd, "side", "buy")
        qty = getattr(cmd, "qty", Decimal("0"))

        decision = self.route(symbol, side, qty)
        result = self.manager.submit_order(decision.venue, cmd)
        return decision, result
