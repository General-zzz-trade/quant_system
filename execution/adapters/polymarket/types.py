"""Polymarket domain types — markets, outcomes, orderbooks."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class PolymarketMarket:
    condition_id: str
    slug: str
    question: str
    outcomes: Tuple[str, ...]
    token_ids: Tuple[str, ...]
    end_date_iso: str
    active: bool
    volume_24h: Decimal
    description: str = ""
    category: str = ""

    def symbol(self, outcome: str) -> str:
        return f"POLY:{self.slug}:{outcome.upper()}"

    def token_id_for(self, outcome: str) -> Optional[str]:
        for i, o in enumerate(self.outcomes):
            if o.lower() == outcome.lower() and i < len(self.token_ids):
                return self.token_ids[i]
        return None

    def is_crypto(self, keywords: Sequence[str] = ()) -> bool:
        text = f"{self.question} {self.slug} {self.description}".lower()
        return any(kw.lower() in text for kw in keywords)

    def hours_to_expiry(self, now_iso: str) -> float:
        from datetime import datetime
        end = datetime.fromisoformat(self.end_date_iso.replace("Z", "+00:00"))
        now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        return max(0.0, (end - now).total_seconds() / 3600)


@dataclass(frozen=True, slots=True)
class PolymarketOrderbook:
    token_id: str
    bids: Tuple[Tuple[Decimal, Decimal], ...]
    asks: Tuple[Tuple[Decimal, Decimal], ...]
    timestamp_ms: int = 0

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None
