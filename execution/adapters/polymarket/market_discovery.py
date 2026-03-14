"""Polymarket market discovery — filter markets by crypto relevance, volume, expiry."""
from __future__ import annotations

from decimal import Decimal
from typing import List, Sequence

from execution.adapters.polymarket.types import PolymarketMarket


def filter_crypto_markets(
    markets: Sequence[PolymarketMarket],
    *,
    keywords: Sequence[str],
    min_volume: Decimal,
    min_hours_to_expiry: float,
    now_iso: str,
) -> List[PolymarketMarket]:
    """Filter markets to crypto-related, sufficiently liquid, not-yet-expiring ones.

    Args:
        markets: All available markets.
        keywords: Crypto keywords to match against question/slug/description.
        min_volume: Minimum 24h volume in USD.
        min_hours_to_expiry: Minimum hours until market expiry.
        now_iso: Current time as ISO string (e.g. "2026-01-01T00:00:00Z").

    Returns:
        Filtered list of matching markets.
    """
    result: List[PolymarketMarket] = []
    for m in markets:
        if not m.active:
            continue
        if not m.is_crypto(keywords):
            continue
        if m.volume_24h < min_volume:
            continue
        if m.hours_to_expiry(now_iso) < min_hours_to_expiry:
            continue
        result.append(m)
    return result
