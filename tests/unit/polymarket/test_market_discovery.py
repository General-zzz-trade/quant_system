"""Tests for Polymarket market discovery filter."""
from decimal import Decimal

from execution.adapters.polymarket.types import PolymarketMarket
from execution.adapters.polymarket.market_discovery import filter_crypto_markets


def _make_market(
    slug: str = "test",
    question: str = "Test?",
    volume: str = "50000",
    end_date: str = "2027-06-01T00:00:00Z",
    active: bool = True,
    description: str = "",
) -> PolymarketMarket:
    return PolymarketMarket(
        condition_id=f"0x{slug}",
        slug=slug,
        question=question,
        outcomes=("Yes", "No"),
        token_ids=("t1", "t2"),
        end_date_iso=end_date,
        active=active,
        volume_24h=Decimal(volume),
        description=description,
    )


class TestFilterCryptoMarkets:
    def test_filters_by_crypto_keywords(self):
        markets = [
            _make_market(slug="btc-100k", question="Will Bitcoin exceed 100K?"),
            _make_market(slug="election", question="Who wins the election?"),
            _make_market(slug="eth-merge", question="Will Ethereum merge succeed?"),
        ]
        result = filter_crypto_markets(
            markets, keywords=("BTC", "Bitcoin", "ETH", "Ethereum"),
            min_volume=Decimal("1000"), min_hours_to_expiry=1.0,
            now_iso="2026-01-01T00:00:00Z",
        )
        slugs = [m.slug for m in result]
        assert "btc-100k" in slugs
        assert "eth-merge" in slugs
        assert "election" not in slugs

    def test_filters_by_volume(self):
        markets = [
            _make_market(slug="btc-low", question="BTC?", volume="500"),
            _make_market(slug="btc-high", question="BTC?", volume="50000"),
        ]
        result = filter_crypto_markets(
            markets, keywords=("BTC",),
            min_volume=Decimal("10000"), min_hours_to_expiry=1.0,
            now_iso="2026-01-01T00:00:00Z",
        )
        assert len(result) == 1
        assert result[0].slug == "btc-high"

    def test_filters_by_expiry(self):
        markets = [
            _make_market(slug="btc-soon", question="BTC?", end_date="2026-01-01T12:00:00Z"),
            _make_market(slug="btc-far", question="BTC?", end_date="2027-06-01T00:00:00Z"),
        ]
        result = filter_crypto_markets(
            markets, keywords=("BTC",),
            min_volume=Decimal("1000"), min_hours_to_expiry=24.0,
            now_iso="2026-01-01T00:00:00Z",
        )
        assert len(result) == 1
        assert result[0].slug == "btc-far"

    def test_filters_inactive(self):
        markets = [
            _make_market(slug="btc-inactive", question="BTC?", active=False),
            _make_market(slug="btc-active", question="BTC?", active=True),
        ]
        result = filter_crypto_markets(
            markets, keywords=("BTC",),
            min_volume=Decimal("1000"), min_hours_to_expiry=1.0,
            now_iso="2026-01-01T00:00:00Z",
        )
        assert len(result) == 1
        assert result[0].slug == "btc-active"

    def test_empty_input(self):
        result = filter_crypto_markets(
            [], keywords=("BTC",),
            min_volume=Decimal("1000"), min_hours_to_expiry=1.0,
            now_iso="2026-01-01T00:00:00Z",
        )
        assert result == []

    def test_description_keyword_match(self):
        markets = [
            _make_market(
                slug="generic-market", question="Will it happen?",
                description="Related to Solana ecosystem",
            ),
        ]
        result = filter_crypto_markets(
            markets, keywords=("Solana",),
            min_volume=Decimal("1000"), min_hours_to_expiry=1.0,
            now_iso="2026-01-01T00:00:00Z",
        )
        assert len(result) == 1
