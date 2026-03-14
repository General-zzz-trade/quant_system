from decimal import Decimal
from execution.adapters.polymarket.types import PolymarketMarket, PolymarketOrderbook


def test_market_symbol_format():
    m = PolymarketMarket(condition_id="0x1", slug="btc-above-100k", question="BTC?",
        outcomes=("Yes", "No"), token_ids=("t1", "t2"), end_date_iso="2026-12-31T00:00:00Z",
        active=True, volume_24h=Decimal("50000"))
    assert m.symbol("Yes") == "POLY:btc-above-100k:YES"


def test_market_is_crypto():
    m = PolymarketMarket(condition_id="0x1", slug="btc-price", question="Will Bitcoin exceed 100K?",
        outcomes=("Yes", "No"), token_ids=("a", "b"), end_date_iso="2026-12-31T00:00:00Z",
        active=True, volume_24h=Decimal("5000"))
    assert m.is_crypto(("BTC", "Bitcoin"))
    m2 = PolymarketMarket(condition_id="0x2", slug="election", question="Who wins?",
        outcomes=("A", "B"), token_ids=("c", "d"), end_date_iso="2026-12-31T00:00:00Z",
        active=True, volume_24h=Decimal("5000"))
    assert not m2.is_crypto(("BTC", "Bitcoin"))


def test_orderbook_properties():
    ob = PolymarketOrderbook(token_id="t1",
        bids=((Decimal("0.65"), Decimal("100")),),
        asks=((Decimal("0.70"), Decimal("150")),))
    assert ob.best_bid == Decimal("0.65")
    assert ob.best_ask == Decimal("0.70")
    assert ob.mid_price == Decimal("0.675")
    assert ob.spread == Decimal("0.05")


def test_orderbook_empty():
    ob = PolymarketOrderbook(token_id="t1", bids=(), asks=())
    assert ob.best_bid is None
    assert ob.best_ask is None
    assert ob.mid_price is None
    assert ob.spread is None


def test_token_id_for():
    m = PolymarketMarket(condition_id="0x1", slug="s", question="q",
        outcomes=("Yes", "No"), token_ids=("ty", "tn"), end_date_iso="2026-12-31T00:00:00Z",
        active=True, volume_24h=Decimal("1000"))
    assert m.token_id_for("Yes") == "ty"
    assert m.token_id_for("No") == "tn"
    assert m.token_id_for("Maybe") is None


def test_hours_to_expiry():
    m = PolymarketMarket(condition_id="0x1", slug="s", question="q",
        outcomes=("Yes",), token_ids=("t1",), end_date_iso="2026-12-31T00:00:00Z",
        active=True, volume_24h=Decimal("1000"))
    hours = m.hours_to_expiry("2026-12-30T00:00:00Z")
    assert hours == 24.0


def test_hours_to_expiry_past():
    m = PolymarketMarket(condition_id="0x1", slug="s", question="q",
        outcomes=("Yes",), token_ids=("t1",), end_date_iso="2026-01-01T00:00:00Z",
        active=True, volume_24h=Decimal("1000"))
    hours = m.hours_to_expiry("2026-06-01T00:00:00Z")
    assert hours == 0.0
