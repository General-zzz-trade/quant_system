"""The account's funding income on one symbol (the strategy capital base, tech lead 2026-11-16): a read-only, paged
GET of Binance's income history. Only USDT funding of that symbol counts; an answer that cannot be read raises, so
the capped base sizes to zero instead of guessing.
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from trading.adapters.binance.adapter import BinanceAdapter
from trading.adapters.binance.config import BinanceConfig
from trading.adapters.binance.rest import BinanceRetryableError
from tests.unit.binance.fake_venue import HOUR, FakeBinance

T = 1_794_000_000_000          # since (ms)
MINUTE, DAY = 60_000, 86_400_000
NOW = T + 30 * DAY


def _adapter(client):
    return BinanceAdapter(BinanceConfig(api_key="k", api_secret="s", testnet=True), client=client)


class Pages:
    """Answers each request with the next prepared page, as given (no filtering), and records every call."""

    def __init__(self, *pages):
        self.pages, self.calls = list(pages), []

    def request_signed(self, *, method, path, params=None):
        self.calls.append((method, path, dict(params or {})))
        return self.pages.pop(0)

    def request_public(self, *, method, path, params=None):
        raise AssertionError("funding income is a signed account read")


def _row(income, time_ms, tran_id, *, symbol="BTCUSDT", asset="USDT", income_type="FUNDING_FEE"):
    return {"symbol": symbol, "incomeType": income_type, "income": income, "asset": asset, "info": income_type,
            "time": time_ms, "tranId": tran_id, "tradeId": ""}


def test_the_symbol_s_usdt_funding_since_the_time_is_summed_from_one_get():
    fake = FakeBinance()
    fake.add_income("BTCUSDT", "-1.5", T)                          # at the start time: counted (inclusive)
    fake.add_income("BTCUSDT", "0.25", T + 8 * HOUR)
    fake.add_income("BTCUSDT", "-9", T - 1)                        # before the strategy's first fill
    fake.add_income("BTCUSDT", "-2", T + HOUR, asset="BNB")        # not USDT
    fake.add_income("BTCUSDT", "-3", T + 2 * HOUR, income_type="COMMISSION")
    fake.add_income("ETHUSDT", "-4", T + 3 * HOUR)
    assert _adapter(fake).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW) == D("-1.25")
    assert fake.income_requests == [{"symbol": "BTCUSDT", "incomeType": "FUNDING_FEE", "startTime": T,
                                     "endTime": NOW, "limit": 1000}]
    assert fake.posts == [] and fake.algo == {} and fake.orders == {}


def test_rows_of_another_symbol_type_or_asset_never_count_even_if_the_venue_returns_them():
    client = Pages([_row("1", T, 1), _row("10", T, 2, symbol="ETHUSDT"), _row("100", T, 3, income_type="TRANSFER"),
                    _row("1000", T, 4, asset="USDC")])
    assert _adapter(client).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW) == D("1")
    assert [(method, path) for method, path, _ in client.calls] == [("GET", "/fapi/v1/income")]


def test_the_history_is_paged_by_time_and_a_row_asked_twice_counts_once():
    fake = FakeBinance()
    for i in range(2500):
        fake.add_income("BTCUSDT", "-0.01", T + i * MINUTE)
    assert _adapter(fake).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW) == D("-25.00")
    assert [r["startTime"] for r in fake.income_requests] == [T, T + 999 * MINUTE, T + 1998 * MINUTE]
    assert {r["limit"] for r in fake.income_requests} == {1000}


def test_a_row_sharing_the_time_of_a_page_s_last_row_is_not_lost():
    fake = FakeBinance()
    for i in range(1000):
        fake.add_income("BTCUSDT", "1", T + i * MINUTE)
    fake.add_income("BTCUSDT", "1", T + 999 * MINUTE)             # same time as the first page's last row
    assert _adapter(fake).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW) == D("1001")


def test_a_history_that_does_not_advance_raises_instead_of_looping():
    fake = FakeBinance()
    for _ in range(1000):
        fake.add_income("BTCUSDT", "1", T)
    with pytest.raises(RuntimeError, match="advance"):
        _adapter(fake).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW)


@pytest.mark.parametrize("page", [
    {"code": -1021, "msg": "Timestamp outside recvWindow"},        # not a list
    [_row("NaN", T, 1)],                                           # not a finite amount
    [{k: v for k, v in _row("1", T, 1).items() if k != "tranId"}],  # no transaction id
])
def test_an_answer_that_cannot_be_read_raises(page):
    with pytest.raises(RuntimeError, match="income"):
        _adapter(Pages(page)).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW)


def test_a_failed_request_propagates():
    fake = FakeBinance()
    fake.income_error = True
    with pytest.raises(BinanceRetryableError):
        _adapter(fake).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW)


def test_funding_older_than_binance_s_income_history_cannot_be_read_whole_and_raises():
    """Binance: "Income history only contains data for the last three months" (at least 89 days). Funding since a
    first fill older than that would come back short without an error, so the total is unknown; 88 days are read."""
    fake = FakeBinance()
    fake.add_income("BTCUSDT", "-1", T)
    assert _adapter(fake).get_funding_income("BTCUSDT", since_ms=T, now_ms=T + 88 * DAY) == D("-1")
    with pytest.raises(RuntimeError, match="three months"):
        _adapter(fake).get_funding_income("BTCUSDT", since_ms=T, now_ms=T + 88 * DAY + 1)
    assert len(fake.income_requests) == 1                         # refused before any request


# -- after the audit of 2026-11-16 (finding 5): the end of the window is sent, and the page cap is pinned ---------
def test_every_page_asks_up_to_the_runtime_s_time_and_a_later_row_does_not_count():
    fake = FakeBinance()
    for i in range(1500):
        fake.add_income("BTCUSDT", "-0.01", T + i * MINUTE)
    fake.add_income("BTCUSDT", "-7", NOW)                          # at the end time: counted (inclusive)
    fake.add_income("BTCUSDT", "-900", NOW + 1)                    # after it: not this request's
    assert _adapter(fake).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW) == D("-22.00")
    assert [(r["startTime"], r["endTime"]) for r in fake.income_requests] == [(T, NOW), (T + 999 * MINUTE, NOW)]
    client = Pages([_row("1", T, 1)])
    _adapter(client).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW)
    assert client.calls == [("GET", "/fapi/v1/income", {"symbol": "BTCUSDT", "incomeType": "FUNDING_FEE",
                                                        "startTime": T, "endTime": NOW, "limit": 1000})]


def test_a_history_of_five_pages_is_read_whole():
    fake = FakeBinance()
    for i in range(4100):
        fake.add_income("BTCUSDT", "-0.01", T + i * MINUTE)
    assert _adapter(fake).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW) == D("-41.00")
    assert len(fake.income_requests) == 5


def test_a_history_beyond_the_page_cap_raises_instead_of_returning_a_part(monkeypatch):
    from trading.adapters.binance import adapter
    assert adapter.MAX_INCOME_PAGES == 100
    monkeypatch.setattr(adapter, "MAX_INCOME_PAGES", 2)
    fake = FakeBinance()
    for i in range(2500):                                          # three pages
        fake.add_income("BTCUSDT", "-0.01", T + i * MINUTE)
    with pytest.raises(RuntimeError, match="more than 2 pages"):
        _adapter(fake).get_funding_income("BTCUSDT", since_ms=T, now_ms=NOW)
    assert len(fake.income_requests) == 2
