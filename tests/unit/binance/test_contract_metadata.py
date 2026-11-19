"""Plan D1: the adapter translates exchangeInfo's contract words and the margin tiers (GET /fapi/v1/leverageBracket)
into the domain's records. The words checked against the live exchangeInfo on 2026-11-19: contract types PERPETUAL and
TRADIFI_PERPETUAL; statuses TRADING, SETTLING, PENDING_TRADING, TRADING_HALT; 2100-12-25 as "no delisting"."""
from __future__ import annotations

from decimal import Decimal as D

from tests.unit.binance.fake_venue import FakeBinance
from trading.adapters.binance import adapter as binance
from trading.adapters.binance.config import BinanceConfig
from trading.domain.contracts import Contract


def _adapter(fake):
    a = binance.BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert a.connect(symbols=("BTCUSDT", "ETHUSDT"))
    return a


def test_contract_words_are_translated_and_a_missing_delivery_date_counts_as_a_delisting():
    fake = FakeBinance()
    fake.listed = ("BTCUSDT", "ETHUSDT", "XAUUSDT", "OMGUSDT", "NEWUSDT", "TSLAUSDT", "BTCUSDT_261225", "OLDUSDT")
    fake.contract = {"XAUUSDT": {"contractType": "TRADIFI_PERPETUAL"},
                     "OMGUSDT": {"status": "SETTLING", "deliveryDate": 1_738_314_000_000},
                     "NEWUSDT": {"status": "PENDING_TRADING"},
                     "TSLAUSDT": {"contractType": "TRADIFI_PERPETUAL", "status": "TRADING_HALT"},
                     "BTCUSDT_261225": {"contractType": "CURRENT_QUARTER", "deliveryDate": 1_798_185_600_000},
                     "OLDUSDT": {"deliveryDate": None}}
    c = _adapter(fake).contracts()
    assert c["BTCUSDT"] == Contract("BTCUSDT", "crypto", "open", "USDT", None)
    assert c["XAUUSDT"] == Contract("XAUUSDT", "tradfi", "open", "USDT", None)
    assert c["OMGUSDT"] == Contract("OMGUSDT", "crypto", "settling", "USDT", 1_738_314_000_000)
    assert c["NEWUSDT"].state == "pending" and c["TSLAUSDT"].state == "halted" and c["TSLAUSDT"].kind == "tradfi"
    assert c["BTCUSDT_261225"].kind == "other" and c["BTCUSDT_261225"].delisting_at == 1_798_185_600_000
    assert c["OLDUSDT"].delisting_at == 0                                   # unknown date: fail closed


def test_margin_tiers_are_read_signed_and_sorted_by_notional():
    fake = FakeBinance()
    fake.brackets["ETHUSDT"] = [
        {"bracket": 2, "initialLeverage": 20, "notionalCap": 25000, "notionalFloor": 5000, "maintMarginRatio": 0.025,
         "cum": 75.0},
        {"bracket": 1, "initialLeverage": 50, "notionalCap": 5000, "notionalFloor": 0, "maintMarginRatio": 0.01,
         "cum": 0.0}]
    a = _adapter(fake)
    calls = []
    original = fake.request_signed

    def spy(*, method, path, params=None):
        calls.append((method, path, dict(params or {})))
        return original(method=method, path=path, params=params)
    fake.request_signed = spy
    tiers = a.get_margin_tiers()
    assert calls == [("GET", "/fapi/v1/leverageBracket", {})]
    eth = tiers["ETHUSDT"]
    assert [b.floor for b in eth.brackets] == [D("0"), D("5000")] and eth.max_leverage() == 50
    assert eth.at(D("6000")).maintenance_rate == D("0.025")
    assert tiers["BTCUSDT"].max_leverage() == 125


def test_a_single_symbols_answer_is_read_too():
    fake = FakeBinance()
    a = _adapter(fake)
    fake.request_signed = lambda **kw: {"symbol": "ETHUSDT", "brackets": [
        {"bracket": 1, "initialLeverage": 75, "notionalCap": 10000, "notionalFloor": 0, "maintMarginRatio": "0.0065",
         "cum": 0}]}
    assert a.get_margin_tiers()["ETHUSDT"].brackets[0].maintenance_rate == D("0.0065")


def test_the_quote_asset_and_an_unknown_status_are_read_as_they_are():
    fake = FakeBinance()
    fake.listed = ("BTCUSDT", "ETHUSDT", "BTCUSDC", "ODDUSDT")
    fake.contract = {"BTCUSDC": {"quoteAsset": "USDC"}, "ODDUSDT": {"status": "BREAK"}}
    c = _adapter(fake).contracts()
    assert c["BTCUSDC"].quote == "USDC" and c["BTCUSDT"].quote == "USDT"
    assert c["ODDUSDT"].state == "other"


def test_one_malformed_contract_does_not_stop_the_connection_and_is_unfit():
    fake = FakeBinance()
    fake.listed = ("BTCUSDT", "ETHUSDT", "ODDUSDT")
    fake.contract = {"ODDUSDT": {"deliveryDate": "2100-12-25"}}
    c = _adapter(fake).contracts()                                         # connect() is asserted in _adapter
    assert c["ODDUSDT"].delisting_at == 0 and c["BTCUSDT"].delisting_at is None


def test_an_infinite_delivery_date_does_not_stop_the_connection_either():
    fake = FakeBinance()
    fake.listed = ("BTCUSDT", "ETHUSDT", "ODDUSDT")
    fake.contract = {"ODDUSDT": {"deliveryDate": float("inf")}}
    assert _adapter(fake).contracts()["ODDUSDT"].delisting_at == 0
