"""Strict Bybit account cash-flow reads; all responses are local fixtures."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

import pytest

from execution.adapters.bybit.adapter import BybitAdapter
from execution.adapters.bybit.config import BybitConfig


class Client:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, path, params):
        self.calls.append((path, params))
        if not self.responses:
            raise AssertionError("unexpected request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def adapter(*responses):
    value = BybitAdapter(BybitConfig.demo(api_key="fixture", api_secret="fixture"))
    value._client = Client(*responses)
    return value


def page(*rows, cursor=""):
    return {"retCode": 0, "result": {"list": list(rows), "nextPageCursor": cursor}}


def transaction(**overrides):
    row = {
        "id": "tx-1", "currency": "USDT", "type": "TRADE", "category": "linear",
        "transactionTime": "1700000000100", "change": "-1.25", "cashFlow": "0",
        "funding": "0", "fee": "1.25", "cashBalance": "998.75", "bonusChange": "0",
    }
    row.update(overrides)
    return row


def wallet(*coins, account_type="UNIFIED"):
    return {
        "retCode": 0,
        "result": {"list": [{"accountType": account_type, "totalEquity": "999999", "coin": list(coins)}]},
    }


def coin(currency="USDT", balance="1000", **overrides):
    row = {
        "coin": currency, "walletBalance": balance, "borrowAmount": "0",
        "spotBorrow": "0", "accruedInterest": "0", "bonus": "0",
    }
    row.update(overrides)
    return row


def test_transaction_log_reads_every_page_without_category_filter():
    second = transaction(id="tx-2", type="TRANSFER_IN", category="", transactionTime="1700000000200",
                         change="5", cashFlow="5", fee="0", cashBalance="1003.75")
    venue = adapter(page(transaction(), cursor="next"), page(second))

    result = venue.get_transaction_log(start_ms=1700000000000, end_ms=1700000000300)

    assert [row["id"] for row in result] == ["tx-1", "tx-2"]
    assert result[0]["fee"] == "1.25"
    assert venue._client.calls == [
        ("/v5/account/transaction-log", {
            "accountType": "UNIFIED", "startTime": "1700000000000",
            "endTime": "1700000000300", "limit": "50",
        }),
        ("/v5/account/transaction-log", {
            "accountType": "UNIFIED", "startTime": "1700000000000",
            "endTime": "1700000000300", "limit": "50", "cursor": "next",
        }),
    ]


def test_identical_duplicate_is_deduplicated_but_conflicting_identity_fails():
    row = transaction()
    venue = adapter(page(row, deepcopy(row)))
    assert len(venue.get_transaction_log(start_ms=1700000000000, end_ms=1700000000300)) == 1

    changed = transaction(fee="1.24", change="-1.24")
    venue = adapter(page(row, changed))
    with pytest.raises(RuntimeError, match="conflicting"):
        venue.get_transaction_log(start_ms=1700000000000, end_ms=1700000000300)

    changed_attribution = transaction(symbol="BTCUSDT")
    venue = adapter(page(row, changed_attribution))
    with pytest.raises(RuntimeError, match="conflicting"):
        venue.get_transaction_log(start_ms=1700000000000, end_ms=1700000000300)


@pytest.mark.parametrize("overrides,match", [
    ({"change": "-1.24"}, "equation"),
    ({"currency": "BTC"}, "unsupported"),
    ({"type": "BONUS"}, "unsupported"),
    ({"category": "inverse"}, "category"),
    ({"bonusChange": "1"}, "bonus"),
    ({"transSubType": "borrow"}, "subtype"),
    ({"extraFees": "1"}, "extra fees"),
    ({"borrowAmount": "1"}, "borrowing"),
    ({"fee": "NaN"}, "Invalid"),
    ({"transactionTime": "1699999999999"}, "outside"),
    ({"id": " tx-1"}, "unsupported"),
])
def test_transaction_log_rejects_unsafe_facts(overrides, match):
    venue = adapter(page(transaction(**overrides)))
    with pytest.raises(RuntimeError, match=match):
        venue.get_transaction_log(start_ms=1700000000000, end_ms=1700000000300)


def test_transaction_window_is_explicit_ordered_and_bounded():
    venue = adapter()
    for start, end in ((-1, 1), (2, 1), (0, 604800001), (True, 2)):
        with pytest.raises(ValueError, match="7 days"):
            venue.get_transaction_log(start_ms=start, end_ms=end)
    assert venue._client.calls == []


def test_failed_or_incomplete_pagination_never_returns_partial_cashflows():
    venue = adapter(page(transaction(), cursor="again"), {"retCode": 10001, "retMsg": "busy"})
    with pytest.raises(RuntimeError, match="query failed"):
        venue.get_transaction_log(start_ms=1700000000000, end_ms=1700000000300)

    venue = adapter(page(transaction(), cursor="same"), page(cursor="same"))
    with pytest.raises(RuntimeError, match="Repeated"):
        venue.get_transaction_log(start_ms=1700000000000, end_ms=1700000000300)


def test_wallet_snapshot_uses_coin_wallet_balance_not_total_equity(monkeypatch):
    venue = adapter(wallet(coin(balance="1000.25"), coin("BTC", "0")))
    monkeypatch.setattr("execution.adapters.bybit.adapter._time.time", lambda: 123.5)

    assert venue.get_wallet_snapshot() == {
        "observed_at": 123.5, "balances": {"USDT": "1000.25"},
    }
    assert venue._client.calls == [
        ("/v5/account/wallet-balance", {"accountType": "UNIFIED"}),
    ]


@pytest.mark.parametrize("response,match", [
    ({"retCode": 10001}, "failed"),
    ({"retCode": 0, "result": {"list": []}}, "Malformed"),
    (wallet(coin(), account_type="CONTRACT"), "account type"),
    (wallet(coin("BTC", "1"), coin()), "nonzero currency"),
    (wallet(coin(borrowAmount="1")), "borrowAmount"),
    (wallet(coin(bonus="1")), "bonus"),
    (wallet(coin(), coin()), "duplicated"),
    (wallet(coin("BTC", "0")), "missing USDT"),
    (wallet(coin(balance="NaN")), "Invalid"),
])
def test_wallet_snapshot_rejects_incomplete_or_unsupported_state(response, match):
    with pytest.raises(RuntimeError, match=match):
        adapter(response).get_wallet_snapshot()


def test_non_unified_configuration_is_rejected_without_request():
    venue = BybitAdapter(BybitConfig(api_key="fixture", api_secret="fixture", account_type="CONTRACT"))
    venue._client = Client()
    with pytest.raises(RuntimeError, match="UNIFIED"):
        venue.get_wallet_snapshot()
    with pytest.raises(ValueError, match="UNIFIED"):
        venue.get_transaction_log(start_ms=0, end_ms=1)
    assert venue._client.calls == []


def test_funding_and_rebate_signs_are_preserved_as_decimal_strings():
    row = transaction(type="SETTLEMENT", change="2.5", cashFlow="0", funding="2", fee="-0.5",
                      cashBalance="1002.5")
    result = adapter(page(row)).get_transaction_log(
        start_ms=1700000000000, end_ms=1700000000300,
    )
    assert result[0]["funding"] == "2"
    assert result[0]["fee"] == "-0.5"
    assert Decimal(result[0]["change"]) == Decimal("2.5")
