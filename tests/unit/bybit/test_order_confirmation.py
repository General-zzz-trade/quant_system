"""Durable order confirmation uses mocked REST facts, never an exchange account."""
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from execution.adapters.bybit.adapter import BybitAdapter
from execution.adapters.bybit.config import BybitConfig


@pytest.fixture
def adapter():
    instance = BybitAdapter(BybitConfig())
    instance._client = MagicMock()
    return instance


def page(items=(), cursor=""):
    return {"retCode": 0, "result": {"list": list(items), "nextPageCursor": cursor}}


def raw_order(**overrides):
    item = {
        "symbol": "BTCUSDT", "orderId": "remote-1", "orderLinkId": "durable-1",
        "orderStatus": "Filled", "side": "Buy", "orderType": "Market",
        "qty": "0.003", "cumExecQty": "0.003", "avgPrice": "70000",
        "createdTime": "1000",
    }
    item.update(overrides)
    return item


def raw_fill(**overrides):
    item = {
        "symbol": "BTCUSDT", "orderId": "remote-1", "execId": "trade-1",
        "execType": "Trade", "side": "Buy", "execQty": "0.001",
        "execPrice": "70000", "execFee": "0.01", "feeCurrency": "USDT",
        "execTime": "2000", "isMaker": "false",
    }
    item.update(overrides)
    return item


def test_durable_market_uses_callers_identity_without_legacy_ttl(adapter):
    adapter._pending_orders["old-legacy-order"] = 1
    adapter._check_order_ttls = MagicMock(side_effect=AssertionError("legacy TTL invoked"))
    adapter._client.post.return_value = {
        "retCode": 0, "result": {"orderId": "remote-1", "orderLinkId": "durable-1"},
    }
    with patch(
        "execution.adapters.bybit.adapter._make_order_link_id",
        side_effect=AssertionError("a replacement ID was generated"),
    ):
        first = adapter.send_market_order("BTCUSDT", "buy", .003, client_order_id="durable-1")
        second = adapter.send_market_order("BTCUSDT", "buy", .003, client_order_id="durable-1")
    assert first == second
    assert adapter._pending_orders == {"old-legacy-order": 1}
    assert all(call.args[1]["orderLinkId"] == "durable-1" for call in adapter._client.post.call_args_list)
    assert all(call.args[0] == "/v5/order/create" for call in adapter._client.post.call_args_list)
    adapter._check_order_ttls.assert_not_called()


def test_legacy_market_keeps_generated_identity_and_ttl(adapter):
    adapter._check_order_ttls = MagicMock()
    adapter._client.post.return_value = {"retCode": 0, "result": {"orderId": "remote-1"}}
    result = adapter.send_market_order("BTCUSDT", "buy", .003)
    adapter._check_order_ttls.assert_called_once_with()
    assert result["orderLinkId"].startswith("qs_BTCUSDT_b_")
    assert result["orderLinkId"] in adapter._pending_orders


@pytest.mark.parametrize("identity", ["", "a" * 37, 123])
def test_invalid_durable_identity_never_submits(adapter, identity):
    with pytest.raises(ValueError):
        adapter.send_market_order("BTCUSDT", "buy", .003, client_order_id=identity)
    adapter._client.post.assert_not_called()


@pytest.mark.parametrize("result", [
    {"orderId": "remote-1", "orderLinkId": "another-intent"},
    {"orderLinkId": "durable-1"},
])
def test_ambiguous_acknowledgement_does_not_confirm_durable_submission(adapter, result):
    adapter._client.post.return_value = {"retCode": 0, "result": result}
    with pytest.raises(RuntimeError):
        adapter.send_market_order("BTCUSDT", "buy", .003, client_order_id="durable-1")
    assert not adapter._pending_orders


def test_get_order_confirms_identity_and_status_from_realtime(adapter):
    adapter._client.get.return_value = page([raw_order()])
    order = adapter.get_order("BTCUSDT", "remote-1", "durable-1")
    assert order.status == "filled"
    assert order.filled_qty == Decimal("0.003")
    adapter._client.get.assert_called_once_with("/v5/order/realtime", {
        "category": "linear", "symbol": "BTCUSDT",
        "orderId": "remote-1", "orderLinkId": "durable-1",
    })


def test_get_order_falls_back_to_history_for_completed_order(adapter):
    adapter._client.get.side_effect = [page(), page([raw_order()])]
    assert adapter.get_order("BTCUSDT", client_order_id="durable-1").order_id == "remote-1"
    assert [call.args[0] for call in adapter._client.get.call_args_list] == [
        "/v5/order/realtime", "/v5/order/history",
    ]


def test_get_order_none_requires_both_successful_empty_endpoints(adapter):
    adapter._client.get.side_effect = [page(), page()]
    assert adapter.get_order("BTCUSDT", client_order_id="durable-1") is None
    assert adapter._client.get.call_count == 2


@pytest.mark.parametrize("responses", [
    [{"retCode": -1, "retMsg": "timeout"}],
    [page(), {"retCode": -1, "retMsg": "history unavailable"}],
    [{"retCode": 0, "result": {}}],
    [page([raw_order(orderId="another-order")])],
    [page([raw_order(orderLinkId="another-intent")])],
    [page([raw_order(symbol="ETHUSDT")])],
    [page([raw_order(orderStatus="UnknownStatus")])],
    [page([raw_order(), raw_order()])],
    [page([raw_order(cumExecQty="NaN")])],
])
def test_failed_or_ambiguous_order_lookup_never_becomes_no_order(adapter, responses):
    adapter._client.get.side_effect = responses
    with pytest.raises(RuntimeError):
        adapter.get_order("BTCUSDT", "remote-1", "durable-1")


def test_order_lookup_requires_identity_before_querying(adapter):
    with pytest.raises(ValueError):
        adapter.get_order("BTCUSDT")
    adapter._client.get.assert_not_called()


def test_order_lookup_reads_later_pages_before_history(adapter):
    adapter._client.get.side_effect = [page(cursor="next"), page([raw_order()])]
    assert adapter.get_order("BTCUSDT", order_id="remote-1").status == "filled"
    assert adapter._client.get.call_args_list[1].args[1]["cursor"] == "next"
    assert all(call.args[0] == "/v5/order/realtime" for call in adapter._client.get.call_args_list)


def test_execution_confirmation_reads_all_pages_filters_and_deduplicates(adapter):
    first = raw_fill()
    second = raw_fill(execId="trade-2", execTime="2100", execQty="0.002")
    adapter._client.get.side_effect = [
        page([first, raw_fill(execId="funding", execType="Funding")], "page-2"),
        page([first, raw_fill(execId="old", execTime="999")], "page-3"),
        page([second]),
    ]
    fills = adapter.get_recent_fills(symbol="BTCUSDT", since_ms=1000, order_id="remote-1", limit=2)
    assert [fill.trade_id for fill in fills] == ["trade-1", "trade-2"]
    assert sum(fill.qty for fill in fills) == Decimal("0.003")
    requests = adapter._client.get.call_args_list
    assert [call.args[1].get("cursor", "") for call in requests] == ["", "page-2", "page-3"]
    assert all(call.args[1]["orderId"] == "remote-1" for call in requests)
    assert all(call.args[1]["startTime"] == "1000" for call in requests)
    assert all(call.args[1]["limit"] == "2" for call in requests)


@pytest.mark.parametrize("responses", [
    [{"retCode": -1, "retMsg": "timeout"}],
    [page([raw_fill()], "more"), {"retCode": -1}],
    [page([raw_fill()], "loop"), page([], "loop")],
    [{"retCode": 0, "result": {"list": "not a list"}}],
    [page([None])],
    [page([raw_fill()], None)],
    [page([raw_fill(orderId="another-order")])],
    [page([raw_fill(symbol="ETHUSDT")])],
    [page([raw_fill(execId="")])],
    [page([raw_fill(execQty="NaN")])],
    [page([raw_fill(execQty="0")])],
    [page([raw_fill(execPrice="0")])],
    [page([raw_fill(execTime="1.5")])],
    [page([raw_fill(execTime="0")])],
    [page([raw_fill(execFee="")])],
    [page([raw_fill(execFee="NaN")])],
    [page([raw_fill(), raw_fill(execQty="0.002")])],
    [page([raw_fill(), raw_fill(feeCurrency="BTC")])],
])
def test_incomplete_or_inconsistent_executions_never_return_partial_facts(adapter, responses):
    adapter._client.get.side_effect = responses
    with pytest.raises(RuntimeError):
        adapter.get_recent_fills(symbol="BTCUSDT", order_id="remote-1")


def test_execution_page_safety_bound_fails_closed(adapter):
    adapter._client.get.side_effect = [page([raw_fill()], "page-2"), page([], "page-3")]
    with patch("execution.adapters.bybit.adapter._MAX_QUERY_PAGES", 2):
        with pytest.raises(RuntimeError, match="Incomplete"):
            adapter.get_recent_fills(symbol="BTCUSDT", order_id="remote-1")


def test_non_trade_activity_is_not_a_fill(adapter):
    adapter._client.get.return_value = page([
        {"execType": "Funding"}, {"execType": "BustTrade"}, {"execType": "Delivery"}, {},
    ])
    assert adapter.get_recent_fills(symbol="BTCUSDT", order_id="remote-1") == ()


@pytest.mark.parametrize("limit,since_ms", [(0, 0), (101, 0), (True, 0), (1, -1), (1, 1.2)])
def test_invalid_execution_query_never_calls_transport(adapter, limit, since_ms):
    with pytest.raises(ValueError):
        adapter.get_recent_fills(order_id="remote-1", limit=limit, since_ms=since_ms)
    adapter._client.get.assert_not_called()


@pytest.mark.parametrize("method", ["get_positions", "get_open_orders"])
def test_legacy_account_reads_keep_empty_on_api_error(adapter, method):
    adapter._client.get.return_value = {"retCode": -1}
    assert getattr(adapter, method)(symbol="BTCUSDT") == ()


def test_legacy_execution_read_keeps_empty_on_initial_api_error(adapter):
    adapter._client.get.return_value = {"retCode": -1}
    assert adapter.get_recent_fills(symbol="BTCUSDT") == ()


@pytest.mark.parametrize("method", ["get_positions", "get_open_orders"])
@pytest.mark.parametrize("responses", [
    [{"retCode": -1, "retMsg": "timeout"}],
    [{"retCode": 0, "result": {}}],
    [page([None])],
    [page([], "more"), {"retCode": -1}],
    [page([], "loop"), page([], "loop")],
    [page([{}])],
])
def test_strict_account_reads_fail_on_unknown_or_incomplete_state(adapter, method, responses):
    adapter._client.get.side_effect = responses
    with pytest.raises(RuntimeError):
        getattr(adapter, method)(symbol="BTCUSDT", strict=True)


@pytest.mark.parametrize("method", ["get_positions", "get_open_orders"])
def test_strict_account_reads_accept_complete_empty_snapshot(adapter, method):
    adapter._client.get.return_value = page()
    assert getattr(adapter, method)(symbol="BTCUSDT", strict=True) == ()


def test_strict_positions_include_later_pages_and_validate_size(adapter):
    adapter._client.get.side_effect = [
        page([{"symbol": "BTCUSDT", "size": "0", "side": ""}], "more"),
        page([{"symbol": "BTCUSDT", "size": "0.003", "side": "Buy"}]),
    ]
    positions = adapter.get_positions("BTCUSDT", strict=True)
    assert len(positions) == 1
    assert positions[0].qty == Decimal("0.003")


@pytest.mark.parametrize("item", [
    {"symbol": "BTCUSDT", "size": "NaN", "side": "Buy"},
    {"symbol": "BTCUSDT", "size": "-1", "side": "Buy"},
    {"symbol": "BTCUSDT", "size": "1", "side": ""},
    {"symbol": "ETHUSDT", "size": "0", "side": ""},
])
def test_strict_positions_reject_malformed_data(adapter, item):
    adapter._client.get.return_value = page([item])
    with pytest.raises(RuntimeError):
        adapter.get_positions("BTCUSDT", strict=True)


def test_strict_open_orders_include_later_pages(adapter):
    adapter._client.get.side_effect = [page(cursor="more"), page([raw_order(orderStatus="New")])]
    orders = adapter.get_open_orders(symbol="BTCUSDT", strict=True)
    assert len(orders) == 1
    assert orders[0].order_id == "remote-1"
    assert orders[0].status == "new"


@pytest.mark.parametrize("method", ["get_positions", "get_open_orders"])
def test_strict_account_wide_reads_specify_settlement_coin(adapter, method):
    adapter._client.get.return_value = page()
    assert getattr(adapter, method)(strict=True) == ()
    assert adapter._client.get.call_args.args[1]["settleCoin"] == "USDT"
