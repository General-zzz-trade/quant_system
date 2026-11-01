"""Real OKX adapter + event integration; REST transport is mocked, never live."""
from decimal import Decimal as D
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from engine.execution_bridge import ExecutionBridge
from event.header import EventHeader
from event.types import EventType, OrderEvent
from execution.adapters.okx.adapter import OkxAdapter
from execution.adapters.okx.config import OkxConfig
from execution.adapters.okx.execution_adapter import OkxExecutionAdapter
from execution.adapters.okx.symbol_map import InstrumentMeta
from execution.order_utils import reliable_close_position


@pytest.fixture
def adapter():
    adapter = OkxAdapter(OkxConfig("test", "test", "test", simulated=True, max_order_notional_usd=1000))
    adapter._instruments["BTCUSDT"] = InstrumentMeta(
        "BTC-USDT-SWAP", D("0.01"), "BTC", D("0.01"), D("0.01"), D("0.1"), 100,
    )
    adapter._client = Mock()
    adapter._client.request_public.return_value = {"data": [{"last": "100", "bidPx": "99", "askPx": "101"}]}
    return adapter


def test_wire_contract_size_client_id_and_reduce_only(adapter):
    adapter._client.request_signed.return_value = {"data": [{"ordId": "o1", "sCode": "0"}]}
    adapter.send_market_order("BTCUSDT", "sell", 0.05, reduce_only=True, client_order_id="qs123")
    body = adapter._client.request_signed.call_args.kwargs["body"]
    assert D(body["sz"]) == D(5)
    assert body["reduceOnly"] == "true"
    assert body["clOrdId"] == "qs123"
    adapter.send_limit_order("BTCUSDT", "buy", 0.05, 100, post_only=True, client_order_id="qs456")
    body = adapter._client.request_signed.call_args.kwargs["body"]
    assert body["ordType"] == "post_only"
    assert body["clOrdId"] == "qs456"


def test_order_state_converts_cumulative_contract_fills(adapter):
    adapter._client.request_signed.return_value = {"data": [{
        "instId": "BTC-USDT-SWAP", "ordId": "o1", "clOrdId": "qs123", "state": "canceled", "accFillSz": "2.5",
    }]}
    state = adapter.get_order("BTCUSDT", client_order_id="qs123")
    assert state.filled_qty == D("0.025")
    assert state.state == "canceled"
    assert adapter._client.request_signed.call_args.kwargs["params"]["clOrdId"] == "qs123"


def test_fill_request_filters_order_id(adapter):
    adapter._client.request_signed.return_value = {"data": []}
    adapter.get_recent_fills("BTCUSDT", order_id="o1", limit=100)
    assert adapter._client.request_signed.call_args.kwargs["params"]["ordId"] == "o1"


def test_position_failure_is_not_an_empty_position_snapshot(adapter):
    adapter._client.request_signed.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        adapter.get_positions("BTCUSDT")


def test_equity_is_total_equity_not_free_balance(adapter):
    adapter._client.request_signed.return_value = {"data": [{"totalEq": "97", "details": [{"availBal": "10"}]}]}
    assert adapter.get_account_equity() == 97


def test_limit_notional_cap_and_missing_market_quote_block_before_post(adapter):
    assert adapter.send_limit_order("BTCUSDT", "buy", 100, 100)["status"] == "error"
    adapter._client.request_public.side_effect = TimeoutError
    assert adapter.send_market_order("BTCUSDT", "buy", 1)["status"] == "error"
    adapter._client.request_signed.assert_not_called()


def test_default_connect_does_not_change_leverage(adapter):
    def public(**kwargs):
        inst_id = kwargs.get("params", {}).get("instId")
        return {"data": [{"instId": inst_id, "ctVal": "0.01", "ctValCcy": "BTC", "lotSz": "0.01",
                          "minSz": "0.01", "tickSz": "0.1", "lever": "100"}]} if inst_id else {}
    adapter._client.request_public.side_effect = public
    adapter._client.request_signed.return_value = {"data": []}
    assert adapter.connect()
    assert all(call.kwargs["method"] == "GET" for call in adapter._client.request_signed.call_args_list)


def test_event_bridge_emits_actual_partial_fill_and_stable_id(adapter, tmp_path, monkeypatch):
    monkeypatch.setenv("OKX_LIMIT_FILL_TIMEOUT_S", "0")
    client_id = ""

    def signed(**kwargs):
        nonlocal client_id
        path = kwargs["path"]
        if path.endswith("/order") and kwargs["method"] == "POST":
            client_id = kwargs["body"]["clOrdId"]
            return {"data": [{"ordId": "o1", "clOrdId": client_id, "sCode": "0"}]}
        if path.endswith("/order"):
            return {"data": [{"instId": "BTC-USDT-SWAP", "ordId": "o1", "clOrdId": client_id,
                              "state": "canceled", "accFillSz": "2"}]}
        if path.endswith("/fills"):
            return {"data": [{"instId": "BTC-USDT-SWAP", "ordId": "o1", "tradeId": "t1", "fillSz": "2",
                              "fillPx": "100", "side": "buy", "ts": "1000"}]}
        raise AssertionError(f"Unexpected REST call: {path}")

    adapter._client.request_signed.side_effect = signed
    execution = OkxExecutionAdapter(adapter, journal_path=tmp_path / "orders.sqlite", scope="test-demo")
    event = OrderEvent(header=EventHeader.new_root(event_type=EventType.ORDER, version=1, source="test"),
                       order_id="intent-1", intent_id="strategy-1", symbol="BTCUSDT", side="buy", qty=D("0.05"),
                       price=D(100))
    emitted = []
    bridge = ExecutionBridge(execution, emitted.append, max_retries=0)
    bridge.handle_event(event)
    assert len(emitted) == 1
    assert emitted[0].qty == D("0.02")
    assert emitted[0].price == D(100)
    assert emitted[0].fill_id == "test-demo:BTCUSDT:o1:t1"
    assert emitted[0].order_id == "intent-1"
    bridge.handle_event(event)
    assert len(emitted) == 1


def test_circuit_breaker_does_not_disable_reduction():
    import time
    adapter = Mock()
    adapter.send_order.return_value = ()
    bridge = ExecutionBridge(adapter, lambda _: None, max_retries=0)
    bridge._circuit_opened_at = time.monotonic()
    bridge.handle_event(NS(qty=D(0)))
    adapter.send_order.assert_called_once()


def test_failed_close_verification_stays_unverified():
    adapter = Mock()
    adapter.close_position.return_value = {"status": "submitted", "orderId": "o1"}
    adapter.get_positions.side_effect = TimeoutError
    assert reliable_close_position(adapter, "BTCUSDT")["verified"] is False
