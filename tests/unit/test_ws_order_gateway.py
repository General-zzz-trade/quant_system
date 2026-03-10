"""Tests for RustWsOrderGateway — signed JSON-RPC message building."""
import json
import pytest

_quant_hotpath = pytest.importorskip("_quant_hotpath")
from _quant_hotpath import RustWsOrderGateway


@pytest.fixture
def gateway():
    return RustWsOrderGateway(
        "test_api_key_123",
        "test_secret_456",
        recv_window=5000,
    )


class TestBuildOrderMessage:
    def test_market_order_basic(self, gateway):
        msg, req_id = gateway.build_order_message(
            "BTCUSDT", "BUY", "MARKET", quantity="0.001",
        )
        assert req_id.startswith("ord_")
        parsed = json.loads(msg)
        assert parsed["method"] == "order.place"
        params = parsed["params"]
        assert params["symbol"] == "BTCUSDT"
        assert params["side"] == "BUY"
        assert params["type"] == "MARKET"
        assert params["quantity"] == "0.001"
        assert params["apiKey"] == "test_api_key_123"
        assert "signature" in params
        assert "timestamp" in params
        assert len(params["signature"]) == 64  # SHA256 hex

    def test_limit_order(self, gateway):
        msg, req_id = gateway.build_order_message(
            "ETHUSDT", "SELL", "LIMIT",
            quantity="0.5", price="3000.00", time_in_force="GTC",
        )
        parsed = json.loads(msg)
        params = parsed["params"]
        assert params["price"] == "3000.00"
        assert params["timeInForce"] == "GTC"

    def test_reduce_only(self, gateway):
        msg, _ = gateway.build_order_message(
            "BTCUSDT", "SELL", "MARKET",
            quantity="0.001", reduce_only=True,
        )
        parsed = json.loads(msg)
        assert parsed["params"]["reduceOnly"] == "true"

    def test_client_order_id(self, gateway):
        msg, _ = gateway.build_order_message(
            "BTCUSDT", "BUY", "MARKET",
            quantity="0.001", new_client_order_id="my_id_123",
        )
        parsed = json.loads(msg)
        assert parsed["params"]["newClientOrderId"] == "my_id_123"

    def test_unique_request_ids(self, gateway):
        _, id1 = gateway.build_order_message("BTCUSDT", "BUY", "MARKET", quantity="0.001")
        _, id2 = gateway.build_order_message("BTCUSDT", "BUY", "MARKET", quantity="0.001")
        assert id1 != id2

    def test_signature_changes_with_params(self, gateway):
        msg1, _ = gateway.build_order_message("BTCUSDT", "BUY", "MARKET", quantity="0.001")
        msg2, _ = gateway.build_order_message("BTCUSDT", "SELL", "MARKET", quantity="0.002")
        sig1 = json.loads(msg1)["params"]["signature"]
        sig2 = json.loads(msg2)["params"]["signature"]
        assert sig1 != sig2


class TestBuildCancelMessage:
    def test_cancel_by_order_id(self, gateway):
        msg, req_id = gateway.build_cancel_message("BTCUSDT", order_id=12345)
        assert req_id.startswith("cxl_")
        parsed = json.loads(msg)
        assert parsed["method"] == "order.cancel"
        assert parsed["params"]["orderId"] == "12345"
        assert "signature" in parsed["params"]

    def test_cancel_by_client_id(self, gateway):
        msg, _ = gateway.build_cancel_message("BTCUSDT", orig_client_order_id="my_cxl")
        parsed = json.loads(msg)
        assert parsed["params"]["origClientOrderId"] == "my_cxl"


class TestBuildQueryMessage:
    def test_query_by_order_id(self, gateway):
        msg, req_id = gateway.build_query_message("BTCUSDT", order_id=12345)
        assert req_id.startswith("qry_")
        parsed = json.loads(msg)
        assert parsed["method"] == "order.status"
        assert parsed["params"]["orderId"] == "12345"

    def test_query_by_client_id(self, gateway):
        msg, _ = gateway.build_query_message("ETHUSDT", orig_client_order_id="ord_1")
        parsed = json.loads(msg)
        assert parsed["params"]["origClientOrderId"] == "ord_1"


class TestProcessTickFull:
    """Test that process_tick_full returns features_dict."""

    def test_features_dict_populated(self):
        tp = _quant_hotpath.RustTickProcessor.create(
            symbols=["BTCUSDT"],
            currency="USDT",
            balance=10000.0,
            model_paths=[_find_model_path()],
        )
        tp.configure_symbol("BTCUSDT", min_hold=0, deadzone=0.5)

        # Push enough bars for feature warmup
        for i in range(70):
            price = 50000.0 + i * 10
            tp.process_tick("BTCUSDT", price, 100.0, price + 5, price - 5, price - 1, i, ts=None)

        result = tp.process_tick_full(
            "BTCUSDT", 50700.0, 100.0, 50705.0, 50695.0, 50699.0, 70,
            warmup_done=True,
        )

        # features_dict should be populated
        assert result.features_dict is not None
        assert isinstance(result.features_dict, dict)
        assert "close" in result.features_dict
        assert result.features_dict["close"] == 50700.0
        assert "volume" in result.features_dict
        assert "ml_score" in result.features_dict

    def test_features_dict_no_warmup(self):
        tp = _quant_hotpath.RustTickProcessor.create(
            symbols=["BTCUSDT"],
            currency="USDT",
            balance=10000.0,
            model_paths=[_find_model_path()],
        )
        tp.configure_symbol("BTCUSDT", min_hold=0, deadzone=0.5)

        for i in range(70):
            price = 50000.0 + i * 10
            tp.process_tick("BTCUSDT", price, 100.0, price + 5, price - 5, price - 1, i)

        result = tp.process_tick_full(
            "BTCUSDT", 50700.0, 100.0, 50705.0, 50695.0, 50699.0, 70,
            warmup_done=False,
        )
        # No ml_score when warmup not done
        assert result.features_dict is not None
        assert "close" in result.features_dict
        assert "ml_score" not in result.features_dict


def _find_model_path():
    """Find a model JSON file for testing."""
    import glob
    # lgbm_v8.json has the correct format for RustTreePredictor
    paths = glob.glob("models_v8/BTCUSDT_gate_v2/lgbm_v8.json")
    if not paths:
        pytest.skip("No model JSON files found")
    return paths[0]
