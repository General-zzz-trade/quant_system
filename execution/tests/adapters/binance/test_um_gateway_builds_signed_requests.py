# execution/tests/adapters/binance/test_um_gateway_builds_signed_requests.py
from __future__ import annotations

import types
import pytest

from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
from execution.adapters.binance.order_gateway_um import BinanceUmFuturesOrderGateway


class _FakeRest(BinanceRestClient):
    def __init__(self) -> None:
        self.calls = []

    def request_signed(self, *, method: str, path: str, params=None):
        self.calls.append((method, path, dict(params or {})))
        return {"ok": True}


def test_submit_maps_required_fields_and_binds_client_id():
    fake = _FakeRest()
    gw = BinanceUmFuturesOrderGateway(rest=fake)

    cmd = types.SimpleNamespace(
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=1.0,
        price=30000.0,
        tif="GTC",
        request_id="rid-001",
        reduce_only=True,
    )

    gw.submit_order(cmd)
    method, path, p = fake.calls[-1]
    assert method == "POST"
    assert path == "/fapi/v1/order"
    assert p["symbol"] == "BTCUSDT"
    assert p["side"] == "BUY"
    assert p["type"] == "LIMIT"
    assert p["quantity"] == 1.0
    assert p["price"] == 30000.0
    assert p["timeInForce"] == "GTC"
    assert p["newClientOrderId"] == "rid-001"
    assert p["reduceOnly"] is True


def test_cancel_requires_orderid_or_orig_client_order_id():
    fake = _FakeRest()
    gw = BinanceUmFuturesOrderGateway(rest=fake)

    with pytest.raises(ValueError):
        gw.cancel_order(types.SimpleNamespace(symbol="BTCUSDT"))

    gw.cancel_order(types.SimpleNamespace(symbol="BTCUSDT", order_id=123))
    _, _, p1 = fake.calls[-1]
    assert p1["orderId"] == 123

    gw.cancel_order(types.SimpleNamespace(symbol="BTCUSDT", client_order_id="cid-9"))
    _, _, p2 = fake.calls[-1]
    assert p2["origClientOrderId"] == "cid-9"
