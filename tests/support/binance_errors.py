"""Test helper: the exception ``BinanceRestClient`` raises for a given HTTP answer (task 1, 2026-11-17).

The answer goes through the real REST mapping (``request_signed`` -> ``_send`` -> ``_raise_http``) with ``urlopen``
patched, so a test sees exactly the exception the adapter would get from Binance. Nothing touches the network.
"""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from trading.adapters.binance.rest import BinanceRestClient, BinanceRestConfig, BinanceRestError

URL = "https://fapi.binance.com/fapi/v1/order"


def _raised(side_effect: BaseException) -> BinanceRestError:
    client = BinanceRestClient(BinanceRestConfig(base_url="https://fapi.binance.com", api_key="k", api_secret="s"))
    with patch("trading.adapters.binance.rest.urlopen", side_effect=side_effect):
        try:
            client.request_signed(method="POST", path="/fapi/v1/order", params={"symbol": "BTCUSDT"})
        except BinanceRestError as exc:
            return exc
    raise AssertionError("the REST client raised nothing")


def rest_error(status: int, code: int | None = None, *, body: str | None = None) -> BinanceRestError:
    """The client's exception for HTTP ``status`` with a Binance error ``code`` (or ``body`` as sent)."""
    if body is None:
        body = json.dumps({"code": code, "msg": "test"}) if code is not None else "<html>error</html>"
    return _raised(HTTPError(URL, status, "test", {}, BytesIO(body.encode())))  # type: ignore[arg-type]


def network_error() -> BinanceRestError:
    """The client's exception when the connection fails (no HTTP answer at all)."""
    return _raised(URLError("timed out"))


__all__ = ["rest_error", "network_error"]
