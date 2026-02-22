# execution/tests/adapters/binance/test_venue_client_um_error_mapping.py
from __future__ import annotations

import types
import pytest

from execution.adapters.binance.venue_client_um import BinanceUmFuturesVenueClient
from execution.adapters.binance.rest import BinanceRetryableError, BinanceNonRetryableError
from execution.bridge.execution_bridge import RetryableVenueError, NonRetryableVenueError


class _FakeGateway:
    def __init__(self, *, submit_exc=None, cancel_exc=None):
        self._submit_exc = submit_exc
        self._cancel_exc = cancel_exc

    def submit_order(self, cmd):
        if self._submit_exc:
            raise self._submit_exc
        return {"ok": True}

    def cancel_order(self, cmd):
        if self._cancel_exc:
            raise self._cancel_exc
        return {"ok": True}


def test_submit_retryable_maps_to_retryable_venue_error():
    vc = BinanceUmFuturesVenueClient(gw=_FakeGateway(submit_exc=BinanceRetryableError("429")))
    with pytest.raises(RetryableVenueError):
        vc.submit_order(types.SimpleNamespace())


def test_submit_non_retryable_maps_to_non_retryable_venue_error():
    vc = BinanceUmFuturesVenueClient(gw=_FakeGateway(submit_exc=BinanceNonRetryableError("bad param")))
    with pytest.raises(NonRetryableVenueError):
        vc.submit_order(types.SimpleNamespace())


def test_cancel_mapping():
    vc = BinanceUmFuturesVenueClient(gw=_FakeGateway(cancel_exc=BinanceRetryableError("timeout")))
    with pytest.raises(RetryableVenueError):
        vc.cancel_order(types.SimpleNamespace())
