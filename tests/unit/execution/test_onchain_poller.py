"""Tests for OnchainPoller."""
import json
import logging
from io import BytesIO
from unittest.mock import patch, MagicMock

logger = logging.getLogger(__name__)

from execution.adapters.onchain_poller import OnchainPoller  # noqa: E402


def test_get_current_before_fetch():
    poller = OnchainPoller(asset="btc", interval_sec=3600)
    assert poller.get_current() is None


def _mock_urlopen(response_data):
    """Helper: return a context manager that yields a JSON response."""
    def _open(req, **kwargs):
        body = json.dumps(response_data).encode()
        cm = MagicMock()
        cm.__enter__ = lambda s: BytesIO(body)
        cm.__exit__ = lambda s, *a: None
        return cm
    return _open


def test_fetch_parses_response():
    api_response = {
        "data": [{
            "asset": "btc",
            "time": "2024-01-15T00:00:00.000000000Z",
            "FlowInExUSD": "150000000.5",
            "FlowOutExUSD": "120000000.3",
            "SplyExNtv": "2300000.123",
            "AdrActCnt": "850000",
            "TxTfrCnt": "320000",
            "HashRate": "550000000000000000000",
        }]
    }

    poller = OnchainPoller(asset="btc")
    with patch("execution.adapters.onchain_poller.urllib.request.urlopen",
               side_effect=_mock_urlopen(api_response)):
        poller._fetch()

    result = poller.get_current()
    assert result is not None
    assert len(result) == 6
    assert abs(result["FlowInExUSD"] - 150000000.5) < 0.01
    assert abs(result["FlowOutExUSD"] - 120000000.3) < 0.01
    assert abs(result["SplyExNtv"] - 2300000.123) < 0.01
    assert abs(result["AdrActCnt"] - 850000) < 0.01
    assert abs(result["TxTfrCnt"] - 320000) < 0.01
    assert result["HashRate"] > 0


def test_fetch_handles_missing_metrics():
    api_response = {
        "data": [{
            "asset": "btc",
            "time": "2024-01-15T00:00:00.000000000Z",
            "FlowInExUSD": "100000",
            "FlowOutExUSD": "",
            "SplyExNtv": "2300000",
        }]
    }

    poller = OnchainPoller(asset="btc")
    with patch("execution.adapters.onchain_poller.urllib.request.urlopen",
               side_effect=_mock_urlopen(api_response)):
        poller._fetch()

    result = poller.get_current()
    assert result is not None
    assert "FlowInExUSD" in result
    assert "FlowOutExUSD" not in result  # empty string → skipped
    assert "AdrActCnt" not in result     # missing key → skipped
    assert len(result) == 2


def test_fetch_handles_empty_data():
    api_response = {"data": []}

    poller = OnchainPoller(asset="btc")
    with patch("execution.adapters.onchain_poller.urllib.request.urlopen",
               side_effect=_mock_urlopen(api_response)):
        poller._fetch()

    assert poller.get_current() is None


def test_fetch_error_returns_none():
    poller = OnchainPoller(asset="btc")
    with patch("execution.adapters.onchain_poller.urllib.request.urlopen",
               side_effect=Exception("network error")):
        try:
            poller._fetch()
        except Exception as e:
            logger.debug("Expected fetch error in test: %s", e)

    assert poller.get_current() is None


def test_start_stop_lifecycle():
    poller = OnchainPoller(asset="btc", interval_sec=9999)
    assert poller._running is False
    with patch("execution.adapters.onchain_poller.urllib.request.urlopen",
               side_effect=Exception("mocked")):
        poller.start()
        assert poller._running is True
        assert poller._thread is not None
        poller.stop()
    assert poller._running is False


def test_get_current_returns_copy():
    """get_current() should return a copy, not the internal dict."""
    api_response = {
        "data": [{
            "asset": "btc",
            "time": "2024-01-15T00:00:00.000000000Z",
            "FlowInExUSD": "100",
            "FlowOutExUSD": "50",
            "SplyExNtv": "2300000",
            "AdrActCnt": "850000",
            "TxTfrCnt": "320000",
            "HashRate": "550000000000000000000",
        }]
    }

    poller = OnchainPoller(asset="btc")
    with patch("execution.adapters.onchain_poller.urllib.request.urlopen",
               side_effect=_mock_urlopen(api_response)):
        poller._fetch()

    r1 = poller.get_current()
    r2 = poller.get_current()
    assert r1 is not r2  # different dict objects
    assert r1 == r2      # same content
