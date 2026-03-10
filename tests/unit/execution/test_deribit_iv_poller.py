"""Tests for DeribitIVPoller."""
import json
from io import BytesIO
from unittest.mock import patch, MagicMock

from execution.adapters.deribit_iv_poller import DeribitIVPoller


def test_get_current_before_fetch():
    poller = DeribitIVPoller(currency="BTC", interval_sec=300)
    iv, pcr = poller.get_current()
    assert iv is None
    assert pcr is None


def _mock_urlopen(responses):
    """Helper: return a context manager that yields different responses per call."""
    call_count = [0]

    def _open(req, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        body = json.dumps(responses[idx]).encode()
        cm = MagicMock()
        cm.__enter__ = lambda s: BytesIO(body)
        cm.__exit__ = lambda s, *a: None
        return cm

    return _open


def test_fetch_parses_hv_response():
    hv_response = {"result": [[1700000000000, 45.0], [1700003600000, 50.0]]}
    book_response = {"result": [
        {"instrument_name": "BTC-30DEC-P", "open_interest": 100},
        {"instrument_name": "BTC-30DEC-C", "open_interest": 200},
    ]}

    poller = DeribitIVPoller(currency="BTC")
    with patch("execution.adapters.deribit_iv_poller.urllib.request.urlopen",
               side_effect=_mock_urlopen([hv_response, book_response])):
        poller._fetch()

    iv, pcr = poller.get_current()
    assert iv is not None
    assert abs(iv - 0.50) < 1e-6  # 50.0 / 100


def test_fetch_computes_put_call_ratio():
    hv_response = {"result": [[1700000000000, 60.0]]}
    book_response = {"result": [
        {"instrument_name": "BTC-30DEC-P", "open_interest": 300},
        {"instrument_name": "BTC-30DEC-C", "open_interest": 100},
    ]}

    poller = DeribitIVPoller(currency="BTC")
    with patch("execution.adapters.deribit_iv_poller.urllib.request.urlopen",
               side_effect=_mock_urlopen([hv_response, book_response])):
        poller._fetch()

    iv, pcr = poller.get_current()
    assert pcr is not None
    assert abs(pcr - 3.0) < 1e-6  # 300/100 = 3.0


def test_start_stop_lifecycle():
    poller = DeribitIVPoller(currency="BTC", interval_sec=9999)
    assert poller._running is False
    with patch("execution.adapters.deribit_iv_poller.urllib.request.urlopen",
               side_effect=Exception("mocked")):
        poller.start()
        assert poller._running is True
        assert poller._thread is not None
        poller.stop()
    assert poller._running is False
