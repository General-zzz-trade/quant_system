"""Public shadow data cannot silently become account traffic or unsafe bars."""
from __future__ import annotations

from copy import deepcopy
import json
from socket import socket
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from data.oi_cache import BinanceOICache
from runner.public_market_data import (
    BINANCE_HOST, BYBIT_HOST, KLINE_PATH, LS_PATH, MAX_RESPONSE_BYTES, OI_PATH,
    TICKER_PATH, BybitPublicMarketData, _NoRedirect,
)


class PublicMarketDataTests(unittest.TestCase):
    def setUp(self):
        self.no_network = patch.object(socket, "connect", side_effect=AssertionError("offline only"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)
        self.hour = 3_600_000
        self.boundary = 1_800_000_000_000
        self.now_ms = self.boundary + 120_000
        self.requests = []
        self.payload = self.klines()
        self.client = BybitPublicMarketData(["BTCUSDT", "ETHUSDT"], fetch=self.fetch,
                                            clock=lambda: self.now_ms / 1000)

    def fetch(self, request, timeout):
        self.requests.append((request, timeout))
        return self.payload if isinstance(self.payload, bytes) else json.dumps(self.payload).encode()

    def klines(self):
        rows = [[str(self.boundary - self.hour * i), "100", "110", "90", "105", "2", "210"]
                for i in range(1, 4)]
        return {"retCode": 0, "time": self.now_ms,
                "result": {"category": "linear", "symbol": "BTCUSDT", "list": rows}}

    def ticker(self):
        return {"retCode": 0, "time": self.now_ms,
                "result": {"category": "linear", "list": [{
                    "symbol": "BTCUSDT", "lastPrice": "100.25", "markPrice": "100.20",
                    "fundingRate": "-0.0001", "not_needed": "ignored",
                }]}}

    def oi_params(self):
        return {"symbol": "BTCUSDT", "period": "1h", "limit": 8, "endTime": self.now_ms}

    def observations(self, path):
        field = "sumOpenInterest" if path == OI_PATH else "longShortRatio"
        return [{"symbol": "BTCUSDT", "timestamp": self.boundary - i * self.hour, field: "2.5"}
                for i in reversed(range(8))]

    def test_closed_kline_request_and_epoch_ms_match_warmup_contract(self):
        rows = self.client.get_klines("BTCUSDT", interval="60", limit=3)
        self.assertEqual([r["time"] for r in rows], [self.boundary - self.hour * i for i in range(1, 4)])
        self.assertTrue(all(r["time"] == r["start"] and r["confirm"] is True for r in rows))
        request, timeout = self.requests[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.data)
        url = urlsplit(request.full_url)
        self.assertEqual(url.scheme + "://" + url.netloc, BYBIT_HOST)
        self.assertEqual(url.path, KLINE_PATH)
        self.assertEqual(parse_qs(url.query), {"category": ["linear"], "symbol": ["BTCUSDT"],
                                             "interval": ["60"], "limit": ["3"],
                                             "end": [str(self.boundary - 1)]})
        self.assertEqual(timeout, 5)
        self.assertEqual(set(dict(request.header_items())), {"Accept", "User-agent"})

    def test_closed_bar_cursor_preserves_chronological_order_without_duplicates(self):
        rows = self.client.get_closed_bars("BTCUSDT", limit=3, after_ms=self.boundary - 3 * self.hour)
        self.assertEqual([r["time"] for r in rows], [self.boundary - 2 * self.hour, self.boundary - self.hour])
        self.assertEqual(self.client.get_closed_bars("BTCUSDT", limit=3,
                                                     after_ms=self.boundary - self.hour), [])
        for cursor in (-1, True, "0", float("nan")):
            with self.assertRaises(ValueError):
                self.client.get_closed_bars("BTCUSDT", after_ms=cursor)

    def test_closed_bar_cursor_rejects_missing_window_and_accepts_no_new_bars(self):
        # Three returned bars are contiguous, but the first expected bar is
        # outside this bounded response: continuing would silently skip it.
        with self.assertRaisesRegex(ValueError, "gap after.*cursor"):
            self.client.get_closed_bars("BTCUSDT", limit=3,
                                        after_ms=self.boundary - 5 * self.hour)
        rows = self.client.get_closed_bars("BTCUSDT", limit=3,
                                           after_ms=self.boundary - 4 * self.hour)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(self.client.get_closed_bars("BTCUSDT", limit=3, after_ms=0)), 3)
        for cursor in (self.boundary - self.hour, self.boundary, 9_999_999_999_999_999):
            with self.subTest(cursor=cursor):
                self.assertEqual(self.client.get_closed_bars("BTCUSDT", limit=3, after_ms=cursor), [])

    def test_public_client_has_no_account_order_or_notification_api(self):
        for name in ("connect", "get_balances", "get_positions", "get_account_equity", "place_order",
                     "send_market_order", "cancel_order", "set_leverage", "send_alert"):
            self.assertFalse(hasattr(self.client, name), name)
        with patch.dict("os.environ", {"BYBIT_API_KEY": "offline-fixture", "BYBIT_API_SECRET": "offline-fixture"}):
            self.client.get_klines("BTCUSDT", limit=3)
        request = self.requests[-1][0]
        self.assertNotIn("offline-fixture", request.full_url + str(request.header_items()))

    def test_unsupported_symbols_intervals_counts_and_endpoints_fail_before_transport(self):
        for symbol in ("UNKNOWN", "btcusdt", "BTCUSDT&api_key=x", None):
            with self.assertRaises(ValueError):
                self.client.get_klines(symbol)
        for interval in ("D", "60&category=spot", 60, []):
            with self.assertRaises(ValueError):
                self.client.get_klines("BTCUSDT", interval=interval)
        for limit in (0, 1001, True, 1.5):
            with self.assertRaises(ValueError):
                self.client.get_klines("BTCUSDT", limit=limit)
        for host, path, params in ((BYBIT_HOST, "/v5/account/wallet-balance", {}),
                                   ("https://attacker.example", KLINE_PATH, {}),
                                   (BYBIT_HOST, TICKER_PATH, {"category": "linear", "symbol": "BTCUSDT",
                                                             "api_key": "forbidden"})):
            with self.assertRaises(ValueError):
                self.client._request(host, path, params)
        self.assertEqual(self.requests, [])

    def test_invalid_client_configuration_is_rejected(self):
        for symbols in ([], ["BTCUSDT", "BTCUSDT"], "BTCUSDT", ["bad/symbol"], [[]]):
            with self.assertRaises(ValueError):
                BybitPublicMarketData(symbols)
        for timeout in (0, -1, 31, True, float("nan")):
            with self.assertRaises(ValueError):
                BybitPublicMarketData(["BTCUSDT"], request_timeout=timeout)

    def test_bad_bybit_envelopes_symbol_and_server_clock_are_rejected(self):
        changes = [{"retCode": 10001}, {"retCode": False}, {"time": self.now_ms // 1000},
                   {"time": self.now_ms + 30_001}, {"time": self.now_ms - 30_001}, {"result": None}]
        for change in changes:
            self.payload = {**self.klines(), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.client.get_klines("BTCUSDT", limit=3)
        for key, value in (("symbol", "ETHUSDT"), ("category", "spot"), ("list", []), ("list", {})):
            self.payload = self.klines()
            self.payload["result"][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.client.get_klines("BTCUSDT", limit=3)

    def test_partial_future_unaligned_and_seconds_bars_are_rejected(self):
        for timestamp in (self.boundary, self.boundary + self.hour,
                          self.boundary - self.hour + 1, (self.boundary - self.hour) // 1000):
            self.payload = self.klines()
            self.payload["result"]["list"][0][0] = str(timestamp)
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.client.get_klines("BTCUSDT", limit=3)

    def test_nonfinite_negative_or_contradictory_market_values_are_rejected(self):
        for index, value in ((1, "NaN"), (1, "1e9999"), (2, "99"), (3, "106"),
                             (4, "0"), (5, "-1"), (6, "Infinity"), (1, True)):
            self.payload = self.klines()
            self.payload["result"]["list"][0][index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                self.client.get_klines("BTCUSDT", limit=3)
        self.payload = self.klines()
        self.payload["result"]["list"][0][5:7] = ["0", "0"]
        self.assertEqual(self.client.get_klines("BTCUSDT", limit=3)[0]["volume"], "0")

    def test_identical_duplicates_are_deduplicated_and_conflicts_or_gaps_fail(self):
        original = self.klines()
        self.payload = deepcopy(original)
        self.payload["result"]["list"].append(deepcopy(self.payload["result"]["list"][0]))
        self.assertEqual(len(self.client.get_klines("BTCUSDT", limit=4)), 3)
        self.payload["result"]["list"][-1][4] = "106"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.client.get_klines("BTCUSDT", limit=4)
        self.payload = original
        self.payload["result"]["list"].pop(1)
        with self.assertRaisesRegex(ValueError, "gap"):
            self.client.get_klines("BTCUSDT", limit=3)

    def test_ticker_only_returns_whitelisted_finite_fields(self):
        self.payload = self.ticker()
        self.assertEqual(self.client.get_ticker("BTCUSDT"), {
            "symbol": "BTCUSDT", "lastPrice": "100.25", "markPrice": "100.20", "fundingRate": "-0.0001",
        })
        self.assertEqual(urlsplit(self.requests[-1][0].full_url).path, TICKER_PATH)
        for field, value in (("symbol", "ETHUSDT"), ("lastPrice", "0"), ("fundingRate", "NaN")):
            self.payload = self.ticker()
            self.payload["result"]["list"][0][field] = value
            with self.assertRaises(ValueError):
                self.client.get_ticker("BTCUSDT")

    def test_oi_transport_preserves_existing_causal_cache_contract(self):
        def fetch(request, timeout):
            self.requests.append((request, timeout))
            url = urlsplit(request.full_url)
            self.assertEqual(url.scheme + "://" + url.netloc, BINANCE_HOST)
            self.assertIn(url.path, (OI_PATH, LS_PATH))
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(parse_qs(url.query), {
                "symbol": ["BTCUSDT"], "period": ["1h"], "limit": ["8"], "endTime": [str(self.now_ms)],
            })
            return json.dumps(self.observations(url.path)).encode()
        client = BybitPublicMarketData(["BTCUSDT"], fetch=fetch, clock=lambda: self.now_ms / 1000)
        cache = BinanceOICache("BTCUSDT", fetch=client.fetch_oi, clock=lambda: self.now_ms / 1000)
        self.assertTrue(cache.refresh())
        observed = cache.get(as_of_ms=self.boundary - self.hour)
        self.assertEqual(observed["status"], "ready")
        self.assertEqual(observed["open_interest"], 2.5)
        self.assertEqual(observed["ls_ratio"], 2.5)
        self.assertEqual(observed["oi_event_ts_ms"], self.boundary - self.hour)

    def test_oi_endpoints_and_query_shape_cannot_be_expanded(self):
        for path, change in (("/fapi/v2/account", {}), (OI_PATH, {"symbol": "UNKNOWN"}),
                             (OI_PATH, {"period": "5m"}), (OI_PATH, {"limit": 9}),
                             (OI_PATH, {"endTime": self.now_ms + 1}), (OI_PATH, {"api_key": "forbidden"})):
            with self.subTest(path=path, change=change), self.assertRaises(ValueError):
                self.client.fetch_oi(path, {**self.oi_params(), **change})
        self.assertEqual(self.requests, [])

    def test_oi_wrong_symbol_future_duplicate_and_bad_values_fail(self):
        for field, value in (("symbol", "ETHUSDT"), ("timestamp", self.now_ms + 1),
                             ("timestamp", self.boundary), ("sumOpenInterest", "NaN")):
            self.payload = self.observations(OI_PATH)
            self.payload[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.client.fetch_oi(OI_PATH, self.oi_params())

    def test_invalid_oversized_and_duplicate_json_are_rejected(self):
        for raw in (b"x" * (MAX_RESPONSE_BYTES + 1), b'{"retCode":0,"retCode":0}',
                    b'{"retCode":NaN}', b'{"retCode":1e9999}', b"\xff", b"not-json"):
            self.payload = raw
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                self.client.get_klines("BTCUSDT", limit=3)

    def test_transport_failure_and_backwards_clock_never_create_observations(self):
        self.client._fetch = Mock(side_effect=TimeoutError("synthetic public timeout"))
        with self.assertRaises(TimeoutError):
            self.client.get_klines("BTCUSDT", limit=3)
        self.client._fetch = self.fetch
        times = iter([self.now_ms / 1000, self.now_ms / 1000, self.now_ms / 1000 - 1])
        self.client._clock = lambda: next(times)
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.client.get_klines("BTCUSDT", limit=3)

    def test_default_transport_refuses_redirects_and_changed_destination(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example"))
        opener = Mock()
        opener.open.side_effect = HTTPError(BYBIT_HOST + KLINE_PATH, 302, "redirect", {}, None)
        with patch("runner.public_market_data.build_opener", return_value=opener) as build:
            client = BybitPublicMarketData(["BTCUSDT"], clock=lambda: self.now_ms / 1000)
            with self.assertRaises(HTTPError):
                client.get_klines("BTCUSDT", limit=3)
            self.assertIsInstance(build.call_args.args[0], _NoRedirect)
            self.assertEqual(opener.open.call_count, 1)
        response = Mock(status=200)
        response.geturl.return_value = BYBIT_HOST + "/v5/account/wallet-balance"
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener.open.side_effect = None
        opener.open.return_value = response
        with patch("runner.public_market_data.build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "destination"):
                client.get_klines("BTCUSDT", limit=3)
        response.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
