"""Public response-shape contracts and cache faults; no network or account access."""
from __future__ import annotations

import copy
import math
import threading
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from data.oi_cache import BinanceOICache, EVENT_TTL_MS, HISTORY_POINTS, LS_PATH, OI_PATH


class OICacheTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_800_000_005.0
        self.hour = int(self.now // 3600) * 3_600_000
        self.calls = []
        self.payloads = {
            OI_PATH: [{"symbol": "BTCUSDT", "timestamp": self.hour - i * 3_600_000,
                       "sumOpenInterest": str(1000 - i), "sumOpenInterestValue": "99999999"}
                      for i in reversed(range(HISTORY_POINTS))],
            LS_PATH: [{"symbol": "BTCUSDT", "timestamp": self.hour - i * 3_600_000,
                       "longShortRatio": str(1.5 - i * 0.01), "longAccount": "0.6", "shortAccount": "0.4"}
                      for i in reversed(range(HISTORY_POINTS))],
        }
        self.cache = BinanceOICache("BTCUSDT", fetch=self.fetch, clock=lambda: self.now)
        self.addCleanup(self.cache.stop)

    def fetch(self, path, params):
        self.calls.append((path, params))
        return copy.deepcopy(self.payloads[path])

    def assert_unready(self, result, status):
        self.assertEqual(result["status"], status, result)
        self.assertTrue(math.isnan(result["open_interest"]))
        self.assertTrue(math.isnan(result["ls_ratio"]))

    def test_initial_cache_is_explicitly_not_ready(self):
        self.assert_unready(self.cache.get(), "error")
        self.assertEqual(self.cache.get()["reason"], "not_initialized")
        self.assertEqual(self.calls, [])

    def test_hourly_source_units_and_timestamp_contract(self):
        self.assertTrue(self.cache.refresh())
        result = self.cache.get()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["open_interest"], 1000)
        self.assertEqual(result["ls_ratio"], 1.5)
        self.assertEqual(result["oi_event_ts_ms"], self.hour)
        self.assertEqual(result["ls_event_ts_ms"], self.hour)
        self.assertEqual(result["received_at_ms"], int(self.now * 1000))
        self.assertEqual(result["source"], "binance-usdm")
        self.assertEqual(result["period"], "1h")
        self.assertEqual([call[0] for call in self.calls], [OI_PATH, LS_PATH])
        for _, params in self.calls:
            self.assertEqual(params, {"symbol": "BTCUSDT", "period": "1h", "limit": HISTORY_POINTS,
                                      "endTime": int(self.now * 1000)})

    def test_closed_four_hour_bar_uses_causal_hourly_observation(self):
        self.assertTrue(self.cache.refresh())
        at = self.hour - 4 * 3_600_000 + 12345
        result = self.cache.get(as_of_ms=at)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["open_interest"], 996)
        self.assertEqual(result["oi_event_ts_ms"], self.hour - 4 * 3_600_000)
        self.assertLessEqual(result["ls_event_ts_ms"], at)

    def test_missing_causal_point_does_not_use_future_data(self):
        self.assertTrue(self.cache.refresh())
        self.assert_unready(self.cache.get(as_of_ms=self.hour - HISTORY_POINTS * 3_600_000), "stale")
        self.assert_unready(self.cache.get(as_of_ms=int(self.now * 1000) + 1), "error")

    def test_causal_gap_expires_even_when_latest_data_is_fresh(self):
        for key in self.payloads:
            self.payloads[key] = [self.payloads[key][0], self.payloads[key][-1]]
        self.assertTrue(self.cache.refresh())
        self.assert_unready(self.cache.get(as_of_ms=self.hour - 2 * 3_600_000), "stale")

    def test_receipt_expiry_and_clock_rollback_revoke_readiness(self):
        self.assertTrue(self.cache.refresh())
        received = self.now
        self.now += 120.001
        self.assert_unready(self.cache.get(), "stale")
        self.assertEqual(self.cache.get()["reason"], "receipt_expired")
        self.now = received - 1
        self.assert_unready(self.cache.get(), "error")

    def test_refetching_old_observation_does_not_reset_its_event_age(self):
        self.assertTrue(self.cache.refresh())
        self.now = (self.hour + EVENT_TTL_MS + 1) / 1000
        self.assertFalse(self.cache.refresh())
        self.assert_unready(self.cache.get(), "stale")
        self.assertEqual(self.cache.get()["reason"], "open_interest_latest_expired")

    def test_old_latest_data_blocks_even_if_causal_query_is_historical(self):
        self.assertTrue(self.cache.refresh())
        self.now = (self.hour + EVENT_TTL_MS + 1) / 1000
        self.cache.refresh()
        self.assert_unready(self.cache.get(as_of_ms=self.hour - 4 * 3_600_000), "stale")

    def test_network_error_and_rate_limit_revoke_previous_good_snapshot(self):
        for error in (TimeoutError("timeout"), HTTPError("https://example.invalid", 429, "limited", {}, None)):
            with self.subTest(error=type(error).__name__):
                self.assertTrue(self.cache.refresh())
                with patch.object(self.cache, "_fetch", side_effect=error):
                    self.assertFalse(self.cache.refresh())
                self.assert_unready(self.cache.get(), "error")
                self.assertTrue(self.cache.refresh())
                self.assertEqual(self.cache.get()["status"], "ready")

    def test_partial_pair_failure_never_publishes_new_oi(self):
        self.assertTrue(self.cache.refresh())
        self.now += 3600
        self.payloads[OI_PATH].append({"symbol": "BTCUSDT", "timestamp": self.hour + 3_600_000,
                                      "sumOpenInterest": "2000"})
        self.payloads[OI_PATH].pop(0)
        old_ls = self.payloads[LS_PATH]
        self.payloads[LS_PATH] = {"code": -1, "msg": "bad response"}
        self.assertFalse(self.cache.refresh())
        self.assert_unready(self.cache.get(), "error")
        self.payloads[LS_PATH] = old_ls

    def test_malformed_and_nonfinite_values_are_rejected(self):
        for path, field in ((OI_PATH, "sumOpenInterest"), (LS_PATH, "longShortRatio")):
            original = copy.deepcopy(self.payloads[path])
            for value in (None, True, 0, -1, "nan", "inf", "-inf", "bad"):
                with self.subTest(path=path, value=value):
                    self.payloads[path] = copy.deepcopy(original)
                    self.payloads[path][-1][field] = value
                    self.assertFalse(self.cache.refresh())
                    self.assert_unready(self.cache.get(), "error")
            self.payloads[path] = original
        for payload in ({"code": -1003}, [], [None], self.payloads[OI_PATH] * 2):
            self.payloads[OI_PATH] = payload
            self.assertFalse(self.cache.refresh())
            self.assert_unready(self.cache.get(), "error")

    def test_wrong_symbol_future_duplicate_and_out_of_order_timestamps_fail(self):
        original = copy.deepcopy(self.payloads[OI_PATH])
        mutations = [
            lambda rows: rows[-1].update(symbol="ETHUSDT"),
            lambda rows: rows[-1].update(timestamp=int(self.now * 1000) + 1),
            lambda rows: rows[-1].update(timestamp=rows[-2]["timestamp"]),
            lambda rows: rows.reverse(),
            lambda rows: rows[-1].update(timestamp=str(self.hour)),
            lambda rows: rows[-1].update(timestamp=True),
        ]
        for mutate in mutations:
            self.payloads[OI_PATH] = copy.deepcopy(original)
            mutate(self.payloads[OI_PATH])
            self.assertFalse(self.cache.refresh())
            self.assert_unready(self.cache.get(), "error")

    def test_regressed_latest_or_conflicting_repeat_is_rejected(self):
        self.assertTrue(self.cache.refresh())
        original = copy.deepcopy(self.payloads[OI_PATH])
        self.payloads[OI_PATH].pop()
        self.assertFalse(self.cache.refresh())
        self.assert_unready(self.cache.get(), "error")
        self.payloads[OI_PATH] = original
        self.payloads[OI_PATH][-1]["sumOpenInterest"] = "9999"
        self.assertFalse(self.cache.refresh())
        self.assert_unready(self.cache.get(), "error")

    def test_short_response_retains_prior_causal_history(self):
        self.assertTrue(self.cache.refresh())
        for key in self.payloads:
            self.payloads[key] = self.payloads[key][-1:]
        self.assertTrue(self.cache.refresh())
        self.assertEqual(self.cache.get(as_of_ms=self.hour - 4 * 3_600_000)["open_interest"], 996)

    def test_invalid_clock_or_query_cannot_be_ready(self):
        self.assertTrue(self.cache.refresh())
        for value in (True, 0, "today", -1, float("nan")):
            self.assert_unready(self.cache.get(as_of_ms=value), "error")
        for value in (0, -1, float("nan"), float("inf")):
            self.now = value
            self.assertFalse(self.cache.refresh())
            self.assert_unready(self.cache.get(), "error")

    def test_thread_start_is_idempotent_and_stop_revokes_readiness(self):
        fetched = threading.Event()
        original = self.cache._fetch

        def fetch(path, params):
            result = original(path, params)
            if path == LS_PATH:
                fetched.set()
            return result

        with patch.object(self.cache, "_fetch", side_effect=fetch):
            self.cache.start()
            worker = self.cache._thread
            self.cache.start()
            self.assertIs(self.cache._thread, worker)
            self.assertTrue(fetched.wait(2))
            self.cache.stop()
            self.assertFalse(worker.is_alive())
            self.assert_unready(self.cache.get(), "error")
            self.assertFalse(self.cache.refresh())
            self.assertEqual(len(self.calls), 2)

    def test_public_fetch_uses_only_public_get_and_bounds_response(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, limit):
                self.limit = limit
                return b'[]'

        response = Response()
        cache = BinanceOICache("BTCUSDT")
        with patch("data.oi_cache.urlopen", return_value=response) as request:
            self.assertEqual(cache._fetch_public(OI_PATH, {"symbol": "BTCUSDT", "period": "1h"}), [])
            req = request.call_args.args[0]
            self.assertEqual(req.get_method(), "GET")
            self.assertTrue(req.full_url.startswith("https://fapi.binance.com/futures/data/openInterestHist?"))
            self.assertEqual(dict(req.header_items()), {"Accept": "application/json"})
            self.assertEqual(response.limit, 1_048_577)


if __name__ == "__main__":
    unittest.main()
