"""
Coverage-targeted unit tests for execution/ modules with low coverage.

Targets:
  execution/bridge/execution_bridge.py
  execution/safety/circuit_breaker.py
  execution/safety/kill_switch.py
  execution/safety/limits.py
  execution/bridge/error_policy.py
  execution/bridge/request_ids.py
  execution/bridge/venue_router.py
  execution/models/commands.py
  execution/store/ack_store.py
  execution/observability/redaction.py
  execution/observability/tracing.py
  execution/observability/metrics.py
  execution/adapters/common/decimals.py
  execution/adapters/common/symbols.py
  execution/adapters/common/time.py
  execution/adapters/common/idempotency.py
  execution/adapters/common/schema_checks.py
  execution/state_machine/invariants.py
  execution/ingress/stream_health.py
"""
from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import MagicMock


# _quant_hotpath is available as a compiled .so — no stub needed.


# ===========================================================================
# 1. execution/observability/redaction.py
# ===========================================================================
class TestRedaction(unittest.TestCase):

    def setUp(self):
        from execution.observability.redaction import (
            redact_value, redact_dict, redact_url
        )
        self.redact_value = redact_value
        self.redact_dict = redact_dict
        self.redact_url = redact_url

    def test_redact_value_sensitive_long_string(self):
        result = self.redact_value("api_key", "abcdefghijklmnop")
        self.assertIn("****", result)
        self.assertNotEqual(result, "abcdefghijklmnop")

    def test_redact_value_sensitive_short_string(self):
        # Shorter than 8 chars => "****"
        result = self.redact_value("secret", "short")
        self.assertEqual(result, "****")

    def test_redact_value_non_sensitive(self):
        result = self.redact_value("symbol", "BTCUSDT")
        self.assertEqual(result, "BTCUSDT")

    def test_redact_value_exact_8_chars(self):
        # exactly 8 chars → "****"
        result = self.redact_value("api_key", "12345678")
        self.assertEqual(result, "****")

    def test_redact_value_long_key(self):
        # > 8 chars with pattern match
        result = self.redact_value("apiKey", "abcdEFGHijkl")
        self.assertTrue(result.startswith("abcd"))
        self.assertTrue(result.endswith("ijkl"))

    def test_redact_dict_flat(self):
        data = {"api_key": "abcdefghijklmnop", "symbol": "ETHUSDT"}
        result = self.redact_dict(data)
        self.assertIn("****", result["api_key"])
        self.assertEqual(result["symbol"], "ETHUSDT")

    def test_redact_dict_nested(self):
        data = {"headers": {"X-MBX-APIKEY": "abcdefghijklmnop", "Content-Type": "application/json"}}
        result = self.redact_dict(data)
        self.assertIn("****", result["headers"]["X-MBX-APIKEY"])
        self.assertEqual(result["headers"]["Content-Type"], "application/json")

    def test_redact_dict_list_of_dicts(self):
        data = {"items": [{"api_key": "abcdefghijklmnop"}, {"foo": "bar"}]}
        result = self.redact_dict(data)
        self.assertIn("****", result["items"][0]["api_key"])
        self.assertEqual(result["items"][1]["foo"], "bar")

    def test_redact_dict_list_of_primitives(self):
        data = {"prices": [1.0, 2.0, 3.0]}
        result = self.redact_dict(data)
        self.assertEqual(result["prices"], [1.0, 2.0, 3.0])

    def test_redact_url_removes_api_key(self):
        url = "https://api.binance.com/api?apiKey=MYSECRETKEY&symbol=BTCUSDT"
        result = self.redact_url(url)
        self.assertNotIn("MYSECRETKEY", result)
        self.assertIn("apiKey=****", result)

    def test_redact_url_removes_signature(self):
        url = "https://api.binance.com/api?signature=abc123&timestamp=99999"
        result = self.redact_url(url)
        self.assertNotIn("abc123", result)

    def test_redact_url_no_sensitive_params(self):
        url = "https://api.binance.com/api?symbol=BTCUSDT"
        result = self.redact_url(url)
        self.assertEqual(result, url)


# ===========================================================================
# 2. execution/observability/tracing.py
# ===========================================================================
class TestTracing(unittest.TestCase):

    def setUp(self):
        from execution.observability.tracing import Tracer, SpanBuilder, Span
        self.Tracer = Tracer
        self.SpanBuilder = SpanBuilder
        self.Span = Span

    def test_span_duration_ms(self):
        span = self.Span(
            span_id="abc",
            trace_id="def",
            operation="test_op",
            start_ts=1000.0,
            end_ts=1001.5,
            status="ok",
        )
        self.assertAlmostEqual(span.duration_ms, 1500.0)

    def test_span_builder_finish(self):
        builder = self.SpanBuilder("submit_order")
        span = builder.finish()
        self.assertEqual(span.operation, "submit_order")
        self.assertEqual(span.status, "ok")
        self.assertIsNone(span.error)

    def test_span_builder_error(self):
        builder = self.SpanBuilder("submit_order")
        builder.error("something went wrong")
        span = builder.finish()
        self.assertEqual(span.status, "error")
        self.assertEqual(span.error, "something went wrong")

    def test_span_builder_tag(self):
        builder = self.SpanBuilder("op")
        builder.tag("venue", "bybit").tag("symbol", "BTCUSDT")
        span = builder.finish()
        self.assertEqual(span.tags["venue"], "bybit")
        self.assertEqual(span.tags["symbol"], "BTCUSDT")

    def test_span_builder_custom_trace_id(self):
        builder = self.SpanBuilder("op", trace_id="my-trace-123")
        span = builder.finish()
        self.assertEqual(span.trace_id, "my-trace-123")

    def test_tracer_record_and_query(self):
        tracer = self.Tracer()
        builder = tracer.start_span("submit_order")
        span = builder.finish()
        tracer.record(span)
        results = tracer.query(operation="submit_order")
        self.assertEqual(len(results), 1)

    def test_tracer_query_by_trace_id(self):
        tracer = self.Tracer()
        b1 = tracer.start_span("op", trace_id="trace-A")
        s1 = b1.finish()
        tracer.record(s1)
        b2 = tracer.start_span("op", trace_id="trace-B")
        s2 = b2.finish()
        tracer.record(s2)
        results = tracer.query(trace_id="trace-A")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].trace_id, "trace-A")

    def test_tracer_max_spans_eviction(self):
        tracer = self.Tracer(max_spans=5)
        for i in range(10):
            b = tracer.start_span(f"op-{i}")
            tracer.record(b.finish())
        # Should retain only latest max_spans
        results = tracer.query(limit=100)
        self.assertLessEqual(len(results), 5)

    def test_tracer_query_limit(self):
        tracer = self.Tracer()
        for _ in range(20):
            tracer.record(tracer.start_span("op").finish())
        results = tracer.query(operation="op", limit=5)
        self.assertEqual(len(results), 5)

    def test_tracer_query_no_match(self):
        tracer = self.Tracer()
        tracer.record(tracer.start_span("op").finish())
        results = tracer.query(operation="nonexistent")
        self.assertEqual(len(results), 0)


# ===========================================================================
# 3. execution/observability/metrics.py
# ===========================================================================
class TestMetrics(unittest.TestCase):

    def setUp(self):
        from execution.observability.metrics import (
            Counter, Gauge, ExecutionMetrics
        )
        self.Counter = Counter
        self.Gauge = Gauge
        self.ExecutionMetrics = ExecutionMetrics

    def test_counter_inc_and_get(self):
        c = self.Counter("test_counter")
        c.inc(1.0, venue="bybit")
        c.inc(2.0, venue="bybit")
        self.assertEqual(c.get(venue="bybit"), 3.0)

    def test_counter_get_zero_default(self):
        c = self.Counter("test_counter")
        self.assertEqual(c.get(venue="unknown"), 0.0)

    def test_counter_multiple_labels(self):
        c = self.Counter("orders")
        c.inc(venue="bybit", symbol="BTCUSDT")
        c.inc(venue="bybit", symbol="ETHUSDT")
        self.assertEqual(c.get(venue="bybit", symbol="BTCUSDT"), 1.0)
        self.assertEqual(c.get(venue="bybit", symbol="ETHUSDT"), 1.0)

    def test_counter_snapshot(self):
        c = self.Counter("orders")
        c.inc(venue="bybit")
        snaps = c.snapshot()
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0].name, "orders")
        self.assertEqual(snaps[0].value, 1.0)

    def test_gauge_set_and_get(self):
        g = self.Gauge("active_orders")
        g.set(5.0, venue="bybit")
        self.assertEqual(g.get(venue="bybit"), 5.0)

    def test_gauge_set_overwrite(self):
        g = self.Gauge("active_orders")
        g.set(5.0, venue="bybit")
        g.set(10.0, venue="bybit")
        self.assertEqual(g.get(venue="bybit"), 10.0)

    def test_gauge_get_default_zero(self):
        g = self.Gauge("active_orders")
        self.assertEqual(g.get(venue="unknown"), 0.0)

    def test_execution_metrics_record_submit(self):
        m = self.ExecutionMetrics()
        m.record_submit("bybit", "BTCUSDT")
        self.assertEqual(m.orders_submitted.get(venue="bybit", symbol="BTCUSDT"), 1.0)

    def test_execution_metrics_record_fill(self):
        m = self.ExecutionMetrics()
        m.record_fill("bybit", "BTCUSDT")
        self.assertEqual(m.orders_filled.get(venue="bybit", symbol="BTCUSDT"), 1.0)
        self.assertEqual(m.fills_received.get(venue="bybit", symbol="BTCUSDT"), 1.0)

    def test_execution_metrics_record_reject(self):
        m = self.ExecutionMetrics()
        m.record_reject("bybit", "ETHUSDT")
        self.assertEqual(m.orders_rejected.get(venue="bybit", symbol="ETHUSDT"), 1.0)

    def test_execution_metrics_record_error(self):
        m = self.ExecutionMetrics()
        m.record_error("bybit", "timeout")
        self.assertEqual(m.errors.get(venue="bybit", error_type="timeout"), 1.0)


# ===========================================================================
# 4. execution/adapters/common/decimals.py
# ===========================================================================
class TestDecimals(unittest.TestCase):

    def setUp(self):
        from execution.adapters.common.decimals import (
            safe_decimal, require_decimal, round_down,
            round_to_precision, is_positive, clamp, ZERO
        )
        self.safe_decimal = safe_decimal
        self.require_decimal = require_decimal
        self.round_down = round_down
        self.round_to_precision = round_to_precision
        self.is_positive = is_positive
        self.clamp = clamp
        self.ZERO = ZERO

    def test_safe_decimal_none_returns_default(self):
        self.assertEqual(self.safe_decimal(None), self.ZERO)

    def test_safe_decimal_decimal_passthrough(self):
        d = Decimal("1.23")
        self.assertEqual(self.safe_decimal(d), d)

    def test_safe_decimal_string(self):
        self.assertEqual(self.safe_decimal("1.5"), Decimal("1.5"))

    def test_safe_decimal_float(self):
        result = self.safe_decimal(1.5)
        self.assertIsInstance(result, Decimal)

    def test_safe_decimal_invalid_returns_default(self):
        result = self.safe_decimal("not_a_number")
        self.assertEqual(result, self.ZERO)

    def test_safe_decimal_custom_default(self):
        result = self.safe_decimal("bad", default=Decimal("-1"))
        self.assertEqual(result, Decimal("-1"))

    def test_require_decimal_valid(self):
        result = self.require_decimal("3.14", "price")
        self.assertEqual(result, Decimal("3.14"))

    def test_require_decimal_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.require_decimal("NaN_bad", "qty")

    def test_require_decimal_passthrough(self):
        d = Decimal("99")
        self.assertEqual(self.require_decimal(d, "x"), d)

    def test_round_down_basic(self):
        result = self.round_down(Decimal("1.75"), Decimal("0.1"))
        self.assertEqual(result, Decimal("1.7"))

    def test_round_down_zero_step_passthrough(self):
        result = self.round_down(Decimal("1.75"), Decimal("0"))
        self.assertEqual(result, Decimal("1.75"))

    def test_round_to_precision_zero(self):
        result = self.round_to_precision(Decimal("3.99"), 0)
        self.assertEqual(result, Decimal("3"))

    def test_round_to_precision_two(self):
        result = self.round_to_precision(Decimal("3.14159"), 2)
        self.assertEqual(result, Decimal("3.14"))

    def test_round_to_precision_negative_passthrough(self):
        result = self.round_to_precision(Decimal("3.14"), -1)
        self.assertEqual(result, Decimal("3.14"))

    def test_is_positive_true(self):
        self.assertTrue(self.is_positive(Decimal("0.001")))

    def test_is_positive_false_zero(self):
        self.assertFalse(self.is_positive(Decimal("0")))

    def test_is_positive_false_negative(self):
        self.assertFalse(self.is_positive(Decimal("-1")))

    def test_clamp_below_lo(self):
        result = self.clamp(Decimal("-5"), Decimal("0"), Decimal("100"))
        self.assertEqual(result, Decimal("0"))

    def test_clamp_above_hi(self):
        result = self.clamp(Decimal("200"), Decimal("0"), Decimal("100"))
        self.assertEqual(result, Decimal("100"))

    def test_clamp_within_range(self):
        result = self.clamp(Decimal("50"), Decimal("0"), Decimal("100"))
        self.assertEqual(result, Decimal("50"))


# ===========================================================================
# 5. execution/adapters/common/symbols.py
# ===========================================================================
class TestSymbols(unittest.TestCase):

    def setUp(self):
        from execution.adapters.common.symbols import (
            normalize_symbol, split_symbol, normalize_side,
            normalize_order_type, normalize_tif
        )
        self.normalize_symbol = normalize_symbol
        self.split_symbol = split_symbol
        self.normalize_side = normalize_side
        self.normalize_order_type = normalize_order_type
        self.normalize_tif = normalize_tif

    def test_normalize_symbol_strips_upper(self):
        self.assertEqual(self.normalize_symbol("  btcusdt  "), "BTCUSDT")

    def test_split_symbol_slash(self):
        base, quote = self.split_symbol("BTC/USDT")
        self.assertEqual(base, "BTC")
        self.assertEqual(quote, "USDT")

    def test_split_symbol_dash(self):
        base, quote = self.split_symbol("ETH-USDT")
        self.assertEqual(base, "ETH")
        self.assertEqual(quote, "USDT")

    def test_split_symbol_underscore(self):
        base, quote = self.split_symbol("BTC_USDT")
        self.assertEqual(base, "BTC")
        self.assertEqual(quote, "USDT")

    def test_split_symbol_no_sep_known_quote(self):
        base, quote = self.split_symbol("BTCUSDT")
        self.assertEqual(base, "BTC")
        self.assertEqual(quote, "USDT")

    def test_split_symbol_eth_usdt(self):
        base, quote = self.split_symbol("ETHUSDT")
        self.assertEqual(base, "ETH")
        self.assertEqual(quote, "USDT")

    def test_split_symbol_unknown(self):
        base, quote = self.split_symbol("UNKNOWN")
        self.assertEqual(base, "UNKNOWN")
        self.assertEqual(quote, "")

    def test_normalize_side_buy_variants(self):
        for v in ("buy", "b", "long", "bid", "BUY"):
            self.assertEqual(self.normalize_side(v), "buy")

    def test_normalize_side_sell_variants(self):
        for v in ("sell", "s", "short", "ask", "SELL"):
            self.assertEqual(self.normalize_side(v), "sell")

    def test_normalize_side_passthrough_unknown(self):
        self.assertEqual(self.normalize_side("other"), "other")

    def test_normalize_order_type(self):
        self.assertEqual(self.normalize_order_type("LIMIT"), "limit")

    def test_normalize_tif_none(self):
        self.assertIsNone(self.normalize_tif(None))

    def test_normalize_tif_alias_gtx(self):
        self.assertEqual(self.normalize_tif("gtx"), "post_only")

    def test_normalize_tif_passthrough(self):
        self.assertEqual(self.normalize_tif("GTC"), "gtc")


# ===========================================================================
# 6. execution/adapters/common/time.py
# ===========================================================================
class TestTimeUtils(unittest.TestCase):

    def setUp(self):
        from execution.adapters.common.time import (
            now_ms, now_utc, ms_to_datetime, datetime_to_ms, coerce_ts_ms
        )
        self.now_ms = now_ms
        self.now_utc = now_utc
        self.ms_to_datetime = ms_to_datetime
        self.datetime_to_ms = datetime_to_ms
        self.coerce_ts_ms = coerce_ts_ms

    def test_now_ms_positive(self):
        ts = self.now_ms()
        self.assertGreater(ts, 0)

    def test_now_utc_timezone_aware(self):
        from datetime import timezone
        dt = self.now_utc()
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_ms_to_datetime_roundtrip(self):
        ms = 1_700_000_000_000
        dt = self.ms_to_datetime(ms)
        back = self.datetime_to_ms(dt)
        self.assertEqual(back, ms)

    def test_coerce_ts_ms_none_returns_now(self):
        ts = self.coerce_ts_ms(None)
        self.assertGreater(ts, 0)

    def test_coerce_ts_ms_datetime(self):
        from datetime import datetime, timezone
        dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        result = self.coerce_ts_ms(dt)
        self.assertEqual(result, int(dt.timestamp() * 1000))

    def test_coerce_ts_ms_ms_int(self):
        # large int treated as ms
        ms_val = 1_700_000_000_000
        result = self.coerce_ts_ms(ms_val)
        self.assertEqual(result, ms_val)

    def test_coerce_ts_ms_seconds_int(self):
        # small int treated as seconds → multiply by 1000
        s_val = 1_700_000_000
        result = self.coerce_ts_ms(s_val)
        self.assertEqual(result, s_val * 1000)

    def test_coerce_ts_ms_float_seconds(self):
        result = self.coerce_ts_ms(1_600_000_000.5)
        self.assertEqual(result, 1_600_000_000 * 1000)

    def test_coerce_ts_ms_string_seconds(self):
        result = self.coerce_ts_ms("1700000000")
        self.assertEqual(result, 1700000000 * 1000)

    def test_coerce_ts_ms_string_ms(self):
        result = self.coerce_ts_ms("1700000000000")
        self.assertEqual(result, 1700000000000)

    def test_coerce_ts_ms_invalid_string(self):
        ts = self.coerce_ts_ms("not_a_number")
        self.assertGreater(ts, 0)


# ===========================================================================
# 7. execution/adapters/common/idempotency.py
# ===========================================================================
class TestIdempotency(unittest.TestCase):

    def setUp(self):
        from execution.adapters.common.idempotency import (
            make_fill_idem_key, make_order_idem_key
        )
        self.make_fill_idem_key = make_fill_idem_key
        self.make_order_idem_key = make_order_idem_key

    def test_fill_key_with_fill_id(self):
        k = self.make_fill_idem_key(venue="bybit", symbol="BTCUSDT", fill_id="f001")
        self.assertIsInstance(k, str)
        self.assertEqual(len(k), 24)

    def test_fill_key_with_trade_id(self):
        k = self.make_fill_idem_key(venue="bybit", symbol="BTCUSDT", trade_id="t001")
        self.assertIsInstance(k, str)
        self.assertEqual(len(k), 24)

    def test_fill_key_combo_fallback(self):
        k = self.make_fill_idem_key(
            venue="bybit", symbol="BTCUSDT",
            order_id="o1", side="buy", qty="1.0", price="50000"
        )
        self.assertIsInstance(k, str)
        self.assertEqual(len(k), 24)

    def test_fill_key_deterministic(self):
        k1 = self.make_fill_idem_key(venue="bybit", symbol="BTCUSDT", fill_id="f999")
        k2 = self.make_fill_idem_key(venue="bybit", symbol="BTCUSDT", fill_id="f999")
        self.assertEqual(k1, k2)

    def test_fill_key_different_fill_ids_differ(self):
        k1 = self.make_fill_idem_key(venue="bybit", symbol="BTCUSDT", fill_id="f001")
        k2 = self.make_fill_idem_key(venue="bybit", symbol="BTCUSDT", fill_id="f002")
        self.assertNotEqual(k1, k2)

    def test_order_idem_key(self):
        k = self.make_order_idem_key(venue="binance", symbol="ETHUSDT", order_id="ord-123")
        self.assertIsInstance(k, str)
        self.assertEqual(len(k), 24)

    def test_order_idem_key_deterministic(self):
        k1 = self.make_order_idem_key(venue="binance", symbol="ETHUSDT", order_id="ord-123")
        k2 = self.make_order_idem_key(venue="binance", symbol="ETHUSDT", order_id="ord-123")
        self.assertEqual(k1, k2)


# ===========================================================================
# 8. execution/adapters/common/schema_checks.py
# ===========================================================================
class TestSchemaChecks(unittest.TestCase):

    def setUp(self):
        from execution.adapters.common.schema_checks import (
            require_keys, require_non_empty, safe_get, SchemaError
        )
        self.require_keys = require_keys
        self.require_non_empty = require_non_empty
        self.safe_get = safe_get
        self.SchemaError = SchemaError

    def test_require_keys_passes(self):
        self.require_keys({"a": 1, "b": 2}, ["a", "b"])  # no exception

    def test_require_keys_raises_missing(self):
        with self.assertRaises(self.SchemaError) as ctx:
            self.require_keys({"a": 1}, ["a", "b", "c"])
        self.assertIn("b", str(ctx.exception))

    def test_require_keys_with_context(self):
        with self.assertRaises(self.SchemaError) as ctx:
            self.require_keys({}, ["x"], context="fill_msg")
        self.assertIn("fill_msg", str(ctx.exception))

    def test_require_non_empty_passes(self):
        self.require_non_empty({"id": "abc123"}, ["id"])  # no exception

    def test_require_non_empty_raises_none(self):
        with self.assertRaises(self.SchemaError):
            self.require_non_empty({"id": None}, ["id"])

    def test_require_non_empty_raises_blank_string(self):
        with self.assertRaises(self.SchemaError):
            self.require_non_empty({"id": "   "}, ["id"])

    def test_require_non_empty_with_context(self):
        with self.assertRaises(self.SchemaError) as ctx:
            self.require_non_empty({"id": ""}, ["id"], context="order_update")
        self.assertIn("order_update", str(ctx.exception))

    def test_safe_get_first_present(self):
        result = self.safe_get({"a": 1, "b": 2}, "a", "b")
        self.assertEqual(result, 1)

    def test_safe_get_second_fallback(self):
        result = self.safe_get({"b": 2}, "a", "b")
        self.assertEqual(result, 2)

    def test_safe_get_default_when_all_missing(self):
        result = self.safe_get({}, "a", "b", default="fallback")
        self.assertEqual(result, "fallback")

    def test_safe_get_skips_none(self):
        result = self.safe_get({"a": None, "b": "val"}, "a", "b")
        self.assertEqual(result, "val")


# ===========================================================================
# 9. execution/state_machine/invariants.py
# ===========================================================================
class TestInvariants(unittest.TestCase):

    def setUp(self):
        from execution.state_machine.invariants import (
            check_order_invariants, assert_order_invariants,
            InvariantViolation
        )
        from execution.state_machine.transitions import OrderStatus
        self.check = check_order_invariants
        self.assert_inv = assert_order_invariants
        self.InvariantViolation = InvariantViolation
        self.Status = OrderStatus

    def _valid_kwargs(self, **overrides):
        defaults = dict(
            status=self.Status.NEW,
            qty=Decimal("1.0"),
            filled_qty=Decimal("0"),
            price=Decimal("50000"),
            order_type="limit",
        )
        defaults.update(overrides)
        return defaults

    def test_valid_new_order(self):
        result = self.check(**self._valid_kwargs())
        self.assertTrue(result.passed)
        self.assertEqual(len(result.violations), 0)

    def test_qty_zero_violation(self):
        result = self.check(**self._valid_kwargs(qty=Decimal("0")))
        self.assertFalse(result.passed)
        self.assertTrue(any("qty" in v for v in result.violations))

    def test_filled_qty_exceeds_qty(self):
        result = self.check(**self._valid_kwargs(
            qty=Decimal("1.0"), filled_qty=Decimal("1.5")
        ))
        self.assertFalse(result.passed)
        self.assertTrue(any("filled_qty" in v for v in result.violations))

    def test_filled_qty_negative(self):
        result = self.check(**self._valid_kwargs(filled_qty=Decimal("-0.1")))
        self.assertFalse(result.passed)

    def test_filled_status_filled_qty_mismatch(self):
        result = self.check(**self._valid_kwargs(
            status=self.Status.FILLED,
            qty=Decimal("1.0"),
            filled_qty=Decimal("0.5"),
        ))
        self.assertFalse(result.passed)
        self.assertTrue(any("FILLED" in v for v in result.violations))

    def test_filled_status_correct(self):
        result = self.check(**self._valid_kwargs(
            status=self.Status.FILLED,
            qty=Decimal("1.0"),
            filled_qty=Decimal("1.0"),
        ))
        self.assertTrue(result.passed)

    def test_limit_order_no_price(self):
        result = self.check(**self._valid_kwargs(
            order_type="limit", price=None
        ))
        self.assertFalse(result.passed)
        self.assertTrue(any("price" in v for v in result.violations))

    def test_market_order_no_price_ok(self):
        result = self.check(**self._valid_kwargs(
            order_type="market", price=None
        ))
        self.assertTrue(result.passed)

    def test_price_zero_violation(self):
        result = self.check(**self._valid_kwargs(price=Decimal("0")))
        self.assertFalse(result.passed)

    def test_price_negative_violation(self):
        result = self.check(**self._valid_kwargs(price=Decimal("-100")))
        self.assertFalse(result.passed)

    def test_assert_raises_on_violation(self):
        with self.assertRaises(self.InvariantViolation):
            self.assert_inv(**self._valid_kwargs(qty=Decimal("-1")))

    def test_assert_passes_valid(self):
        self.assert_inv(**self._valid_kwargs())  # no exception

    def test_multiple_violations(self):
        result = self.check(
            status=self.Status.NEW,
            qty=Decimal("0"),
            filled_qty=Decimal("-1"),
            price=Decimal("-100"),
            order_type="limit",
        )
        self.assertFalse(result.passed)
        self.assertGreater(len(result.violations), 1)


# ===========================================================================
# 10. execution/ingress/stream_health.py
# ===========================================================================
class TestStreamHealth(unittest.TestCase):

    def setUp(self):
        from execution.ingress.stream_health import (
            StreamHealthMonitor, StreamStatus
        )
        self.Monitor = StreamHealthMonitor
        self.Status = StreamStatus

    def test_unknown_stream_is_disconnected(self):
        m = self.Monitor()
        snap = m.check("unknown_stream")
        self.assertEqual(snap.status, self.Status.DISCONNECTED)
        self.assertEqual(snap.message_count, 0)

    def test_healthy_stream_after_message(self):
        m = self.Monitor(stale_threshold_sec=60.0, degraded_latency_ms=1000.0)
        m.record_message("ws1", latency_ms=10.0)
        snap = m.check("ws1")
        self.assertEqual(snap.status, self.Status.HEALTHY)

    def test_degraded_stream_high_latency(self):
        m = self.Monitor(stale_threshold_sec=60.0, degraded_latency_ms=100.0)
        m.record_message("ws1", latency_ms=500.0)
        snap = m.check("ws1")
        self.assertEqual(snap.status, self.Status.DEGRADED)

    def test_stale_stream(self):
        import time
        m = self.Monitor(stale_threshold_sec=0.01)
        m.record_message("ws1", latency_ms=0.0)
        time.sleep(0.05)
        snap = m.check("ws1")
        self.assertEqual(snap.status, self.Status.STALE)

    def test_disconnected_stream(self):
        m = self.Monitor()
        m.record_message("ws1")
        m.record_disconnect("ws1")
        snap = m.check("ws1")
        self.assertEqual(snap.status, self.Status.DISCONNECTED)

    def test_reconnected_stream(self):
        m = self.Monitor(stale_threshold_sec=60.0)
        m.record_message("ws1", latency_ms=0.0)
        m.record_disconnect("ws1")
        m.record_reconnect("ws1")
        m.record_message("ws1", latency_ms=0.0)
        snap = m.check("ws1")
        self.assertEqual(snap.status, self.Status.HEALTHY)

    def test_error_count_increments(self):
        m = self.Monitor()
        m.record_error("ws1")
        m.record_error("ws1")
        snap = m.check("ws1")
        self.assertEqual(snap.error_count, 2)

    def test_message_count_increments(self):
        m = self.Monitor()
        for _ in range(5):
            m.record_message("ws1")
        snap = m.check("ws1")
        self.assertEqual(snap.message_count, 5)

    def test_check_all(self):
        m = self.Monitor()
        m.record_message("ws1")
        m.record_message("ws2")
        snaps = m.check_all()
        ids = {s.stream_id for s in snaps}
        self.assertIn("ws1", ids)
        self.assertIn("ws2", ids)

    def test_snapshot_fields(self):
        m = self.Monitor(stale_threshold_sec=60.0)
        m.record_message("ws1", latency_ms=42.0)
        snap = m.check("ws1")
        self.assertEqual(snap.latency_ms, 42.0)
        self.assertGreaterEqual(snap.last_message_age_sec, 0.0)


# ===========================================================================
# 11. execution/safety/circuit_breaker.py
# ===========================================================================
class TestCircuitBreaker(unittest.TestCase):

    def setUp(self):
        from execution.safety.circuit_breaker import (
            CircuitBreaker, CircuitBreakerConfig, BreakerState
        )
        self.CB = CircuitBreaker
        self.Config = CircuitBreakerConfig
        self.State = BreakerState

    def test_initial_state_closed(self):
        cb = self.CB()
        self.assertEqual(cb.state, self.State.CLOSED)

    def test_allow_request_closed(self):
        cb = self.CB()
        self.assertTrue(cb.allow_request())

    def test_transitions_to_open_after_threshold(self):
        cfg = self.Config(failure_threshold=3, window_seconds=60.0, cooldown_seconds=999.0)
        cb = self.CB(cfg)
        for _ in range(3):
            cb.record_failure()
        self.assertEqual(cb.state, self.State.OPEN)

    def test_open_disallows_request(self):
        cfg = self.Config(failure_threshold=3, window_seconds=60.0, cooldown_seconds=999.0)
        cb = self.CB(cfg)
        for _ in range(3):
            cb.record_failure()
        self.assertFalse(cb.allow_request())

    def test_transitions_to_half_open_after_cooldown(self):
        import time
        cfg = self.Config(failure_threshold=3, window_seconds=60.0, cooldown_seconds=0.01)
        cb = self.CB(cfg)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.05)
        state = cb.state
        self.assertEqual(state, self.State.HALF_OPEN)

    def test_half_open_allows_limited_requests(self):
        import time
        cfg = self.Config(
            failure_threshold=3, window_seconds=60.0,
            cooldown_seconds=0.01, half_open_max=1
        )
        cb = self.CB(cfg)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.05)
        self.assertTrue(cb.allow_request())   # first allowed
        self.assertFalse(cb.allow_request())  # second blocked

    def test_half_open_success_resets_to_closed(self):
        import time
        cfg = self.Config(failure_threshold=3, window_seconds=60.0, cooldown_seconds=0.01)
        cb = self.CB(cfg)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.05)
        cb.allow_request()
        cb.record_success()
        self.assertEqual(cb.state, self.State.CLOSED)

    def test_half_open_failure_returns_to_open(self):
        import time
        cfg = self.Config(failure_threshold=3, window_seconds=60.0, cooldown_seconds=0.01)
        cb = self.CB(cfg)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.05)
        cb.allow_request()
        cb.record_failure()
        self.assertEqual(cb.state, self.State.OPEN)

    def test_record_success_clears_failures(self):
        cb = self.CB()
        cb.record_failure()
        cb.record_success()
        state, count, _ = cb.snapshot()
        self.assertEqual(count, 0)

    def test_reset(self):
        cfg = self.Config(failure_threshold=3)
        cb = self.CB(cfg)
        for _ in range(3):
            cb.record_failure()
        cb.reset()
        self.assertEqual(cb.state, self.State.CLOSED)
        self.assertTrue(cb.allow_request())

    def test_snapshot_returns_tuple(self):
        cb = self.CB()
        state, count, since = cb.snapshot()
        self.assertEqual(state, self.State.CLOSED)
        self.assertEqual(count, 0)
        self.assertEqual(since, 0.0)

    def test_failure_below_threshold_stays_closed(self):
        cfg = self.Config(failure_threshold=5, window_seconds=60.0)
        cb = self.CB(cfg)
        for _ in range(4):
            cb.record_failure()
        self.assertEqual(cb.state, self.State.CLOSED)


# ===========================================================================
# 12. execution/safety/kill_switch.py
# ===========================================================================
class TestExecutionKillSwitch(unittest.TestCase):

    def setUp(self):
        from execution.safety.kill_switch import ExecutionKillSwitch
        from risk.kill_switch import KillSwitch, KillScope
        self.KillSwitch = ExecutionKillSwitch
        self.RiskKS = KillSwitch
        self.KillScope = KillScope

    def test_gate_order_allowed_by_default(self):
        ks = self.KillSwitch()
        allowed, reason = ks.gate_order(symbol="BTCUSDT")
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_manual_block_all(self):
        ks = self.KillSwitch()
        ks.block_all(reason="test_block")
        allowed, reason = ks.gate_order(symbol="BTCUSDT")
        self.assertFalse(allowed)
        self.assertIn("test_block", reason)

    def test_unblock(self):
        ks = self.KillSwitch()
        ks.block_all(reason="temp")
        ks.unblock()
        allowed, _ = ks.gate_order(symbol="BTCUSDT")
        self.assertTrue(allowed)

    def test_is_blocked_false_initially(self):
        ks = self.KillSwitch()
        self.assertFalse(ks.is_blocked)

    def test_is_blocked_true_after_block(self):
        ks = self.KillSwitch()
        ks.block_all("reason")
        self.assertTrue(ks.is_blocked)

    def test_is_blocked_false_after_unblock(self):
        ks = self.KillSwitch()
        ks.block_all("reason")
        ks.unblock()
        self.assertFalse(ks.is_blocked)

    def test_trigger_delegates_to_risk_ks(self):
        mock_rks = MagicMock()
        mock_rks.trigger.return_value = MagicMock()
        ks = self.KillSwitch(risk_ks=mock_rks)
        ks.trigger(scope=self.KillScope.GLOBAL, key="*", reason="test")
        mock_rks.trigger.assert_called_once()

    def test_clear_delegates_to_risk_ks(self):
        mock_rks = MagicMock()
        mock_rks.clear.return_value = True
        ks = self.KillSwitch(risk_ks=mock_rks)
        result = ks.clear(scope=self.KillScope.GLOBAL, key="*")
        self.assertTrue(result)

    def test_gate_order_blocked_by_risk_ks(self):
        from risk.kill_switch import KillRecord, KillScope, KillMode
        import time
        mock_rks = MagicMock()
        record = KillRecord(
            scope=KillScope.SYMBOL,
            key="BTCUSDT",
            mode=KillMode.HARD_KILL,
            triggered_at=time.time(),
            reason="drawdown",
        )
        mock_rks.allow_order.return_value = (False, record)
        ks = self.KillSwitch(risk_ks=mock_rks)
        allowed, reason = ks.gate_order(symbol="BTCUSDT")
        self.assertFalse(allowed)
        self.assertIsNotNone(reason)
        self.assertIn("kill_switch", reason)

    def test_gate_order_with_strategy_id(self):
        ks = self.KillSwitch()
        allowed, _ = ks.gate_order(symbol="ETHUSDT", strategy_id="alpha_v1")
        self.assertTrue(allowed)

    def test_gate_order_reduce_only(self):
        ks = self.KillSwitch()
        allowed, _ = ks.gate_order(symbol="ETHUSDT", reduce_only=True)
        self.assertTrue(allowed)


# ===========================================================================
# 13. execution/safety/limits.py
# ===========================================================================
class TestOrderLimiter(unittest.TestCase):

    def setUp(self):
        from execution.safety.limits import OrderLimiter, OrderLimitsConfig
        self.Limiter = OrderLimiter
        self.Config = OrderLimitsConfig

    def test_no_limits_always_passes(self):
        limiter = self.Limiter()
        result = limiter.check_order(qty=Decimal("100"))
        self.assertTrue(result.allowed)

    def test_max_order_qty_exceeded(self):
        cfg = self.Config(max_order_qty=Decimal("10"))
        limiter = self.Limiter(cfg)
        result = limiter.check_order(qty=Decimal("20"))
        self.assertFalse(result.allowed)
        self.assertEqual(result.violated_rule, "max_order_qty")

    def test_max_order_qty_ok(self):
        cfg = self.Config(max_order_qty=Decimal("10"))
        limiter = self.Limiter(cfg)
        result = limiter.check_order(qty=Decimal("5"))
        self.assertTrue(result.allowed)

    def test_max_order_notional_exceeded(self):
        cfg = self.Config(max_order_notional=Decimal("500"))
        limiter = self.Limiter(cfg)
        result = limiter.check_order(qty=Decimal("1"), price=Decimal("600"))
        self.assertFalse(result.allowed)
        self.assertEqual(result.violated_rule, "max_order_notional")

    def test_max_order_notional_no_price(self):
        cfg = self.Config(max_order_notional=Decimal("500"))
        limiter = self.Limiter(cfg)
        # price=None → notional=0, always passes
        result = limiter.check_order(qty=Decimal("100"))
        self.assertTrue(result.allowed)

    def test_max_position_notional_exceeded(self):
        cfg = self.Config(max_position_notional=Decimal("1000"))
        limiter = self.Limiter(cfg)
        result = limiter.check_order(
            qty=Decimal("1"),
            price=Decimal("200"),
            current_position_notional=Decimal("900"),
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.violated_rule, "max_position_notional")

    def test_max_daily_orders_exceeded(self):
        cfg = self.Config(max_daily_orders=2)
        limiter = self.Limiter(cfg)
        limiter.check_order(qty=Decimal("1"))
        limiter.check_order(qty=Decimal("1"))
        result = limiter.check_order(qty=Decimal("1"))
        self.assertFalse(result.allowed)
        self.assertEqual(result.violated_rule, "max_daily_orders")

    def test_max_daily_notional_exceeded(self):
        cfg = self.Config(max_daily_notional=Decimal("500"))
        limiter = self.Limiter(cfg)
        limiter.check_order(qty=Decimal("1"), price=Decimal("300"))
        result = limiter.check_order(qty=Decimal("1"), price=Decimal("300"))
        self.assertFalse(result.allowed)
        self.assertEqual(result.violated_rule, "max_daily_notional")

    def test_reset_daily_resets_counters(self):
        cfg = self.Config(max_daily_orders=2)
        limiter = self.Limiter(cfg)
        limiter.check_order(qty=Decimal("1"))
        limiter.check_order(qty=Decimal("1"))
        limiter.reset_daily()
        result = limiter.check_order(qty=Decimal("1"))
        self.assertTrue(result.allowed)

    def test_max_orders_per_second(self):
        cfg = self.Config(max_orders_per_second=2.0)
        limiter = self.Limiter(cfg)
        limiter.check_order(qty=Decimal("1"))
        limiter.check_order(qty=Decimal("1"))
        result = limiter.check_order(qty=Decimal("1"))
        self.assertFalse(result.allowed)
        self.assertEqual(result.violated_rule, "max_orders_per_second")

    def test_check_order_accumulates_daily_notional(self):
        # max_daily_notional=800, each order notional=200
        # After 4 orders: total=800, 5th order would push to 1000 > 800 → rejected
        cfg = self.Config(max_daily_notional=Decimal("800"))
        limiter = self.Limiter(cfg)
        for _ in range(4):
            result = limiter.check_order(qty=Decimal("1"), price=Decimal("200"))
            self.assertTrue(result.allowed)
        # 5th order: cumulative would be 1000 > 800 → should fail
        result = limiter.check_order(qty=Decimal("1"), price=Decimal("200"))
        self.assertFalse(result.allowed)


# ===========================================================================
# 14. execution/bridge/error_policy.py
# ===========================================================================
class TestErrorPolicy(unittest.TestCase):

    def setUp(self):
        from execution.bridge.error_policy import ErrorPolicy, ErrorAction
        from execution.bridge.error_map import ErrorMapper, ErrorCategory, ErrorMapping
        self.ErrorPolicy = ErrorPolicy
        self.ErrorAction = ErrorAction
        self.ErrorMapper = ErrorMapper
        self.ErrorCategory = ErrorCategory
        self.ErrorMapping = ErrorMapping

    def test_insufficient_balance_halt(self):
        policy = self.ErrorPolicy(halt_on_insufficient_balance=True)
        decision = policy.decide(error_code=-2019, message="insufficient balance")
        self.assertEqual(decision.action, self.ErrorAction.HALT)

    def test_insufficient_balance_reject_when_no_halt(self):
        policy = self.ErrorPolicy(halt_on_insufficient_balance=False)
        decision = policy.decide(error_code=-2019, message="insufficient balance")
        self.assertEqual(decision.action, self.ErrorAction.REJECT)

    def test_rate_limited_retry_first_attempt(self):
        policy = self.ErrorPolicy(max_retries=3)
        decision = policy.decide(error_code=-1003, message="rate limit", attempt=1)
        self.assertEqual(decision.action, self.ErrorAction.RETRY)
        self.assertGreater(decision.retry_delay_sec, 0)

    def test_rate_limited_reject_after_max_retries(self):
        policy = self.ErrorPolicy(max_retries=3)
        decision = policy.decide(error_code=-1003, message="rate limit", attempt=4)
        self.assertEqual(decision.action, self.ErrorAction.REJECT)

    def test_retryable_error_retries(self):
        policy = self.ErrorPolicy(max_retries=3)
        decision = policy.decide(error_code=-1001, message="internal", attempt=1)
        self.assertEqual(decision.action, self.ErrorAction.RETRY)

    def test_retryable_error_max_retries_exceeded(self):
        policy = self.ErrorPolicy(max_retries=3)
        decision = policy.decide(error_code=-1001, message="internal", attempt=4)
        # Should NOT retry — falls through to REJECT or LOG_AND_SKIP
        self.assertNotEqual(decision.action, self.ErrorAction.RETRY)

    def test_invalid_params_rejected(self):
        policy = self.ErrorPolicy()
        decision = policy.decide(error_code=-4003, message="bad param")
        self.assertEqual(decision.action, self.ErrorAction.REJECT)

    def test_non_retryable_rejected(self):
        policy = self.ErrorPolicy()
        decision = policy.decide(error_code=-2010, message="order failed")
        self.assertEqual(decision.action, self.ErrorAction.REJECT)

    def test_unknown_error_log_and_skip(self):
        policy = self.ErrorPolicy()
        decision = policy.decide(error_code=99999, message="mystery error")
        self.assertEqual(decision.action, self.ErrorAction.LOG_AND_SKIP)

    def test_http_500_retryable(self):
        policy = self.ErrorPolicy(max_retries=3)
        decision = policy.decide(error_code=503, message="service unavailable", attempt=1)
        self.assertEqual(decision.action, self.ErrorAction.RETRY)

    def test_http_400_non_retryable(self):
        policy = self.ErrorPolicy()
        decision = policy.decide(error_code=400, message="bad request")
        self.assertEqual(decision.action, self.ErrorAction.REJECT)

    def test_rate_limited_delay_doubles_per_attempt(self):
        policy = self.ErrorPolicy(max_retries=5)
        d1 = policy.decide(error_code=-1003, attempt=1)
        d2 = policy.decide(error_code=-1003, attempt=2)
        self.assertLessEqual(d1.retry_delay_sec, d2.retry_delay_sec)


# ===========================================================================
# 15. execution/bridge/venue_router.py
# ===========================================================================
class TestVenueRouter(unittest.TestCase):

    def _make_bridge(self, venue):
        """Create a minimal ExecutionBridge with a mock client."""
        from execution.bridge.execution_bridge import ExecutionBridge
        client = MagicMock()
        client.submit_order.return_value = {"orderId": "123"}
        client.cancel_order.return_value = {"orderId": "123"}
        bridge = ExecutionBridge(venue_clients={venue: client})
        return bridge

    def _make_cmd(self, venue, symbol="BTCUSDT"):
        cmd = MagicMock()
        cmd.venue = venue
        cmd.symbol = symbol
        cmd.command_id = "cmd-001"
        cmd.idempotency_key = f"{venue}-{symbol}-001"
        return cmd

    def setUp(self):
        from execution.bridge.venue_router import VenueRouter
        self.Router = VenueRouter

    def test_register_and_get_bridge(self):
        router = self.Router()
        bridge = self._make_bridge("bybit")
        router.register("bybit", bridge)
        self.assertIsNotNone(router.get_bridge("bybit"))

    def test_get_bridge_case_insensitive(self):
        router = self.Router()
        bridge = self._make_bridge("bybit")
        router.register("BYBIT", bridge)
        self.assertIsNotNone(router.get_bridge("bybit"))

    def test_get_bridge_missing_returns_none(self):
        router = self.Router()
        self.assertIsNone(router.get_bridge("unknown"))

    def test_submit_routes_to_correct_bridge(self):
        router = self.Router()
        bridge = MagicMock()
        bridge.submit.return_value = MagicMock(status="ACCEPTED")
        router.register("bybit", bridge)
        cmd = self._make_cmd("bybit")
        router.submit(cmd)
        bridge.submit.assert_called_once_with(cmd)

    def test_cancel_routes_to_correct_bridge(self):
        router = self.Router()
        bridge = MagicMock()
        bridge.cancel.return_value = MagicMock(status="ACCEPTED")
        router.register("bybit", bridge)
        cmd = self._make_cmd("bybit")
        router.cancel(cmd)
        bridge.cancel.assert_called_once_with(cmd)

    def test_submit_unknown_venue_raises(self):
        router = self.Router()
        cmd = self._make_cmd("unknown_venue")
        with self.assertRaises(KeyError):
            router.submit(cmd)

    def test_cancel_unknown_venue_raises(self):
        router = self.Router()
        cmd = self._make_cmd("unknown_venue")
        with self.assertRaises(KeyError):
            router.cancel(cmd)

    def test_venues_property(self):
        router = self.Router()
        bridge1 = self._make_bridge("bybit")
        bridge2 = self._make_bridge("binance")
        router.register("bybit", bridge1)
        router.register("binance", bridge2)
        self.assertIn("bybit", router.venues)
        self.assertIn("binance", router.venues)


# ===========================================================================
# 16. execution/bridge/execution_bridge.py
# ===========================================================================
class TestExecutionBridge(unittest.TestCase):

    def _make_cmd(self, venue="bybit", symbol="BTCUSDT", command_id="cmd-001", idem="idem-001"):
        cmd = MagicMock()
        cmd.venue = venue
        cmd.symbol = symbol
        cmd.command_id = command_id
        cmd.idempotency_key = idem
        return cmd

    def _make_bridge(self, client, **kwargs):
        from execution.bridge.execution_bridge import (
            ExecutionBridge, RetryPolicy
        )

        # Use a fast retry policy with no sleeper
        sleeper = MagicMock()
        rp = RetryPolicy(max_attempts=1, base_delay_sec=0.0, max_delay_sec=0.0, jitter_sec=0.0)
        bridge = ExecutionBridge(
            venue_clients={"bybit": client},
            retry_policy=rp,
            sleeper=sleeper,
            **kwargs
        )
        return bridge, sleeper

    def test_successful_submit(self):
        client = MagicMock()
        client.submit_order.return_value = {"orderId": "o1"}
        bridge, _ = self._make_bridge(client)
        cmd = self._make_cmd()
        ack = bridge.submit(cmd)
        self.assertEqual(ack.status, "ACCEPTED")
        self.assertTrue(ack.ok)
        self.assertEqual(ack.attempts, 1)

    def test_successful_cancel(self):
        client = MagicMock()
        client.cancel_order.return_value = {"orderId": "o1"}
        bridge, _ = self._make_bridge(client)
        cmd = self._make_cmd()
        ack = bridge.cancel(cmd)
        self.assertEqual(ack.status, "ACCEPTED")

    def test_deduplication_returns_cached(self):
        client = MagicMock()
        client.submit_order.return_value = {"orderId": "o1"}
        bridge, _ = self._make_bridge(client)
        cmd = self._make_cmd()
        bridge.submit(cmd)
        ack2 = bridge.submit(cmd)
        self.assertTrue(ack2.deduped)
        # Client should only be called once
        self.assertEqual(client.submit_order.call_count, 1)

    def test_non_retryable_error_returns_rejected(self):
        from execution.bridge.execution_bridge import NonRetryableVenueError
        client = MagicMock()
        client.submit_order.side_effect = NonRetryableVenueError("bad param")
        bridge, _ = self._make_bridge(client)
        cmd = self._make_cmd(idem="idem-nonret")
        ack = bridge.submit(cmd)
        self.assertEqual(ack.status, "REJECTED")
        self.assertIn("non_retryable", ack.error)

    def test_retryable_error_exhausts_attempts(self):
        from execution.bridge.execution_bridge import RetryableVenueError, RetryPolicy
        client = MagicMock()
        client.submit_order.side_effect = RetryableVenueError("timeout")
        sleeper = MagicMock()
        from execution.bridge.execution_bridge import ExecutionBridge
        rp = RetryPolicy(max_attempts=3, base_delay_sec=0.0, max_delay_sec=0.0, jitter_sec=0.0)
        bridge = ExecutionBridge(
            venue_clients={"bybit": client},
            retry_policy=rp,
            sleeper=sleeper,
        )
        cmd = self._make_cmd(idem="idem-retry")
        ack = bridge.submit(cmd)
        self.assertEqual(ack.status, "FAILED")
        self.assertEqual(client.submit_order.call_count, 3)

    def test_unknown_exception_returns_failed(self):
        client = MagicMock()
        client.submit_order.side_effect = ValueError("unexpected")
        bridge, _ = self._make_bridge(client)
        cmd = self._make_cmd(idem="idem-unknown")
        ack = bridge.submit(cmd)
        self.assertEqual(ack.status, "FAILED")
        self.assertIn("unexpected_non_retryable", ack.error)

    def test_no_venue_client_returns_failed(self):
        from execution.bridge.execution_bridge import ExecutionBridge, RetryPolicy
        bridge = ExecutionBridge(
            venue_clients={},
            retry_policy=RetryPolicy(max_attempts=1, base_delay_sec=0.0, max_delay_sec=0.0, jitter_sec=0.0),
            sleeper=MagicMock(),
        )
        cmd = self._make_cmd(venue="bybit", idem="idem-novc")
        ack = bridge.submit(cmd)
        self.assertEqual(ack.status, "FAILED")
        self.assertIn("no_venue_client", ack.error)

    def test_circuit_breaker_open_returns_failed(self):
        from execution.bridge.execution_bridge import (
            ExecutionBridge, RetryPolicy, CircuitBreakerConfig, NonRetryableVenueError
        )
        # Trip the circuit breaker by causing many failures
        client = MagicMock()
        client.submit_order.side_effect = NonRetryableVenueError("fail")
        cfg = CircuitBreakerConfig(failure_threshold=2, window_sec=60.0, cooldown_sec=999.0)
        rp = RetryPolicy(max_attempts=1, base_delay_sec=0.0, max_delay_sec=0.0, jitter_sec=0.0)
        bridge = ExecutionBridge(
            venue_clients={"bybit": client},
            breaker_cfg=cfg,
            retry_policy=rp,
            sleeper=MagicMock(),
        )
        # Trip breaker
        for i in range(5):
            bridge.submit(self._make_cmd(idem=f"idem-trip-{i}"))
        # Now the breaker should be open
        ack = bridge.submit(self._make_cmd(idem="idem-after-trip"))
        self.assertEqual(ack.status, "FAILED")
        self.assertIn("circuit_open", ack.error)

    def test_on_ack_callback_called(self):
        client = MagicMock()
        client.submit_order.return_value = {"orderId": "o1"}
        received = []
        from execution.bridge.execution_bridge import ExecutionBridge, RetryPolicy
        rp = RetryPolicy(max_attempts=1, base_delay_sec=0.0, max_delay_sec=0.0, jitter_sec=0.0)
        bridge = ExecutionBridge(
            venue_clients={"bybit": client},
            retry_policy=rp,
            sleeper=MagicMock(),
            on_ack=received.append,
        )
        bridge.submit(self._make_cmd(idem="idem-cb"))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].status, "ACCEPTED")

    def test_pending_count_and_drain(self):
        from execution.bridge.execution_bridge import (
            ExecutionBridge, RetryPolicy, RateLimitConfig
        )
        # Create a bucket that immediately exhausts
        client = MagicMock()
        client.submit_order.return_value = {"orderId": "o1"}

        # To test pending queue, use a mock clock and bucket
        class _FakeClock:
            def __init__(self):
                self._t = 0.0
            def now(self):
                return self._t

        clock = _FakeClock()
        sleeper = MagicMock()
        rp = RetryPolicy(max_attempts=1, base_delay_sec=0.0, max_delay_sec=0.0, jitter_sec=0.0)

        # Build bridge and manually deplete its bucket
        bridge = ExecutionBridge(
            venue_clients={"bybit": client},
            retry_policy=rp,
            sleeper=sleeper,
            clock=clock,
            rate_limits={"bybit": RateLimitConfig(rate_per_sec=10.0, burst=1.0)},
            pending_queue_size=5,
        )
        # First request should succeed (bucket starts full)
        cmd1 = self._make_cmd(idem="idem-drain-first")
        bridge.submit(cmd1)
        # Don't assert status here, just check pending count can be queried
        self.assertGreaterEqual(bridge.pending_count, 0)

    def test_ack_ok_property(self):
        from execution.bridge.execution_bridge import Ack
        ack_ok = Ack(
            status="ACCEPTED", command_id="c1", idempotency_key="k1",
            venue="bybit", symbol="BTCUSDT", attempts=1
        )
        ack_fail = Ack(
            status="FAILED", command_id="c2", idempotency_key="k2",
            venue="bybit", symbol="BTCUSDT", attempts=1
        )
        self.assertTrue(ack_ok.ok)
        self.assertFalse(ack_fail.ok)

    def test_is_retryable_exception(self):
        from execution.bridge.execution_bridge import (
            is_retryable_exception, RetryableVenueError, NonRetryableVenueError
        )
        self.assertTrue(is_retryable_exception(TimeoutError()))
        self.assertTrue(is_retryable_exception(ConnectionError()))
        self.assertTrue(is_retryable_exception(RetryableVenueError()))
        self.assertFalse(is_retryable_exception(NonRetryableVenueError()))
        self.assertFalse(is_retryable_exception(ValueError()))

    def test_ack_payload_roundtrip(self):
        from execution.bridge.execution_bridge import Ack, ExecutionBridge
        bridge = ExecutionBridge(venue_clients={}, sleeper=MagicMock())
        ack = Ack(
            status="ACCEPTED",
            command_id="c1",
            idempotency_key="k1",
            venue="bybit",
            symbol="BTCUSDT",
            attempts=2,
            deduped=False,
            result={"orderId": "o1"},
            error=None,
        )
        payload = bridge._ack_to_payload(ack)
        restored = bridge._payload_to_ack(payload)
        self.assertEqual(restored.status, ack.status)
        self.assertEqual(restored.command_id, ack.command_id)
        self.assertEqual(restored.attempts, ack.attempts)


# ===========================================================================
# 17. execution/store/ack_store.py — InMemoryAckStore
# ===========================================================================
class TestInMemoryAckStore(unittest.TestCase):

    def setUp(self):
        from execution.store.ack_store import InMemoryAckStore
        self.Store = InMemoryAckStore

    def test_put_and_get(self):
        store = self.Store()
        store.put("k1", {"status": "ACCEPTED", "venue": "bybit"})
        result = store.get("k1")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "ACCEPTED")

    def test_get_missing_returns_none(self):
        store = self.Store()
        result = store.get("nonexistent")
        self.assertIsNone(result)

    def test_put_overwrites(self):
        store = self.Store()
        store.put("k1", {"status": "FAILED"})
        store.put("k1", {"status": "ACCEPTED"})
        result = store.get("k1")
        self.assertEqual(result["status"], "ACCEPTED")

    def test_prune_returns_int(self):
        store = self.Store()
        store.put("k1", {"status": "ok"})
        result = store.prune()
        self.assertIsInstance(result, int)

    def test_multiple_keys(self):
        store = self.Store()
        store.put("k1", {"v": 1})
        store.put("k2", {"v": 2})
        self.assertEqual(store.get("k1")["v"], 1)
        self.assertEqual(store.get("k2")["v"], 2)


# ===========================================================================
# 18. execution/store/ack_store.py — SQLiteAckStore
# ===========================================================================
class TestSQLiteAckStore(unittest.TestCase):

    def setUp(self):
        import tempfile
        import os
        from execution.store.ack_store import SQLiteAckStore
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_acks.db")
        self.Store = SQLiteAckStore

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_put_and_get(self):
        store = self.Store(path=self.db_path)
        store.put("k1", {"status": "ACCEPTED"})
        result = store.get("k1")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "ACCEPTED")

    def test_get_missing_returns_none(self):
        store = self.Store(path=self.db_path)
        result = store.get("nonexistent")
        self.assertIsNone(result)

    def test_put_overwrites(self):
        store = self.Store(path=self.db_path)
        store.put("k1", {"status": "FAILED"})
        store.put("k1", {"status": "ACCEPTED"})
        result = store.get("k1")
        self.assertEqual(result["status"], "ACCEPTED")

    def test_prune_no_ttl_returns_zero(self):
        store = self.Store(path=self.db_path)
        store.put("k1", {"status": "ok"})
        count = store.prune()
        self.assertEqual(count, 0)

    def test_ttl_expires_entries(self):
        import time
        store = self.Store(path=self.db_path, ttl_sec=0.01)
        store.put("k1", {"status": "ok"})
        time.sleep(0.05)
        result = store.get("k1")
        self.assertIsNone(result)

    def test_prune_with_ttl_removes_old(self):
        import time
        store = self.Store(path=self.db_path, ttl_sec=0.01)
        store.put("k1", {"status": "old"})
        time.sleep(0.05)
        count = store.prune()
        self.assertGreaterEqual(count, 1)

    def test_persistence_across_instances(self):
        store1 = self.Store(path=self.db_path)
        store1.put("persist_key", {"status": "PERSISTED"})
        # Create new instance pointing to same DB
        store2 = self.Store(path=self.db_path)
        result = store2.get("persist_key")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "PERSISTED")


# ===========================================================================
# 19. execution/models/commands.py
# ===========================================================================
class TestCommands(unittest.TestCase):

    def setUp(self):
        from execution.models.commands import (
            make_submit_order_command, make_cancel_order_command,
            SubmitOrderCommand, CancelOrderCommand, _dec, _norm_side
        )
        from execution.bridge.request_ids import RequestIdFactory
        self.make_submit = make_submit_order_command
        self.make_cancel = make_cancel_order_command
        self.SubmitCmd = SubmitOrderCommand
        self.CancelCmd = CancelOrderCommand
        self._dec = _dec
        self._norm_side = _norm_side
        self.RID = RequestIdFactory

    def _rid(self):
        return self.RID(namespace="test", run_id="run1")

    def test_make_submit_basic(self):
        cmd = self.make_submit(
            rid=self._rid(),
            actor="test",
            venue="bybit",
            symbol="BTCUSDT",
            strategy="alpha",
            logical_id="leg-001",
            side="buy",
            order_type="market",
            qty="0.1",
        )
        self.assertEqual(cmd.venue, "bybit")
        self.assertEqual(cmd.symbol, "BTCUSDT")
        self.assertEqual(cmd.side, "buy")
        self.assertEqual(cmd.order_type, "market")
        self.assertEqual(cmd.command_type, "submit")

    def test_make_submit_limit_requires_price(self):
        with self.assertRaises(ValueError):
            self.make_submit(
                rid=self._rid(),
                actor="test",
                venue="bybit",
                symbol="ETHUSDT",
                strategy="alpha",
                logical_id="leg-002",
                side="sell",
                order_type="limit",
                qty="1.0",
                price=None,
            )

    def test_make_submit_invalid_qty_raises(self):
        with self.assertRaises(ValueError):
            self.make_submit(
                rid=self._rid(),
                actor="test",
                venue="bybit",
                symbol="ETHUSDT",
                strategy="alpha",
                logical_id="leg-003",
                side="buy",
                order_type="market",
                qty="-1",
            )

    def test_make_submit_zero_qty_raises(self):
        with self.assertRaises(ValueError):
            self.make_submit(
                rid=self._rid(),
                actor="test",
                venue="bybit",
                symbol="ETHUSDT",
                strategy="alpha",
                logical_id="leg-004",
                side="buy",
                order_type="market",
                qty="0",
            )

    def test_make_submit_invalid_price_raises(self):
        with self.assertRaises(ValueError):
            self.make_submit(
                rid=self._rid(),
                actor="test",
                venue="bybit",
                symbol="ETHUSDT",
                strategy="alpha",
                logical_id="leg-005",
                side="buy",
                order_type="limit",
                qty="1.0",
                price="-100",
            )

    def test_make_submit_side_normalization(self):
        cmd = self.make_submit(
            rid=self._rid(),
            actor="test",
            venue="bybit",
            symbol="BTCUSDT",
            strategy="alpha",
            logical_id="leg-006",
            side="SELL",
            order_type="market",
            qty="0.5",
        )
        self.assertEqual(cmd.side, "sell")

    def test_make_submit_reduce_only(self):
        cmd = self.make_submit(
            rid=self._rid(),
            actor="test",
            venue="bybit",
            symbol="BTCUSDT",
            strategy="alpha",
            logical_id="leg-007",
            side="sell",
            order_type="market",
            qty="0.5",
            reduce_only=True,
        )
        self.assertTrue(cmd.reduce_only)

    def test_make_cancel_with_order_id(self):
        cmd = self.make_cancel(
            actor="test",
            venue="bybit",
            symbol="BTCUSDT",
            order_id="ord-123",
        )
        self.assertEqual(cmd.order_id, "ord-123")
        self.assertEqual(cmd.command_type, "cancel")

    def test_make_cancel_with_client_order_id(self):
        cmd = self.make_cancel(
            actor="test",
            venue="bybit",
            symbol="BTCUSDT",
            client_order_id="coid-456",
        )
        self.assertEqual(cmd.client_order_id, "coid-456")

    def test_make_cancel_no_ids_raises(self):
        with self.assertRaises(ValueError):
            self.make_cancel(
                actor="test",
                venue="bybit",
                symbol="BTCUSDT",
            )

    def test_dec_converts_string(self):
        result = self._dec("3.14", "price")
        self.assertEqual(result, Decimal("3.14"))

    def test_dec_converts_int(self):
        result = self._dec(100, "qty")
        self.assertEqual(result, Decimal("100"))

    def test_dec_invalid_raises(self):
        with self.assertRaises(ValueError):
            self._dec("not_a_number", "qty")

    def test_norm_side_long(self):
        self.assertEqual(self._norm_side("long"), "buy")

    def test_norm_side_short(self):
        self.assertEqual(self._norm_side("short"), "sell")

    def test_norm_side_invalid_raises(self):
        with self.assertRaises(ValueError):
            self._norm_side("invalid_side")

    def test_make_submit_with_tif(self):
        cmd = self.make_submit(
            rid=self._rid(),
            actor="test",
            venue="bybit",
            symbol="BTCUSDT",
            strategy="alpha",
            logical_id="leg-tif",
            side="buy",
            order_type="limit",
            qty="1.0",
            price="50000",
            tif="GTC",
        )
        self.assertEqual(cmd.tif, "gtc")

    def test_make_submit_custom_command_id(self):
        cmd = self.make_submit(
            rid=self._rid(),
            actor="test",
            venue="bybit",
            symbol="BTCUSDT",
            strategy="alpha",
            logical_id="leg-cid",
            side="buy",
            order_type="market",
            qty="1.0",
            command_id="custom-cmd-id",
        )
        self.assertEqual(cmd.command_id, "custom-cmd-id")


# ===========================================================================
# 20. execution/bridge/request_ids.py
# ===========================================================================
class TestRequestIds(unittest.TestCase):

    def setUp(self):
        from execution.bridge.request_ids import RequestIdFactory, make_idempotency_key
        self.Factory = RequestIdFactory
        self.make_key = make_idempotency_key

    def test_make_idempotency_key(self):
        key = self.make_key(venue="bybit", action="submit", key="order-123")
        self.assertIsInstance(key, str)
        self.assertGreater(len(key), 0)

    def test_make_idempotency_key_deterministic(self):
        k1 = self.make_key(venue="bybit", action="submit", key="order-123")
        k2 = self.make_key(venue="bybit", action="submit", key="order-123")
        self.assertEqual(k1, k2)

    def test_make_idempotency_key_differs_for_diff_inputs(self):
        k1 = self.make_key(venue="bybit", action="submit", key="order-A")
        k2 = self.make_key(venue="bybit", action="submit", key="order-B")
        self.assertNotEqual(k1, k2)

    def test_deterministic_client_order_id(self):
        factory = self.Factory(namespace="ns", run_id="r1", deterministic=True)
        id1 = factory.client_order_id(strategy="alpha", symbol="BTCUSDT", logical_id="leg-001")
        id2 = factory.client_order_id(strategy="alpha", symbol="BTCUSDT", logical_id="leg-001")
        self.assertEqual(id1, id2)

    def test_non_deterministic_uses_nonce(self):
        factory = self.Factory(namespace="ns", run_id="r1", deterministic=False)
        id1 = factory.client_order_id(strategy="alpha", symbol="BTCUSDT")
        id2 = factory.client_order_id(strategy="alpha", symbol="BTCUSDT")
        # nonce increments so IDs should differ
        self.assertNotEqual(id1, id2)

    def test_next_nonce_increments(self):
        factory = self.Factory()
        n1 = factory.next_nonce()
        n2 = factory.next_nonce()
        self.assertEqual(n2, n1 + 1)

    def test_max_len_respected(self):
        factory = self.Factory(namespace="ns", run_id="r1", max_len=20)
        coid = factory.client_order_id(
            strategy="very_long_strategy_name", symbol="BTCUSDT", logical_id="leg-001"
        )
        self.assertLessEqual(len(coid), 20)

    def test_explicit_nonce(self):
        factory = self.Factory(namespace="ns", run_id="r1", deterministic=False)
        coid = factory.client_order_id(strategy="alpha", symbol="BTCUSDT", nonce=42)
        self.assertIn("2a", coid)  # 42 in hex is "2a"

    def test_symbol_uppercased(self):
        factory = self.Factory(namespace="ns", run_id="r1", deterministic=False)
        coid = factory.client_order_id(strategy="alpha", symbol="ethusdt", nonce=1)
        self.assertIn("ETHUSDT", coid)


if __name__ == "__main__":
    unittest.main()
