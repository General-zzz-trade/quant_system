"""Tests for structured logging with context propagation."""
from __future__ import annotations

import json
import logging


from infra.logging.structured import (
    JSONFormatter,
    LogContext,
    StructuredLogger,
)


# ── StructuredLogger tests ───────────────────────────────────


class TestStructuredLogger:
    def test_info_outputs_json(self, caplog):
        with caplog.at_level(logging.INFO):
            log = StructuredLogger("test.structured")
            log.info("hello world")

        assert len(caplog.records) == 1
        payload = json.loads(caplog.records[0].message)
        assert payload["msg"] == "hello world"
        assert payload["level"] == "INFO"
        assert "ts" in payload

    def test_extra_fields_included(self, caplog):
        with caplog.at_level(logging.INFO):
            log = StructuredLogger("test.extra")
            log.info("trade", symbol="BTCUSDT", price=42000.0)

        payload = json.loads(caplog.records[0].message)
        assert payload["symbol"] == "BTCUSDT"
        assert payload["price"] == 42000.0

    def test_context_propagation(self, caplog):
        ctx = LogContext(trace_id="abc123", strategy="momentum")
        with caplog.at_level(logging.INFO):
            log = StructuredLogger("test.ctx", default_context=ctx)
            log.info("signal")

        payload = json.loads(caplog.records[0].message)
        assert payload["trace_id"] == "abc123"
        assert payload["strategy"] == "momentum"

    def test_with_context_creates_child(self, caplog):
        log = StructuredLogger("test.child", default_context=LogContext(trace_id="t1"))
        child = log.with_context(symbol="ETHUSDT")

        with caplog.at_level(logging.INFO):
            child.info("data")

        payload = json.loads(caplog.records[0].message)
        assert payload["trace_id"] == "t1"
        assert payload["symbol"] == "ETHUSDT"

    def test_empty_fields_omitted(self, caplog):
        with caplog.at_level(logging.INFO):
            log = StructuredLogger("test.empty")
            log.info("no context")

        payload = json.loads(caplog.records[0].message)
        assert "trace_id" not in payload
        assert "span_id" not in payload
        assert "symbol" not in payload

    def test_warning_level(self, caplog):
        with caplog.at_level(logging.WARNING):
            log = StructuredLogger("test.warn")
            log.warning("low balance", balance=50.0)

        payload = json.loads(caplog.records[0].message)
        assert payload["level"] == "WARNING"
        assert payload["balance"] == 50.0

    def test_error_level(self, caplog):
        with caplog.at_level(logging.ERROR):
            log = StructuredLogger("test.err")
            log.error("connection lost", retry=3)

        payload = json.loads(caplog.records[0].message)
        assert payload["level"] == "ERROR"

    def test_context_property(self):
        ctx = LogContext(trace_id="xyz")
        log = StructuredLogger("test.prop", default_context=ctx)
        assert log.context.trace_id == "xyz"


# ── JSONFormatter tests ──────────────────────────────────────


class TestJSONFormatter:
    def test_formats_plain_message_as_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="simple message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        payload = json.loads(output)
        assert payload["msg"] == "simple message"
        assert "logger" in payload

    def test_parses_json_message(self):
        formatter = JSONFormatter()
        inner = json.dumps({"msg": "structured", "level": "INFO", "ts": "2024-01-01T00:00:00"})
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=inner, args=(), exc_info=None,
        )
        output = formatter.format(record)
        payload = json.loads(output)
        assert payload["msg"] == "structured"

    def test_includes_exception_info(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="error occurred", args=(), exc_info=exc_info,
        )
        output = formatter.format(record)
        payload = json.loads(output)
        assert "exception" in payload
        assert payload["exception"]["type"] == "ValueError"
