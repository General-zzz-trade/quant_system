"""Tests for infra.logging.structured."""
from __future__ import annotations

import json
import logging

from infra.logging.structured import JsonFormatter, setup_structured_logging


class TestJsonFormatter:

    def test_formats_as_json(self) -> None:
        fmt = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )

        output = fmt.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "hello world"
        assert "ts" in data

    def test_includes_extra_fields(self) -> None:
        fmt = JsonFormatter(extra_fields={"env": "test"})
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )

        data = json.loads(fmt.format(record))
        assert data["env"] == "test"

    def test_includes_trading_fields(self) -> None:
        fmt = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="fill", args=(), exc_info=None,
        )
        record.symbol = "BTCUSDT"  # type: ignore[attr-defined]
        record.side = "buy"  # type: ignore[attr-defined]

        data = json.loads(fmt.format(record))
        assert data["symbol"] == "BTCUSDT"
        assert data["side"] == "buy"

    def test_includes_exception_info(self) -> None:
        fmt = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="error", args=(), exc_info=exc_info,
        )

        data = json.loads(fmt.format(record))
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"


class TestSetupStructuredLogging:

    def test_returns_logger(self) -> None:
        logger = setup_structured_logging(level="DEBUG")
        assert logger.name == "quant_system"
        assert logger.level == logging.DEBUG
