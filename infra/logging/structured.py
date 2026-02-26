"""Structured JSON logging for production environments.

Outputs each log record as a single JSON line — easy to ingest into
Elasticsearch, CloudWatch, Loki, etc.

Usage:
    from infra.logging.structured import setup_structured_logging
    setup_structured_logging(level="INFO", log_file="app.log")

    # Or use the StructuredLogger with context propagation:
    from infra.logging.structured import StructuredLogger, LogContext
    ctx = LogContext(trace_id="abc", strategy="momentum")
    log = StructuredLogger("my_module", default_context=ctx)
    log.info("signal generated", symbol="BTCUSDT", score=0.85)
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def __init__(self, *, extra_fields: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._extra = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Merge extra fields from LogRecord
        for key in ("symbol", "side", "qty", "price", "order_id", "fill_id",
                     "actor", "event_type", "correlation_id", "run_id"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        payload.update(self._extra)
        return json.dumps(payload, default=str)


def setup_structured_logging(
    *,
    level: str = "INFO",
    log_file: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """Configure the quant_system root logger with JSON output.

    Returns the configured logger.
    """
    logger = logging.getLogger("quant_system")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = JsonFormatter(extra_fields=extra_fields)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # Optional file handler
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ── Context-aware structured logger ─────────────────────────


@dataclass(frozen=True, slots=True)
class LogContext:
    """Propagated context fields attached to every log record."""

    trace_id: str = ""
    span_id: str = ""
    order_id: str = ""
    symbol: str = ""
    strategy: str = ""


class StructuredLogger:
    """JSON structured logger with context propagation.

    Wraps the stdlib logger and enriches each record with trace/span IDs,
    symbol, strategy, and arbitrary extra fields.
    """

    def __init__(
        self,
        name: str,
        *,
        default_context: Optional[LogContext] = None,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._context = default_context or LogContext()

    @property
    def context(self) -> LogContext:
        return self._context

    def with_context(self, **kwargs: Any) -> StructuredLogger:
        """Create child logger with additional/overridden context fields."""
        new_ctx = replace(self._context, **kwargs)
        child = StructuredLogger(self._logger.name, default_context=new_ctx)
        return child

    def debug(self, msg: str, **extra: Any) -> None:
        self._log(logging.DEBUG, msg, extra)

    def info(self, msg: str, **extra: Any) -> None:
        self._log(logging.INFO, msg, extra)

    def warning(self, msg: str, **extra: Any) -> None:
        self._log(logging.WARNING, msg, extra)

    def error(self, msg: str, **extra: Any) -> None:
        self._log(logging.ERROR, msg, extra)

    def critical(self, msg: str, **extra: Any) -> None:
        self._log(logging.CRITICAL, msg, extra)

    def _log(self, level: int, msg: str, extra: dict[str, Any]) -> None:
        if not self._logger.isEnabledFor(level):
            return

        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": logging.getLevelName(level),
            "msg": msg,
            "trace_id": self._context.trace_id,
            "span_id": self._context.span_id,
            "order_id": self._context.order_id,
            "symbol": self._context.symbol,
            "strategy": self._context.strategy,
        }
        record.update(extra)
        record = {k: v for k, v in record.items() if v}
        self._logger.log(level, json.dumps(record, default=str))


class JSONFormatter(logging.Formatter):
    """Logging formatter that outputs JSON lines.

    Similar to ``JsonFormatter`` but parses pre-formatted JSON messages
    produced by ``StructuredLogger`` and enriches them with logger metadata.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            payload = {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": message,
            }
        payload.setdefault("logger", record.name)
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        return json.dumps(payload, default=str)
