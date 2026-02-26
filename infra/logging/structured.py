"""Structured JSON logging for production environments.

Outputs each log record as a single JSON line — easy to ingest into
Elasticsearch, CloudWatch, Loki, etc.

Usage:
    from infra.logging.structured import setup_structured_logging
    setup_structured_logging(level="INFO", log_file="app.log")
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
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
