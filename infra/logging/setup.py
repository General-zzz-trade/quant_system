"""Logging setup for the stdlib logger: ``setup_logging()`` and ``get_logger()``.

(The effects-logger bridge went with infra/effects.py and the engine package, 2026-11-15.)
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    *,
    level: str = "INFO",
    log_file: str | None = None,
    fmt: str = "%(asctime)s %(levelname)s %(name)s %(message)s",
) -> logging.Logger:
    """Configure the quant_system root logger.

    Returns the configured logger. Safe to call multiple times (idempotent).
    """
    logger = logging.getLogger("quant_system")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_file:
            p = Path(log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                p, maxBytes=100_000_000, backupCount=7, encoding="utf-8",
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a named child logger under the quant_system namespace."""
    log = logging.getLogger(f"quant_system.{name}")
    if level is not None:
        log.setLevel(level)
    return log

