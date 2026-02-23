"""Infrastructure utilities: config, logging, run context, and metrics.

Renamed from platform/ to avoid shadowing Python's stdlib ``platform`` module.
These helpers avoid heavy dependencies and are safe to use in both backtests and live runs.

For core services (ConfigService, Effects, Clock), prefer importing from ``core/``
directly. This module provides convenience wrappers and backwards compatibility.
"""
from infra.runtime.run_context import RunContext
from infra.config.loader import load_config
from infra.logging.setup import setup_logging
from infra.metrics.registry import Metrics, create_metrics

__all__ = [
    "RunContext",
    "load_config",
    "setup_logging",
    "Metrics",
    "create_metrics",
]
