"""Infrastructure utilities: config, logging, run context, and metrics.

Renamed from platform/ to avoid shadowing Python's stdlib ``platform`` module.
These helpers avoid heavy dependencies and are safe to use in both backtests and live runs.
"""

from .runtime.run_context import RunContext
from .config.loader import load_config
from .logging.setup import setup_logging

__all__ = ["RunContext", "load_config", "setup_logging"]
