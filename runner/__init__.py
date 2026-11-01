"""Backtest runner and CLI."""
from importlib import import_module


def __getattr__(name: str):
    # Reading a risk snapshot and running offline checks must not load Rust,
    # market data clients, or the complete backtest engine.
    if name == "run_backtest":
        return getattr(import_module("runner.backtest_runner"), name)
    if name in {"OperatorControlPlane", "OperatorControlRequest", "OperatorControlResult"}:
        return getattr(import_module("runner.control_plane"), name)
    raise AttributeError(name)

__all__ = [
    "run_backtest",
    "OperatorControlPlane",
    "OperatorControlRequest",
    "OperatorControlResult",
]
