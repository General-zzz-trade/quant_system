"""Trading strategies module.

Provides a unified StrategyProtocol interface, a registry for strategy
discovery, and concrete strategy implementations.
"""
from strategies.base import Signal, StrategyProtocol
from strategies.registry import StrategyRegistry

__all__ = [
    "Signal",
    "StrategyProtocol",
    "StrategyRegistry",
]
