"""Alpha models produce trading signals from features or snapshots.

This package provides small, dependency-free building blocks.
"""

from .base import AlphaModel, Signal
from .registry import AlphaRegistry

__all__ = ["AlphaModel", "Signal", "AlphaRegistry"]
