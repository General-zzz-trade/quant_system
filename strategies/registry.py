"""Strategy registry -- discover and instantiate strategies."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from strategies.base import StrategyProtocol

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Central registry for all available strategies.

    Strategies are registered by name and can be looked up or listed.
    Duplicate registrations for the same name raise ``ValueError``.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, StrategyProtocol] = {}

    def register(self, strategy: StrategyProtocol) -> None:
        """Register a strategy instance. Uses ``strategy.name`` as the key."""
        name = strategy.name
        if name in self._strategies:
            raise ValueError(f"Strategy '{name}' is already registered")
        self._strategies[name] = strategy
        logger.info("Registered strategy: %s (v%s, %s, %s)",
                     name, strategy.version, strategy.venue, strategy.timeframe)

    def get_strategy(self, name: str) -> Optional[StrategyProtocol]:
        """Return the strategy with the given name, or ``None``."""
        return self._strategies.get(name)

    def list_strategies(self) -> List[Dict[str, Any]]:
        """Return a list of dicts describing each registered strategy."""
        return [
            {
                "name": s.name,
                "version": s.version,
                "venue": s.venue,
                "timeframe": s.timeframe,
                "description": s.describe(),
            }
            for s in self._strategies.values()
        ]

    def unregister(self, name: str) -> bool:
        """Remove a strategy by name. Returns True if it existed."""
        return self._strategies.pop(name, None) is not None

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, name: str) -> bool:
        return name in self._strategies
