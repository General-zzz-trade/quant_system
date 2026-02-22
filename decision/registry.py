from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class Registry:
    """Simple name->factory registry."""
    factories: Dict[str, Callable[[], Any]] = field(default_factory=dict)

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        self.factories[name] = factory

    def build(self, name: str) -> Any:
        if name not in self.factories:
            raise KeyError(f"Unknown component: {name}")
        return self.factories[name]()
