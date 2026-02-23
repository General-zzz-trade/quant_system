"""Component registry — name-based lookup for strategies, signals, and allocators.

Provides type-safe registration and lookup with introspection support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class Registry:
    """Name -> factory registry with category support.

    Usage::

        reg = Registry()
        reg.register("ma_cross", lambda: MACrossSignal(), category="signal")
        sig = reg.build("ma_cross")
        names = reg.list_names(category="signal")
    """
    factories: Dict[str, Callable[[], Any]] = field(default_factory=dict)
    _categories: Dict[str, str] = field(default_factory=dict)

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        category: str = "default",
        overwrite: bool = False,
    ) -> None:
        """Register a factory under ``name``.

        Raises ``KeyError`` if name already registered and overwrite is False.
        """
        if name in self.factories and not overwrite:
            raise KeyError(f"Already registered: {name}")
        self.factories[name] = factory
        self._categories[name] = category

    def build(self, name: str) -> Any:
        """Build an instance by name."""
        if name not in self.factories:
            available = ", ".join(sorted(self.factories))
            raise KeyError(f"Unknown component: {name}. Available: {available}")
        return self.factories[name]()

    def get(self, name: str) -> Optional[Callable[[], Any]]:
        """Get factory without building, or None if not found."""
        return self.factories.get(name)

    def list_names(self, *, category: Optional[str] = None) -> List[str]:
        """List registered names, optionally filtered by category."""
        if category is None:
            return sorted(self.factories)
        return sorted(n for n, c in self._categories.items() if c == category)

    def has(self, name: str) -> bool:
        return name in self.factories

    def __len__(self) -> int:
        return len(self.factories)


# ── Global registries (optional convenience) ─────────────

signal_registry = Registry()
allocator_registry = Registry()
sizer_registry = Registry()
