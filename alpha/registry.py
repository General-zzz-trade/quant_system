from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Optional

from .base import AlphaModel

if TYPE_CHECKING:
    from core.plugins import PluginRegistry


@dataclass
class AlphaRegistry:
    """A simple name -> alpha model registry."""

    _models: Dict[str, AlphaModel]

    def __init__(self, plugin_registry: Optional["PluginRegistry"] = None) -> None:
        self._models = {}
        self._plugin_registry = plugin_registry

    def register(self, model: AlphaModel) -> None:
        key = getattr(model, "name", model.__class__.__name__).strip()
        if not key:
            raise ValueError("alpha model must have a non-empty name")
        self._models[key] = model
        if self._plugin_registry is not None:
            self._plugin_registry.register_instance(model, name=key)

    def get(self, name: str) -> Optional[AlphaModel]:
        return self._models.get(name)

    def list_names(self) -> Iterable[str]:
        return list(self._models.keys())
