from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .base import AlphaModel


@dataclass
class AlphaRegistry:
    """A simple name -> alpha model registry."""

    _models: Dict[str, AlphaModel]

    def __init__(self) -> None:
        self._models = {}

    def register(self, model: AlphaModel) -> None:
        key = getattr(model, "name", model.__class__.__name__).strip()
        if not key:
            raise ValueError("alpha model must have a non-empty name")
        self._models[key] = model

    def get(self, name: str) -> Optional[AlphaModel]:
        return self._models.get(name)

    def list_names(self) -> Iterable[str]:
        return list(self._models.keys())
