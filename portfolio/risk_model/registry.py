# portfolio/risk_model/registry.py
"""Risk model registry."""
from __future__ import annotations

from typing import Optional

from portfolio.risk_model.base import RiskModel


class RiskModelRegistry:
    """风险模型注册表。"""

    def __init__(self) -> None:
        self._models: dict[str, RiskModel] = {}
        self._default: Optional[str] = None

    def register(self, model: RiskModel, *, default: bool = False) -> None:
        self._models[model.name] = model
        if default or self._default is None:
            self._default = model.name

    def get(self, name: str) -> RiskModel:
        if name not in self._models:
            raise KeyError(f"unknown risk model: {name}")
        return self._models[name]

    def get_default(self) -> RiskModel:
        if self._default is None:
            raise RuntimeError("no risk models registered")
        return self._models[self._default]

    def list_models(self) -> list[str]:
        return list(self._models.keys())
