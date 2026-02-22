# context/registry.py
"""Context registry — manage multiple Context instances."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from context.context import Context


class ContextRegistry:
    """
    Context 注册中心。

    支持多策略/多交易所场景下的 Context 管理。
    """

    def __init__(self) -> None:
        self._contexts: Dict[str, Context] = {}

    def register(self, context_id: str, context: Context) -> None:
        self._contexts[context_id] = context

    def get(self, context_id: str) -> Optional[Context]:
        return self._contexts.get(context_id)

    def require(self, context_id: str) -> Context:
        ctx = self._contexts.get(context_id)
        if ctx is None:
            raise KeyError(f"context not found: {context_id}")
        return ctx

    @property
    def all_ids(self) -> Sequence[str]:
        return list(self._contexts.keys())

    def __len__(self) -> int:
        return len(self._contexts)
