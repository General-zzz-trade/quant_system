# context/views/base.py
"""Base context view — read-only filtered projection of Context."""
from __future__ import annotations

from typing import Any, Protocol


class ContextView(Protocol):
    """
    Context 视图协议。

    不同消费者（strategy, risk, execution, monitoring）
    只能看到属于自己权限范围的数据。
    """

    def refresh(self) -> None:
        """从 Context 刷新视图数据。"""
        ...
