# event/event.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from .header import EventHeader


@dataclass(frozen=True)
class Event:
    """
    Event —— 系统中唯一合法的事件事实形态

    冻结约定：
    - header：制度信息（不可由业务侧构造）
    - payload：业务数据（schema 校验对象）
    """
    header: EventHeader
    payload: Dict[str, Any]
