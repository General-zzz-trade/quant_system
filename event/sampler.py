# event/sampler.py
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================
# Sampling Mode
# ============================================================

class SamplingMode(str, Enum):
    ALL = "all"
    RATE = "rate"
    TIME = "time"
    COUNT = "count"
    CUSTOM = "custom"


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True, slots=True)
class SamplerConfig:
    """
    Sampler 配置（冻结）

    说明：
    - RATE：基于 event_id 的确定性采样（稳定、可复现）
    - TIME / COUNT：进程级采样（非 per-symbol）
    """
    mode: SamplingMode = SamplingMode.ALL

    # RATE
    rate: float = 1.0

    # TIME
    interval_s: float = 0.0

    # COUNT
    every_n: int = 1

    # CUSTOM
    func: Optional[Callable[[Any], bool]] = None


# ============================================================
# Sampler
# ============================================================

class EventSampler:
    """
    EventSampler —— 事件采样器（机构级）

    冻结约定：
    - 只返回 bool
    - 不抛异常
    - 不产生副作用
    """

    def __init__(self, cfg: Optional[SamplerConfig] = None) -> None:
        self._cfg = cfg or SamplerConfig()
        self._lock = threading.Lock()

        # TIME 模式
        self._last_ts: float = 0.0

        # COUNT 模式（进程级）
        self._count: int = 0

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def accept(self, event: Any) -> bool:
        mode = self._cfg.mode

        if mode == SamplingMode.ALL:
            return True

        if mode == SamplingMode.RATE:
            return self._accept_rate(event)

        if mode == SamplingMode.TIME:
            return self._accept_time()

        if mode == SamplingMode.COUNT:
            return self._accept_count()

        if mode == SamplingMode.CUSTOM:
            return self._accept_custom(event)

        # 防御式兜底：未知模式直接放行
        return True

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------

    def _accept_rate(self, event: Any) -> bool:
        """
        RATE 采样（确定性）

        使用 event_id 的 hash，避免时间相关性与并发偏差
        """
        r = self._cfg.rate

        if r >= 1.0:
            return True
        if r <= 0.0:
            return False

        # 无 header 或 event_id：直接放行（安全兜底）
        event_id = getattr(getattr(event, "header", None), "event_id", None)
        if not event_id:
            return True

        h = hashlib.sha256(event_id.encode()).digest()
        v = int.from_bytes(h[:2], "big") / 65535.0
        return v < r

    def _accept_time(self) -> bool:
        interval = self._cfg.interval_s
        if interval <= 0.0:
            return True

        now = time.monotonic()
        with self._lock:
            if now - self._last_ts >= interval:
                self._last_ts = now
                return True
            return False

    def _accept_count(self) -> bool:
        n = self._cfg.every_n
        if n <= 1:
            return True

        with self._lock:
            self._count += 1
            return (self._count % n) == 0

    def _accept_custom(self, event: Any) -> bool:
        func = self._cfg.func
        if func is None:
            return True

        try:
            return bool(func(event))
        except Exception:
            # CUSTOM 失败默认放行（制度级安全原则）
            return True
