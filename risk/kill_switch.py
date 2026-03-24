# quant_system/risk/kill_switch.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from time import time
from typing import Any, Dict, Mapping, Optional, Tuple


# ============================================================
# 顶级机构级 Kill Switch（熔断/停机开关）
# ============================================================
# 设计目标：
# 1) KillSwitch 是"执行级闸门"（gate），不是风控规则本身
#    - risk/aggregator 产出 RiskDecision(action=KILL, scope=...)
#    - kill_switch 负责把 KILL 变成"系统行为"：阻断下单/只允许减仓/全局停机
#
# 2) KillSwitch 必须支持多作用域：
#    - GLOBAL：全系统停机
#    - ACCOUNT/PORTFOLIO：账户级停机
#    - STRATEGY：某策略停机
#    - SYMBOL：某标的停机
#
# 3) KillSwitch 必须可审计：
#    - 谁触发（source）
#    - 为什么（reason / tags / meta）
#    - 何时触发（ts）
#    - 是否自动恢复（ttl）
#
# 4) KillSwitch 必须支持"降级模式"：
#    - HARD_KILL：完全禁止任何新单（可选：仅允许 reduce-only）
#    - REDUCE_ONLY：只允许减仓单通过
#
# 5) KillSwitch 不依赖 Context/Execution 的具体实现
#    - 输入以字符串/基础结构为主，避免环依赖
# ============================================================


class KillMode(str, Enum):
    """
    熔断模式
    """
    HARD_KILL = "hard_kill"          # 全阻断（默认）
    REDUCE_ONLY = "reduce_only"      # 只允许减仓


class KillScope(str, Enum):
    """
    熔断作用域
    """
    SYMBOL = "symbol"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"
    ACCOUNT = "account"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class KillRecord:
    """
    单条熔断记录（可落库）
    """
    scope: KillScope
    key: str                      # scope 对应的 key；GLOBAL 用 "*"
    mode: KillMode

    triggered_at: float           # unix timestamp（秒）
    ttl_seconds: Optional[int] = None

    source: str = "risk"          # 触发来源（risk/ops/manual）
    reason: str = ""
    tags: Tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def expires_at(self) -> Optional[float]:
        if self.ttl_seconds is None:
            return None
        return self.triggered_at + float(self.ttl_seconds)

    def is_expired(self, now_ts: Optional[float] = None) -> bool:
        exp = self.expires_at
        if exp is None:
            return False
        now = time() if now_ts is None else now_ts
        return now >= exp


class KillSwitchError(RuntimeError):
    pass


class KillSwitch:
    """
    Kill switch with Rust-delegated core, Python API wrapper.

    State management delegated to RustKillSwitch for lock-free performance.
    Python layer maintains KillRecord metadata (tags, meta, source) for auditability.
    """

    def __init__(self) -> None:
        try:
            from _quant_hotpath import RustKillSwitch
            self._rust = RustKillSwitch()
        except ImportError:
            self._rust = None
        self._lock = RLock()
        # (scope, key) -> KillRecord  — metadata overlay for audit trail
        self._records: Dict[Tuple[KillScope, str], KillRecord] = {}

    # ------------------------------------------------------------
    # 触发与解除
    # ------------------------------------------------------------

    def trigger(
        self,
        *,
        scope: KillScope,
        key: str,
        mode: KillMode = KillMode.HARD_KILL,
        reason: str = "",
        ttl_seconds: Optional[int] = None,
        source: str = "risk",
        tags: Tuple[str, ...] = (),
        meta: Optional[Mapping[str, Any]] = None,
        now_ts: Optional[float] = None,
    ) -> KillRecord:
        """触发熔断（覆盖写）"""
        if scope == KillScope.GLOBAL:
            key = "*"
        if not key:
            raise KillSwitchError("key 不能为空")

        now = time() if now_ts is None else float(now_ts)

        rec = KillRecord(
            scope=scope, key=key, mode=mode, triggered_at=now,
            ttl_seconds=ttl_seconds, source=source, reason=reason,
            tags=tuple(tags), meta=dict(meta or {}),
        )

        with self._lock:
            self._records[(scope, key)] = rec
            if self._rust is not None:
                self._rust.arm(
                    scope.value, key, mode.value, reason,
                    ttl_seconds=ttl_seconds, source=source, now_ts=now,
                )
        return rec

    def clear(self, *, scope: KillScope, key: str) -> bool:
        """手动解除熔断"""
        if scope == KillScope.GLOBAL:
            key = "*"
        with self._lock:
            removed = self._records.pop((scope, key), None) is not None
            if self._rust is not None:
                self._rust.disarm(scope.value, key)
            return removed

    def clear_all(self) -> None:
        """清空所有熔断"""
        with self._lock:
            self._records.clear()
            if self._rust is not None:
                self._rust.clear_all()

    # ------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------

    def _get_active(self, now_ts: Optional[float] = None) -> Dict[Tuple[KillScope, str], KillRecord]:
        """返回所有未过期熔断，并清理已过期项"""
        now = time() if now_ts is None else float(now_ts)
        with self._lock:
            expired = [k for k, rec in self._records.items() if rec.is_expired(now)]
            for k in expired:
                self._records.pop(k, None)
                if self._rust is not None:
                    self._rust.disarm(k[0].value, k[1])
            return dict(self._records)

    def active_records(self, *, now_ts: Optional[float] = None) -> Tuple[KillRecord, ...]:
        active = self._get_active(now_ts=now_ts)
        return tuple(sorted(active.values(), key=lambda r: (r.scope.value, r.key, r.triggered_at)))

    def is_killed(
        self,
        *,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        now_ts: Optional[float] = None,
    ) -> Optional[KillRecord]:
        """检查是否命中熔断 (GLOBAL > STRATEGY > SYMBOL)"""
        # Fast path: if Rust says nothing armed, skip Python dict scan
        if self._rust is not None and not self._rust.is_armed(now_ts=now_ts):
            return None

        active = self._get_active(now_ts=now_ts)

        rec = active.get((KillScope.GLOBAL, "*"))
        if rec is not None:
            return rec
        if strategy_id:
            rec = active.get((KillScope.STRATEGY, strategy_id))
            if rec is not None:
                return rec
        if symbol:
            rec = active.get((KillScope.SYMBOL, symbol))
            if rec is not None:
                return rec
        return None

    # ------------------------------------------------------------
    # 闸门接口
    # ------------------------------------------------------------

    def allow_order(
        self,
        *,
        symbol: str,
        strategy_id: Optional[str],
        reduce_only: bool,
        now_ts: Optional[float] = None,
    ) -> Tuple[bool, Optional[KillRecord]]:
        """执行前闸门判断"""
        rec = self.is_killed(symbol=symbol, strategy_id=strategy_id, now_ts=now_ts)
        if rec is None:
            return True, None
        if rec.mode == KillMode.HARD_KILL:
            return False, rec
        if rec.mode == KillMode.REDUCE_ONLY:
            return (True, rec) if reduce_only else (False, rec)
        return False, rec
