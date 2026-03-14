# event/security.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from typing import Any, Iterable, Optional, Sequence, Set

from event.errors import EventSecurityError


# ============================================================
# 安全模型：谁(Actor) 能发什么(Event) 在什么模式(Mode)下
# ============================================================


class RunMode(str, Enum):
    """
    运行模式
    - LIVE：实盘/仿真在线
    - REPLAY：回放/回测
    """
    LIVE = "live"
    REPLAY = "replay"


@dataclass(frozen=True)
class Actor:
    """
    事件发射者（权限主体）

    module: 逻辑模块名（例如 "replay", "execution", "strategy", "risk"）
    roles:  权限角色集合（例如 {"replay"}, {"execution"}, {"system"}）
    mode:   当前运行模式
    source: 来源标签（例如 "engine", "backtest", "live_gateway"）
    """
    module: str
    roles: Set[str]
    mode: RunMode
    source: str = "engine"


@dataclass(frozen=True)
class PermissionRule:
    """
    权限规则（机构级：白名单优先）

    event_pattern:
      - 支持 fnmatch（例如 "Order*"、"Market*"、"Execution.*"）
      - 也支持精确 event_type（例如 "OrderSubmit"）

    allowed_roles:
      - 至少满足其一即可
      - 为空表示不允许任何角色（等于 deny）

    allowed_modes:
      - 允许在哪些模式下发射该类事件

    allowed_sources:
      - 可选：限制来源（例如只允许 live_gateway 发 Execution 相关事件）
    """
    event_pattern: str
    allowed_roles: Set[str]
    allowed_modes: Set[RunMode] = field(default_factory=lambda: {RunMode.LIVE, RunMode.REPLAY})
    allowed_sources: Optional[Set[str]] = None
    doc: str = ""


@dataclass(frozen=True)
class SecurityConfig:
    """
    安全配置

    deny_by_default:
      - True：未命中任何规则直接拒绝（机构默认）
      - False：未命中规则放行（仅用于快速原型，不建议）

    allow_system_bypass:
      - True：roles 包含 "system" 时可放行（只建议用于底层框架事件）
    """
    deny_by_default: bool = True
    allow_system_bypass: bool = True


# ============================================================
# EventSecurity：统一权限校验入口
# ============================================================

class EventSecurity:
    """
    EventSecurity：事件权限边界（机构级）

    使用位置（唯一正确）：
    - runtime.emit() 最前面（store 之前、publish 之前）

    设计纪律：
    - 只做校验，不做日志/I/O
    - 规则以“白名单”为主
    """

    def __init__(
        self,
        *,
        rules: Sequence[PermissionRule],
        config: Optional[SecurityConfig] = None,
    ) -> None:
        self._rules = list(rules)
        self._cfg = config or SecurityConfig()

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def check(self, *, event: Any, actor: Actor) -> None:
        """
        校验 actor 是否有权发射 event
        """
        event_type = self._event_type_of(event)

        # system bypass（只建议给框架内部事件使用）
        if self._cfg.allow_system_bypass and "system" in actor.roles:
            return

        matched = False
        for r in self._rules:
            if not fnmatch(event_type.lower(), r.event_pattern.lower()):
                continue

            matched = True

            # mode 限制
            if actor.mode not in r.allowed_modes:
                raise EventSecurityError(
                    f"事件权限拒绝：mode 不允许 "
                    f"(event_type={event_type}, actor_mode={actor.mode}, rule_modes={sorted(m.value for m in r.allowed_modes)})",
                    event_type=event_type,
                )

            # source 限制（可选）
            if r.allowed_sources is not None and actor.source not in r.allowed_sources:
                raise EventSecurityError(
                    f"事件权限拒绝：source 不允许 "
                    f"(event_type={event_type}, actor_source={actor.source}, rule_sources={sorted(r.allowed_sources)})",
                    event_type=event_type,
                )

            # role 限制（必须命中至少一个）
            if not (actor.roles & r.allowed_roles):
                raise EventSecurityError(
                    f"事件权限拒绝：role 不匹配 "
                    f"(event_type={event_type}, actor_roles={sorted(actor.roles)}, rule_roles={sorted(r.allowed_roles)})",
                    event_type=event_type,
                )

            # 命中且通过：直接放行
            return

        # 未命中任何规则：默认拒绝（机构默认）
        if self._cfg.deny_by_default and not matched:
            raise EventSecurityError(
                f"事件权限拒绝：未命中任何白名单规则 (event_type={event_type}, actor_module={actor.module})",
                event_type=event_type,
            )

        # 原型模式：未命中规则放行
        return

    # --------------------------------------------------------
    # 工具
    # --------------------------------------------------------

    @staticmethod
    def _event_type_of(event: Any) -> str:
        """
        获取事件类型（协议化）
        优先级：
        1) event.EVENT_TYPE（推荐）
        2) event.header.event_type（如果你在 header 里存了）
        3) event.__class__.__name__
        """
        et = getattr(event, "EVENT_TYPE", None)
        if isinstance(et, str) and et:
            return et

        et2 = getattr(event, "event_type", None)
        if isinstance(et2, str) and et2:
            return et2
        if hasattr(et2, "value"):
            val = getattr(et2, "value")
            if isinstance(val, str) and val:
                return val

        header = getattr(event, "header", None)
        if header is not None:
            ht = getattr(header, "event_type", None)
            if isinstance(ht, str) and ht:
                return ht

        return event.__class__.__name__


# ============================================================
# 默认规则模板（可直接用，也可自行替换）
# ============================================================

def default_security_rules() -> Sequence[PermissionRule]:
    """
    默认安全规则（机构级保守版本）

    约定角色：
    - system：框架内部（可 bypass）
    - replay：回放/回测事件源
    - market：行情/数据源
    - strategy：策略层（只能发信号类事件）
    - risk：风控层（可发风控/限制类事件）
    - execution：执行网关（订单/成交相关）
    """
    return [
        # 行情类（回放与实盘都可，通常来自 market/replay）
        PermissionRule(
            event_pattern="Market*",
            allowed_roles={"market", "replay"},
            allowed_modes={RunMode.LIVE, RunMode.REPLAY},
            doc="行情/数据事件",
        ),

        # 策略信号（允许 replay / live，但只允许 strategy/risk 发）
        PermissionRule(
            event_pattern="Signal*",
            allowed_roles={"strategy", "risk"},
            allowed_modes={RunMode.LIVE, RunMode.REPLAY},
            doc="策略信号事件（不应直接等同于下单）",
        ),

        # 下单/撤单（live：只允许 execution；replay：可允许 replay 复盘订单流）
        PermissionRule(
            event_pattern="Order*",
            allowed_roles={"execution", "replay"},
            allowed_modes={RunMode.LIVE, RunMode.REPLAY},
            doc="订单类事件（实盘必须来自执行网关）",
        ),

        # 成交/回报（live：只允许 execution；replay：允许 replay 复盘成交流）
        PermissionRule(
            event_pattern="Fill*",
            allowed_roles={"execution", "replay"},
            allowed_modes={RunMode.LIVE, RunMode.REPLAY},
            doc="成交/回报类事件",
        ),

        # 风控控制类（例如 KillSwitch、RiskLimit）
        PermissionRule(
            event_pattern="Risk*",
            allowed_roles={"risk"},
            allowed_modes={RunMode.LIVE, RunMode.REPLAY},
            doc="风控控制事件",
        ),

        # 系统内部（例如 Heartbeat、CheckpointSaved 等框架事件，建议让 system 角色发）
        PermissionRule(
            event_pattern="Sys*",
            allowed_roles={"system"},
            allowed_modes={RunMode.LIVE, RunMode.REPLAY},
            doc="框架内部事件",
        ),
    ]


# ============================================================
# 快速构造 Actor（推荐在 runtime / engine 里统一注入）
# ============================================================

def make_actor(
    *,
    module: str,
    roles: Iterable[str],
    mode: RunMode,
    source: str = "engine",
) -> Actor:
    return Actor(module=module, roles=set(roles), mode=mode, source=source)
