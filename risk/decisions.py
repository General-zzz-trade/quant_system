# quant_system/risk/decisions.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


class RiskAction(str, Enum):
    """
    风控动作（顶级机构标准：动作与“理由”解耦）

    - ALLOW : 允许继续（可能附带降级建议）
    - REJECT: 拒绝本次意图/订单（不影响系统继续运行）
    - REDUCE: 允许但要求降级（例如缩量、改成只减仓、切换被动单）
    - KILL  : 熔断（symbol/strategy/global），需要 kill_switch 执行
    """
    ALLOW = "allow"
    REJECT = "reject"
    REDUCE = "reduce"
    KILL = "kill"


class RiskScope(str, Enum):
    """
    决策作用域（用于 kill-switch / 降级）
    """
    SYMBOL = "symbol"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"
    ACCOUNT = "account"
    GLOBAL = "global"


class RiskCode(str, Enum):
    """
    风控原因码（必须稳定：可用于统计/告警/回测归因）
    """
    UNKNOWN = "unknown"

    # 账户级
    INSUFFICIENT_MARGIN = "insufficient_margin"
    MAX_LEVERAGE = "max_leverage"
    MAX_DRAWDOWN = "max_drawdown"
    LIQUIDATION_RISK = "liquidation_risk"

    # 仓位/敞口级
    MAX_POSITION = "max_position"
    MAX_NOTIONAL = "max_notional"
    MAX_DELTA = "max_delta"
    MAX_GROSS = "max_gross"
    MAX_NET = "max_net"

    # 市场状态级（与 regime 可联动）
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_DRY = "liquidity_dry"
    PRICE_GAP = "price_gap"

    # 系统/运维级
    STALE_DATA = "stale_data"
    EXCHANGE_DOWN = "exchange_down"
    OMS_DEGRADED = "oms_degraded"


@dataclass(frozen=True, slots=True)
class RiskViolation:
    """
    单条风控违规记录（可审计、可落库）

    说明：
    - severity 用来表述严重性，不直接等价于 action（由 aggregator 决定）
    - details 必须是可 JSON 序列化的基础类型
    """
    code: RiskCode
    message: str
    scope: RiskScope = RiskScope.GLOBAL

    symbol: Optional[str] = None          # 用 normalized symbol 字符串，避免跨模块依赖
    strategy_id: Optional[str] = None

    severity: str = "error"               # "warn" | "error" | "fatal"
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RiskAdjustment:
    """
    风控建议/强制调整（REDUCE 时使用）

    典型用法：
    - max_qty: 将订单数量上限强制压到某个值
    - reduce_only: 强制只减仓
    - post_only: 强制只挂被动单
    - tif_override: 强制改 TIF（例如 IOC -> GTC）
    """
    max_qty: Optional[float] = None
    reduce_only: Optional[bool] = None
    post_only: Optional[bool] = None
    tif_override: Optional[str] = None
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """
    风控聚合决策（顶级机构标准输出）

    重要原则：
    1) 决策必须是“可解释”的（violations）
    2) 决策必须是“可执行”的（action + scope + adjustments）
    3) 决策必须是“可观测”的（tags / meta / trace）
    """
    action: RiskAction
    scope: RiskScope = RiskScope.GLOBAL

    # 若 action=REJECT/REDUCE/KILL，必须包含至少一条 violation
    violations: Tuple[RiskViolation, ...] = ()

    # 对订单/意图的强制调整（REDUCE 常用）
    adjustment: Optional[RiskAdjustment] = None

    # 额外标记（用于监控、告警、报表）
    tags: Tuple[str, ...] = ()

    # 可选元信息（例如: risk_engine_version, model_id）
    meta: Mapping[str, Any] = field(default_factory=dict)

    def require_ok(self) -> None:
        if self.action != RiskAction.ALLOW:
            raise RuntimeError(f"RiskDecision 非 ALLOW：{self.action} violations={len(self.violations)}")

    @property
    def ok(self) -> bool:
        return self.action == RiskAction.ALLOW

    @staticmethod
    def allow(*, tags: Tuple[str, ...] = (), meta: Optional[Mapping[str, Any]] = None) -> "RiskDecision":
        return RiskDecision(action=RiskAction.ALLOW, tags=tags, meta=dict(meta or {}))

    @staticmethod
    def reject(
        violations: Tuple[RiskViolation, ...],
        *,
        scope: RiskScope = RiskScope.GLOBAL,
        tags: Tuple[str, ...] = (),
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "RiskDecision":
        if not violations:
            raise ValueError("REJECT 必须包含 violations")
        return RiskDecision(
            action=RiskAction.REJECT,
            scope=scope,
            violations=violations,
            tags=tags,
            meta=dict(meta or {}),
        )

    @staticmethod
    def reduce(
        violations: Tuple[RiskViolation, ...],
        *,
        adjustment: RiskAdjustment,
        scope: RiskScope = RiskScope.GLOBAL,
        tags: Tuple[str, ...] = (),
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "RiskDecision":
        if not violations:
            raise ValueError("REDUCE 必须包含 violations")
        return RiskDecision(
            action=RiskAction.REDUCE,
            scope=scope,
            violations=violations,
            adjustment=adjustment,
            tags=tags,
            meta=dict(meta or {}),
        )

    @staticmethod
    def kill(
        violations: Tuple[RiskViolation, ...],
        *,
        scope: RiskScope = RiskScope.GLOBAL,
        tags: Tuple[str, ...] = (),
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "RiskDecision":
        if not violations:
            raise ValueError("KILL 必须包含 violations")
        return RiskDecision(
            action=RiskAction.KILL,
            scope=scope,
            violations=violations,
            tags=tags,
            meta=dict(meta or {}),
        )


def merge_decisions(decisions: Tuple[RiskDecision, ...]) -> RiskDecision:
    """
    将多个规则的决策合并为一个最终决策（用于 aggregator）

    合并优先级（从高到低）：
    KILL > REJECT > REDUCE > ALLOW

    规则：
    - scope 取最高优先级决策的 scope（若同优先级可提升到更广 scope）
    - violations 合并
    - adjustment 仅保留最高优先级里最近一条（更复杂的 merge 在 aggregator 做）
    """
    if not decisions:
        return RiskDecision.allow()

    priority = {
        RiskAction.ALLOW: 0,
        RiskAction.REDUCE: 1,
        RiskAction.REJECT: 2,
        RiskAction.KILL: 3,
    }

    best = max(decisions, key=lambda d: priority[d.action])
    all_violations: list[RiskViolation] = []
    all_tags: list[str] = []
    meta: dict[str, Any] = {}

    for d in decisions:
        all_violations.extend(d.violations)
        all_tags.extend(d.tags)
        meta.update(d.meta)

    # 如果所有都是 ALLOW，直接允许
    if priority[best.action] == 0:
        return RiskDecision.allow(tags=tuple(all_tags), meta=meta)

    return RiskDecision(
        action=best.action,
        scope=best.scope,
        violations=tuple(all_violations),
        adjustment=best.adjustment,
        tags=tuple(all_tags),
        meta=meta,
    )
