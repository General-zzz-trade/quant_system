# quant_system/risk/aggregator.py
from __future__ import annotations

from dataclasses import dataclass
import threading
from time import perf_counter
import traceback
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from event.types import IntentEvent, OrderEvent
from risk.decisions import RiskAction, RiskDecision, merge_decisions


class RiskAggregatorError(RuntimeError):
    pass


class RiskRule(Protocol):
    """
    风控规则接口（机构级最小契约）
    - 规则只读：从 meta 读取事实
    - 规则只输出：RiskDecision
    """
    name: str

    def evaluate_intent(self, intent: IntentEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        ...

    def evaluate_order(self, order: OrderEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        ...


@dataclass(frozen=True, slots=True)
class RiskEvalMetaBuilder:
    """
    meta 构建器（可插拔）

    顶级机构做法：
    - Risk 层不直接 import Context/AccountState（避免环依赖）
    - 上层把“事实”打包成 meta 传入
    - meta 的构建是独立职责（可测试、可复用）
    """
    build_for_intent: Callable[[IntentEvent], Mapping[str, Any]]
    build_for_order: Callable[[OrderEvent], Mapping[str, Any]]


@dataclass(slots=True)
class RuleStats:
    """
    规则级统计（可观测）
    注意：这里刻意使用“可变对象”，确保统计更新是单点、原子语义（更不容易写出不一致）。
    """
    name: str
    calls: int = 0
    allow: int = 0
    reduce: int = 0
    reject: int = 0
    kill: int = 0
    errors: int = 0

    # 可选：延迟统计（不影响正确性）
    last_ms: float = 0.0
    max_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class AggregatorSnapshot:
    """
    聚合器快照（用于监控/诊断）
    """
    enabled: Tuple[str, ...]
    disabled: Tuple[str, ...]
    stats: Tuple[RuleStats, ...]


class RiskAggregator:
    """
    顶级机构级 RiskAggregator（可冻结版）

    目标：
    1) 以稳定契约驱动风险评估：Intent/Order -> RiskDecision
    2) 可插拔规则集 + 可热禁用规则
    3) 可观测：每条规则调用次数、动作分布、异常次数、（可选）耗时
    4) Fail-safe：规则异常时保守失败（默认 REJECT，可配置 KILL）
    5) “不吞异常”：将异常信息写入 violation.details，并允许外部注入 on_error 钩子
    """

    def __init__(
        self,
        *,
        rules: Sequence[RiskRule],
        meta_builder: RiskEvalMetaBuilder,
        fail_safe_action: RiskAction = RiskAction.REJECT,
        stop_on_kill: bool = True,
        stop_on_reject: bool = False,
        stop_on_reduce: bool = False,
        enabled: Optional[Sequence[str]] = None,
        disabled: Optional[Sequence[str]] = None,
        on_error: Optional[Callable[[str, str, BaseException, Mapping[str, Any]], None]] = None,
    ) -> None:
        if not rules:
            raise RiskAggregatorError("RiskAggregator.rules 不能为空")

        self._rules: Tuple[RiskRule, ...] = tuple(rules)
        self._meta_builder = meta_builder

        self._fail_safe_action = fail_safe_action
        self._stop_on_kill = stop_on_kill
        self._stop_on_reject = stop_on_reject
        self._stop_on_reduce = stop_on_reduce

        self._on_error = on_error

        names = [r.name for r in self._rules]
        if len(names) != len(set(names)):
            raise RiskAggregatorError("规则 name 必须唯一")

        enabled_set = set(names) if enabled is None else set(enabled)
        disabled_set = set(disabled or ())

        self._enabled: Dict[str, bool] = {n: (n in enabled_set and n not in disabled_set) for n in names}

        # 统一锁：保护 _stats 和 _enabled 的线程安全
        self._lock = threading.Lock()
        self._stats: Dict[str, RuleStats] = {n: RuleStats(name=n) for n in names}

    # ------------------------------------------------------------
    # 开关与快照
    # ------------------------------------------------------------

    def enable(self, rule_name: str) -> None:
        with self._lock:
            if rule_name not in self._enabled:
                raise RiskAggregatorError(f"未知规则：{rule_name}")
            self._enabled[rule_name] = True

    def disable(self, rule_name: str) -> None:
        with self._lock:
            if rule_name not in self._enabled:
                raise RiskAggregatorError(f"未知规则：{rule_name}")
            self._enabled[rule_name] = False

    def snapshot(self) -> AggregatorSnapshot:
        with self._lock:
            enabled = tuple(n for n, on in self._enabled.items() if on)
            disabled = tuple(n for n, on in self._enabled.items() if not on)
            stats = tuple(
                RuleStats(
                    name=s.name,
                    calls=s.calls,
                    allow=s.allow,
                    reduce=s.reduce,
                    reject=s.reject,
                    kill=s.kill,
                    errors=s.errors,
                    last_ms=s.last_ms,
                    max_ms=s.max_ms,
                )
                for s in self._stats.values()
            )
        return AggregatorSnapshot(enabled=enabled, disabled=disabled, stats=stats)

    # ------------------------------------------------------------
    # 核心评估：Intent / Order
    # ------------------------------------------------------------

    def evaluate_intent(self, intent: IntentEvent) -> RiskDecision:
        meta = dict(self._meta_builder.build_for_intent(intent))
        return self._evaluate("intent", intent, meta)

    def evaluate_order(self, order: OrderEvent) -> RiskDecision:
        meta = dict(self._meta_builder.build_for_order(order))
        return self._evaluate("order", order, meta)

    # ------------------------------------------------------------
    # 内部：统一评估循环
    # ------------------------------------------------------------

    def _evaluate(self, mode: str, obj: Any, meta: Mapping[str, Any]) -> RiskDecision:
        decisions: list[RiskDecision] = []

        # 快照 enabled 状态，避免评估过程中被并发修改
        with self._lock:
            enabled_snap = dict(self._enabled)

        for rule in self._rules:
            if not enabled_snap.get(rule.name, False):
                continue

            st = self._stats[rule.name]
            had_error = False

            t0 = perf_counter()
            try:
                if mode == "intent":
                    d = rule.evaluate_intent(obj, meta=meta)
                elif mode == "order":
                    d = rule.evaluate_order(obj, meta=meta)
                else:
                    raise RiskAggregatorError(f"未知 mode：{mode}")
            except BaseException as e:
                had_error = True
                d = self._fail_safe_decision(rule_name=rule.name, mode=mode, meta=meta, exc=e)
                if self._on_error is not None:
                    try:
                        self._on_error(rule.name, mode, e, meta)
                    except Exception:
                        pass

            dt_ms = (perf_counter() - t0) * 1000.0

            # 原子更新：单次加锁写入所有 stat 变更，避免并发 snapshot() 读到不一致状态
            with self._lock:
                st.calls += 1
                if had_error:
                    st.errors += 1
                st.last_ms = dt_ms
                if dt_ms > st.max_ms:
                    st.max_ms = dt_ms
                if d.action == RiskAction.ALLOW:
                    st.allow += 1
                elif d.action == RiskAction.REDUCE:
                    st.reduce += 1
                elif d.action == RiskAction.REJECT:
                    st.reject += 1
                elif d.action == RiskAction.KILL:
                    st.kill += 1

            decisions.append(d)

            # 可配置短路（机构级常用）
            if self._stop_on_kill and d.action == RiskAction.KILL:
                break
            if self._stop_on_reject and d.action == RiskAction.REJECT:
                break
            if self._stop_on_reduce and d.action == RiskAction.REDUCE:
                break

        return merge_decisions(tuple(decisions))

    def _fail_safe_decision(
        self,
        *,
        rule_name: str,
        mode: str,
        meta: Mapping[str, Any],
        exc: BaseException,
    ) -> RiskDecision:
        """
        当规则抛异常时的保守决策（不吞异常上下文）
        - 默认 REJECT（可配置为 KILL）
        """
        from risk.decisions import RiskViolation, RiskCode, RiskScope

        # 控制堆栈长度，避免过大（同时保证足够定位）
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if len(tb) > 8000:
            tb = tb[-8000:]

        v = RiskViolation(
            code=RiskCode.UNKNOWN,
            message=f"风险规则异常：{rule_name} mode={mode}",
            scope=RiskScope.GLOBAL,
            severity="fatal" if self._fail_safe_action == RiskAction.KILL else "error",
            details={
                "rule": rule_name,
                "mode": mode,
                "exc_type": type(exc).__name__,
                "exc": repr(exc),
                "traceback": tb,
            },
        )

        if self._fail_safe_action == RiskAction.KILL:
            return RiskDecision.kill((v,), scope=v.scope, tags=("fail_safe", rule_name))
        # REDUCE 在异常场景不安全：保守降为 REJECT
        return RiskDecision.reject((v,), scope=v.scope, tags=("fail_safe", rule_name))
