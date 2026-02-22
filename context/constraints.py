# quant_system/context/constraints.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from event.types import (
    Symbol,
    Venue,
    Side,
    OrderType,
    TimeInForce,
    Qty,
    Price,
    Money,
    IntentEvent,
    OrderEvent,
)


# ============================================================
# 一、约束层的定位（顶级机构标准）
# ============================================================
# constraints.py 只定义“硬规则/软规则/交易所规则/组合约束”等约束接口与评估框架。
# 它不做风控决策（ALLOW/REJECT/KILL），不负责下单，不做策略。
#
# 典型用途：
# - Intent / Order 进入 execution 前先过约束层（规则、合规、交易所限制）
# - 回测与实盘共用同一套约束评估（不可作弊）
#
# 风控（risk）与约束（constraints）的分工：
# - constraints：规则正确性与可执行性（tick size、min qty、reduce-only、tif 等）
# - risk：资金/敞口/回撤/组合风险的一票否决与降级（ALLOW/REJECT/REDUCE/KILL）


# ============================================================
# 二、约束评估结果（可审计）
# ============================================================

class Severity(str, Enum):
    """
    约束严重性：
    - ERROR：不可执行/必须拒绝（例如价格精度不对）
    - WARN ：可执行但不推荐（例如滑点预估过高、撮合环境不佳）
    """
    ERROR = "error"
    WARN = "warn"


class ConstraintCode(str, Enum):
    # 通用
    UNKNOWN = "unknown"
    MISSING_FIELD = "missing_field"
    INVALID_VALUE = "invalid_value"

    # 交易所/合约规则
    TICK_SIZE = "tick_size"
    STEP_SIZE = "step_size"
    MIN_QTY = "min_qty"
    MIN_NOTIONAL = "min_notional"
    MAX_QTY = "max_qty"
    PRICE_BANDS = "price_bands"
    TIME_IN_FORCE = "time_in_force"
    ORDER_TYPE = "order_type"
    POST_ONLY = "post_only"
    REDUCE_ONLY = "reduce_only"

    # 组合/策略级约束（这里只放“规则”，不放风控）
    SYMBOL_DISABLED = "symbol_disabled"
    VENUE_DISABLED = "venue_disabled"
    SIDE_DISABLED = "side_disabled"


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    """
    单条违反记录（必须可审计、可落库）
    """
    code: ConstraintCode
    severity: Severity
    message: str
    # 归属范围：symbol/venue/portfolio/global 等
    scope: str = "global"
    symbol: Optional[Symbol] = None
    venue: Optional[Venue] = None
    # 附加调试信息（比如期望 step、实际 qty）
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """
    一次评估的聚合结果
    """
    violations: Tuple[ConstraintViolation, ...] = ()
    tags: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return all(v.severity != Severity.ERROR for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        return any(v.severity == Severity.WARN for v in self.violations)

    def errors(self) -> Tuple[ConstraintViolation, ...]:
        return tuple(v for v in self.violations if v.severity == Severity.ERROR)

    def warnings(self) -> Tuple[ConstraintViolation, ...]:
        return tuple(v for v in self.violations if v.severity == Severity.WARN)

    @staticmethod
    def combine(results: Sequence["ConstraintResult"]) -> "ConstraintResult":
        violations: list[ConstraintViolation] = []
        tags: list[str] = []
        for r in results:
            violations.extend(r.violations)
            tags.extend(r.tags)
        return ConstraintResult(violations=tuple(violations), tags=tuple(tags))


# ============================================================
# 三、评估上下文（不依赖具体 Context 实现，避免耦合）
# ============================================================

@dataclass(frozen=True, slots=True)
class ConstraintContext:
    """
    约束评估上下文（只读）

    说明：
    - 这里不直接 import 你的 Context 类，避免循环依赖与耦合。
    - 需要的东西（如 market_price、bar_index、账户字段）由上层组装传入。
    """
    ts: Any
    bar_index: int
    market_price: Optional[Decimal] = None   # 最新价（可用于 min_notional 或 price_bands）
    meta: Mapping[str, Any] = field(default_factory=dict)


# ============================================================
# 四、交易所规则：规格（tick/step/min/max 等）
# ============================================================

@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """
    单一标的在某 venue 的交易规格（由 data/normalization 或 exchange metadata 提供）
    """
    venue: Venue
    symbol: Symbol

    # 数量与价格精度规则（典型：tickSize / stepSize）
    tick_size: Optional[Decimal] = None
    step_size: Optional[Decimal] = None

    # 下单最小/最大限制
    min_qty: Optional[Decimal] = None
    max_qty: Optional[Decimal] = None

    # 最小名义金额（例如现货/合约都有）
    min_notional: Optional[Decimal] = None

    # 允许的订单类型与 TIF（可为空表示不限制）
    allowed_order_types: Tuple[OrderType, ...] = ()
    allowed_tif: Tuple[TimeInForce, ...] = ()

    # 价格带限制（例如涨跌停、banding；可选）
    # 这里用相对带宽（±band_pct），也可由上层替换成绝对带宽规则
    price_band_pct: Optional[Decimal] = None


# ============================================================
# 五、约束接口与集合
# ============================================================

class Constraint(Protocol):
    """
    约束接口：输入 Intent 或 Order，输出约束结果。
    约束应是纯函数：同样输入→同样输出（可测试、可回放）。
    """
    name: str

    def evaluate_intent(
        self,
        intent: IntentEvent,
        *,
        ctx: ConstraintContext,
        spec: Optional[InstrumentSpec] = None,
    ) -> ConstraintResult:
        ...

    def evaluate_order(
        self,
        order: OrderEvent,
        *,
        ctx: ConstraintContext,
        spec: Optional[InstrumentSpec] = None,
    ) -> ConstraintResult:
        ...


@dataclass(frozen=True, slots=True)
class ConstraintSet:
    """
    约束集合（机构级：可组合、可插拔）
    """
    constraints: Tuple[Constraint, ...] = ()

    def evaluate_intent(
        self,
        intent: IntentEvent,
        *,
        ctx: ConstraintContext,
        spec: Optional[InstrumentSpec] = None,
    ) -> ConstraintResult:
        results = [c.evaluate_intent(intent, ctx=ctx, spec=spec) for c in self.constraints]
        return ConstraintResult.combine(results)

    def evaluate_order(
        self,
        order: OrderEvent,
        *,
        ctx: ConstraintContext,
        spec: Optional[InstrumentSpec] = None,
    ) -> ConstraintResult:
        results = [c.evaluate_order(order, ctx=ctx, spec=spec) for c in self.constraints]
        return ConstraintResult.combine(results)


# ============================================================
# 六、基础约束实现（顶级机构常用的最小集）
# ============================================================

def _is_multiple_of(x: Decimal, step: Decimal) -> bool:
    """
    判断 x 是否为 step 的整数倍（避免 float；对 Decimal 友好）
    """
    if step == 0:
        return True
    # x / step 必须是整数
    q = (x / step)
    return q == q.to_integral_value()


def _d(v: Decimal | int | float | str) -> Decimal:
    return Decimal(str(v))


@dataclass(frozen=True, slots=True)
class BasicFieldsConstraint:
    """
    基础字段完整性约束：
    - Intent：target_qty 或 target_position_notional 至少有一个（取决于你的上层约定）
    - Order：qty 必须存在，limit/stop 规则必须匹配 order_type
    """
    name: str = "basic_fields"

    def evaluate_intent(self, intent: IntentEvent, *, ctx: ConstraintContext, spec: Optional[InstrumentSpec] = None) -> ConstraintResult:
        violations: list[ConstraintViolation] = []

        if intent.target_qty is None and intent.target_position_notional is None:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.MISSING_FIELD,
                    severity=Severity.ERROR,
                    message="Intent 缺少 target_qty 或 target_position_notional（至少需要一个）",
                    scope="symbol",
                    symbol=intent.symbol,
                    details={},
                )
            )

        if intent.urgency not in ("low", "normal", "high"):
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.INVALID_VALUE,
                    severity=Severity.ERROR,
                    message=f"Intent.urgency 非法：{intent.urgency}",
                    scope="global",
                    details={"urgency": intent.urgency},
                )
            )

        return ConstraintResult(violations=tuple(violations))

    def evaluate_order(self, order: OrderEvent, *, ctx: ConstraintContext, spec: Optional[InstrumentSpec] = None) -> ConstraintResult:
        violations: list[ConstraintViolation] = []

        if order.qty is None:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.MISSING_FIELD,
                    severity=Severity.ERROR,
                    message="Order 缺少 qty",
                    scope="symbol",
                    symbol=order.symbol,
                    venue=order.venue,
                )
            )
            return ConstraintResult(violations=tuple(violations))

        # limit/stop 字段与订单类型匹配检查（最小但关键）
        if order.order_type in (OrderType.LIMIT, OrderType.POST_ONLY):
            if order.limit_price is None:
                violations.append(
                    ConstraintViolation(
                        code=ConstraintCode.MISSING_FIELD,
                        severity=Severity.ERROR,
                        message="LIMIT/POST_ONLY 订单必须提供 limit_price",
                        scope="symbol",
                        symbol=order.symbol,
                        venue=order.venue,
                    )
                )

        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if order.stop_price is None:
                violations.append(
                    ConstraintViolation(
                        code=ConstraintCode.MISSING_FIELD,
                        severity=Severity.ERROR,
                        message="STOP/STOP_LIMIT 订单必须提供 stop_price",
                        scope="symbol",
                        symbol=order.symbol,
                        venue=order.venue,
                    )
                )
            if order.order_type == OrderType.STOP_LIMIT and order.limit_price is None:
                violations.append(
                    ConstraintViolation(
                        code=ConstraintCode.MISSING_FIELD,
                        severity=Severity.ERROR,
                        message="STOP_LIMIT 订单必须同时提供 stop_price 与 limit_price",
                        scope="symbol",
                        symbol=order.symbol,
                        venue=order.venue,
                    )
                )

        return ConstraintResult(violations=tuple(violations))


@dataclass(frozen=True, slots=True)
class VenueSpecConstraint:
    """
    交易所规格约束（tick/step/min/max/min_notional/允许的订单类型与 TIF）
    """
    name: str = "venue_spec"

    def evaluate_intent(self, intent: IntentEvent, *, ctx: ConstraintContext, spec: Optional[InstrumentSpec] = None) -> ConstraintResult:
        # Intent 层一般不强卡 tick/step（因为还没转成 order），但可以做 min_notional 预检查
        violations: list[ConstraintViolation] = []
        if spec is None:
            return ConstraintResult()

        # min_notional 预检查（需要 market_price 或 limit_price）
        if spec.min_notional is not None:
            notional: Optional[Decimal] = None

            if intent.target_position_notional is not None:
                notional = intent.target_position_notional.amount

            elif intent.target_qty is not None:
                px = None
                if intent.limit_price is not None:
                    px = intent.limit_price.value
                elif ctx.market_price is not None:
                    px = ctx.market_price
                if px is not None:
                    notional = intent.target_qty.value * px

            if notional is not None and notional < spec.min_notional:
                violations.append(
                    ConstraintViolation(
                        code=ConstraintCode.MIN_NOTIONAL,
                        severity=Severity.ERROR,
                        message="Intent 名义金额低于最小下单名义金额",
                        scope="symbol",
                        symbol=intent.symbol,
                        venue=spec.venue,
                        details={"notional": str(notional), "min_notional": str(spec.min_notional)},
                    )
                )

        return ConstraintResult(violations=tuple(violations))

    def evaluate_order(self, order: OrderEvent, *, ctx: ConstraintContext, spec: Optional[InstrumentSpec] = None) -> ConstraintResult:
        violations: list[ConstraintViolation] = []
        if spec is None:
            return ConstraintResult()

        # allowed order types
        if spec.allowed_order_types and order.order_type not in spec.allowed_order_types:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.ORDER_TYPE,
                    severity=Severity.ERROR,
                    message="该 venue/spec 不允许此订单类型",
                    scope="symbol",
                    symbol=order.symbol,
                    venue=order.venue,
                    details={"order_type": order.order_type.value, "allowed": [t.value for t in spec.allowed_order_types]},
                )
            )

        # allowed tif
        if spec.allowed_tif and order.tif not in spec.allowed_tif:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.TIME_IN_FORCE,
                    severity=Severity.ERROR,
                    message="该 venue/spec 不允许此 TimeInForce",
                    scope="symbol",
                    symbol=order.symbol,
                    venue=order.venue,
                    details={"tif": order.tif.value, "allowed": [t.value for t in spec.allowed_tif]},
                )
            )

        # qty rules
        q = order.qty.value
        if spec.min_qty is not None and q < spec.min_qty:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.MIN_QTY,
                    severity=Severity.ERROR,
                    message="下单数量低于最小数量",
                    scope="symbol",
                    symbol=order.symbol,
                    venue=order.venue,
                    details={"qty": str(q), "min_qty": str(spec.min_qty)},
                )
            )
        if spec.max_qty is not None and q > spec.max_qty:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.MAX_QTY,
                    severity=Severity.ERROR,
                    message="下单数量超过最大数量",
                    scope="symbol",
                    symbol=order.symbol,
                    venue=order.venue,
                    details={"qty": str(q), "max_qty": str(spec.max_qty)},
                )
            )
        if spec.step_size is not None and not _is_multiple_of(q, spec.step_size):
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.STEP_SIZE,
                    severity=Severity.ERROR,
                    message="下单数量不符合 step_size（数量步进）",
                    scope="symbol",
                    symbol=order.symbol,
                    venue=order.venue,
                    details={"qty": str(q), "step_size": str(spec.step_size)},
                )
            )

        # price rules (only when relevant)
        px: Optional[Decimal] = order.limit_price.value if order.limit_price is not None else None
        if px is not None:
            if spec.tick_size is not None and not _is_multiple_of(px, spec.tick_size):
                violations.append(
                    ConstraintViolation(
                        code=ConstraintCode.TICK_SIZE,
                        severity=Severity.ERROR,
                        message="限价不符合 tick_size（价格最小变动）",
                        scope="symbol",
                        symbol=order.symbol,
                        venue=order.venue,
                        details={"price": str(px), "tick_size": str(spec.tick_size)},
                    )
                )

            # price band check (optional)
            if spec.price_band_pct is not None and ctx.market_price is not None:
                mp = ctx.market_price
                band = spec.price_band_pct
                lo = mp * (Decimal("1") - band)
                hi = mp * (Decimal("1") + band)
                if px < lo or px > hi:
                    violations.append(
                        ConstraintViolation(
                            code=ConstraintCode.PRICE_BANDS,
                            severity=Severity.ERROR,
                            message="限价超出允许价格带",
                            scope="symbol",
                            symbol=order.symbol,
                            venue=order.venue,
                            details={"price": str(px), "market_price": str(mp), "band_pct": str(band), "lo": str(lo), "hi": str(hi)},
                        )
                    )

        # min_notional（order 层更准确）
        if spec.min_notional is not None:
            notional_px = None
            if order.limit_price is not None:
                notional_px = order.limit_price.value
            elif ctx.market_price is not None:
                notional_px = ctx.market_price
            if notional_px is not None:
                notional = q * notional_px
                if notional < spec.min_notional:
                    violations.append(
                        ConstraintViolation(
                            code=ConstraintCode.MIN_NOTIONAL,
                            severity=Severity.ERROR,
                            message="下单名义金额低于最小下单名义金额",
                            scope="symbol",
                            symbol=order.symbol,
                            venue=order.venue,
                            details={"notional": str(notional), "min_notional": str(spec.min_notional)},
                        )
                    )

        return ConstraintResult(violations=tuple(violations))


@dataclass(frozen=True, slots=True)
class TradingEnableConstraint:
    """
    交易开关类约束（symbol/venue/side 禁用）
    - 这是“规则层”，不是风控层
    """
    name: str = "trading_enable"

    disabled_symbols: Tuple[str, ...] = ()
    disabled_venues: Tuple[Venue, ...] = ()
    disabled_sides: Tuple[Side, ...] = ()

    def evaluate_intent(self, intent: IntentEvent, *, ctx: ConstraintContext, spec: Optional[InstrumentSpec] = None) -> ConstraintResult:
        violations: list[ConstraintViolation] = []

        if intent.symbol.normalized in self.disabled_symbols:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.SYMBOL_DISABLED,
                    severity=Severity.ERROR,
                    message="该 symbol 已被禁用交易",
                    scope="symbol",
                    symbol=intent.symbol,
                )
            )
        if spec is not None and spec.venue in self.disabled_venues:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.VENUE_DISABLED,
                    severity=Severity.ERROR,
                    message="该 venue 已被禁用交易",
                    scope="venue",
                    symbol=intent.symbol,
                    venue=spec.venue,
                )
            )
        if intent.side in self.disabled_sides:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.SIDE_DISABLED,
                    severity=Severity.ERROR,
                    message="该方向已被禁用交易",
                    scope="symbol",
                    symbol=intent.symbol,
                    details={"side": intent.side.value},
                )
            )
        return ConstraintResult(violations=tuple(violations))

    def evaluate_order(self, order: OrderEvent, *, ctx: ConstraintContext, spec: Optional[InstrumentSpec] = None) -> ConstraintResult:
        violations: list[ConstraintViolation] = []

        if order.symbol.normalized in self.disabled_symbols:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.SYMBOL_DISABLED,
                    severity=Severity.ERROR,
                    message="该 symbol 已被禁用下单",
                    scope="symbol",
                    symbol=order.symbol,
                    venue=order.venue,
                )
            )
        if order.venue in self.disabled_venues:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.VENUE_DISABLED,
                    severity=Severity.ERROR,
                    message="该 venue 已被禁用下单",
                    scope="venue",
                    symbol=order.symbol,
                    venue=order.venue,
                )
            )
        if order.side in self.disabled_sides:
            violations.append(
                ConstraintViolation(
                    code=ConstraintCode.SIDE_DISABLED,
                    severity=Severity.ERROR,
                    message="该方向已被禁用下单",
                    scope="symbol",
                    symbol=order.symbol,
                    venue=order.venue,
                    details={"side": order.side.value},
                )
            )
        return ConstraintResult(violations=tuple(violations))


# ============================================================
# 七、常用工厂：构建“机构级最小约束集”
# ============================================================

def build_default_constraints(
    *,
    disabled_symbols: Sequence[str] = (),
    disabled_venues: Sequence[Venue] = (),
    disabled_sides: Sequence[Side] = (),
) -> ConstraintSet:
    """
    默认约束集（可直接用于 production 的最小组合）
    """
    return ConstraintSet(
        constraints=(
            BasicFieldsConstraint(),
            TradingEnableConstraint(
                disabled_symbols=tuple(disabled_symbols),
                disabled_venues=tuple(disabled_venues),
                disabled_sides=tuple(disabled_sides),
            ),
            VenueSpecConstraint(),
        )
    )
