# quant_system/risk/rules/max_position.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Optional, Tuple

from event.types import IntentEvent, OrderEvent, Side, Venue, Symbol
from risk.decisions import (
    RiskAction,
    RiskAdjustment,
    RiskCode,
    RiskDecision,
    RiskScope,
    RiskViolation,
)


class MaxPositionRuleError(RuntimeError):
    pass


def _d(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def _norm_symbol(s: Symbol | str) -> str:
    return s.normalized if hasattr(s, "normalized") else str(s)


def _abs(x: Decimal) -> Decimal:
    return x if x >= 0 else -x


def _signed_delta(side: Side, qty: Decimal) -> Decimal:
    # BUY 增加仓位，SELL 减少仓位（空头仓位用负数表示）
    return qty if side == Side.BUY else -qty


def _get_from_meta(meta: Mapping[str, Any], *keys: str, default=None):
    for k in keys:
        if k in meta:
            return meta[k]
    return default


def _get_current_position_qty(meta: Mapping[str, Any], *, venue: Optional[Venue], symbol: Symbol) -> Decimal:
    """
    约定：Risk 规则不强依赖你的 Context/AccountState 类型，因此从 meta 读取“当前仓位”。

    支持的 meta 形态（任选其一）：
    1) meta["position_qty"] = Decimal/float/int/str （单 symbol 单 venue 已预先选好）
    2) meta["positions_qty"] = { (venue, "BTCUSDT"): Decimal, ... }
    3) meta["positions_qty"] = { "BINANCE:BTCUSDT": Decimal, ... }  (字符串 key)
    4) meta["account_positions"] = { (venue, "BTCUSDT"): {"qty": Decimal, ...}, ... }  (更丰富的结构)
    """
    # 1) 直接给了当前 symbol 的 qty
    v = _get_from_meta(meta, "position_qty", "cur_position_qty", default=None)
    if v is not None:
        return _d(v)

    sym = _norm_symbol(symbol)

    # 2) positions_qty: dict[(venue, sym)] -> qty
    positions_qty = _get_from_meta(meta, "positions_qty", "positions", default=None)
    if isinstance(positions_qty, Mapping):
        # tuple key
        if venue is not None and (venue, sym) in positions_qty:
            return _d(positions_qty[(venue, sym)])
        # string key: "VENUE:SYMBOL"
        if venue is not None and f"{venue.value}:{sym}" in positions_qty:
            return _d(positions_qty[f"{venue.value}:{sym}"])
        # 无 venue 时，退化为按 symbol 找（不推荐，但允许最小系统先跑通）
        for k, val in positions_qty.items():
            if isinstance(k, tuple) and len(k) == 2 and k[1] == sym:
                return _d(val)
            if isinstance(k, str) and k.endswith(f":{sym}"):
                return _d(val)

    # 3) account_positions: dict[(venue, sym)] -> {"qty": ...}
    account_positions = _get_from_meta(meta, "account_positions", "positions_detail", default=None)
    if isinstance(account_positions, Mapping):
        if venue is not None and (venue, sym) in account_positions:
            rec = account_positions[(venue, sym)]
            if isinstance(rec, Mapping) and "qty" in rec:
                return _d(rec["qty"])
        if venue is not None and f"{venue.value}:{sym}" in account_positions:
            rec = account_positions[f"{venue.value}:{sym}"]
            if isinstance(rec, Mapping) and "qty" in rec:
                return _d(rec["qty"])

    # 默认无仓位
    return Decimal("0")


def _get_market_price(meta: Mapping[str, Any]) -> Optional[Decimal]:
    v = _get_from_meta(meta, "market_price", "last_price", "px", default=None)
    if v is None:
        return None
    return _d(v)


@dataclass(frozen=True, slots=True)
class MaxPositionRule:
    """
    顶级机构级：单标的最大仓位约束（风控规则）

    规则只做“限制与调整建议”，不做执行、不改状态。
    输出：
      - ALLOW：未触发上限
      - REDUCE：可以把本次下单缩量到上限以内
      - REJECT：无法合理调整（例如没有价格无法做 notional 判断、或超过过多且不允许自动缩量）
      - KILL：一般不在此规则触发（KILL 多为 drawdown / liquidation / data stale 等），这里默认不使用

    关键设计点：
    - 支持 qty 上限（max_abs_qty）
    - 支持 notional 上限（max_abs_notional）
    - 支持 per-symbol override
    - 对“减仓方向”天然放行（即使当前已超限，也允许 reduce-only/减仓把风险降下来）
    """

    name: str = "max_position"

    # 全局默认上限（可为空表示不限制）
    max_abs_qty: Optional[Decimal] = None
    max_abs_notional: Optional[Decimal] = None

    # 分品种覆盖：normalized_symbol -> (max_abs_qty, max_abs_notional)
    # 任何一个可为 None，表示只覆盖另一项
    per_symbol_limits: Mapping[str, Tuple[Optional[Decimal], Optional[Decimal]]] = field(default_factory=dict)

    # 是否允许自动给出缩量建议（REDUCE）
    allow_auto_reduce: bool = True

    # 当缺少价格无法计算 notional 时的行为：True=REJECT，False=仅按 qty 检查
    reject_if_missing_price_for_notional: bool = True

    def _limits_for(self, symbol: Symbol) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        sym = _norm_symbol(symbol)
        if sym in self.per_symbol_limits:
            q, n = self.per_symbol_limits[sym]
            return q if q is None else _d(q), n if n is None else _d(n)
        return self.max_abs_qty, self.max_abs_notional

    def _is_reducing_exposure(self, cur_qty: Decimal, delta: Decimal) -> bool:
        """
        判断本次动作是否“减少绝对仓位”
        - cur_qty > 0 (多头)，delta < 0 => 减仓
        - cur_qty < 0 (空头)，delta > 0 => 减仓
        - cur_qty == 0：不属于减仓
        """
        if cur_qty == 0:
            return False
        return (cur_qty > 0 and delta < 0) or (cur_qty < 0 and delta > 0)

    # ---------------------------
    # 对 Intent 的评估
    # ---------------------------

    def evaluate_intent(self, intent: IntentEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        """
        Intent 层尽量不做精细限制（因为还没变成具体订单），但可以做“硬上限预检”：
        - 若 intent 给了 target_qty：按 qty 检查
        - 若 intent 给了 target_position_notional：按 notional 检查（需要价格或直接给 notional）
        """
        max_qty, max_notional = self._limits_for(intent.symbol)
        sym = _norm_symbol(intent.symbol)

        # 当前仓位（尽量读取；没有也不报错）
        venue = _get_from_meta(meta, "venue", default=None)
        cur_qty = _get_current_position_qty(meta, venue=venue, symbol=intent.symbol)

        # 1) qty 预检
        if intent.target_qty is not None and max_qty is not None:
            target_delta = _signed_delta(intent.side, _d(intent.target_qty.value))
            projected = cur_qty + target_delta

            # 如果是减仓方向，放行（哪怕当前已超限）
            if self._is_reducing_exposure(cur_qty, target_delta):
                return RiskDecision.allow(tags=(self.name, "reducing_exposure"))

            if _abs(projected) > _d(max_qty):
                v = RiskViolation(
                    code=RiskCode.MAX_POSITION,
                    message="Intent 触发最大仓位上限（按数量）",
                    scope=RiskScope.SYMBOL,
                    symbol=sym,
                    severity="error",
                    details={
                        "cur_qty": str(cur_qty),
                        "delta_qty": str(target_delta),
                        "projected_qty": str(projected),
                        "max_abs_qty": str(max_qty),
                    },
                )
                # Intent 阶段通常直接 REJECT（避免把“缩量逻辑”放到 intent 层）
                return RiskDecision.reject((v,), scope=RiskScope.SYMBOL, tags=(self.name,))

        # 2) notional 预检
        if intent.target_position_notional is not None and max_notional is not None:
            target_notional = _d(intent.target_position_notional.amount)
            if _abs(target_notional) > _d(max_notional):
                v = RiskViolation(
                    code=RiskCode.MAX_NOTIONAL,
                    message="Intent 触发最大仓位名义金额上限（按 notional）",
                    scope=RiskScope.SYMBOL,
                    symbol=sym,
                    severity="error",
                    details={
                        "target_notional": str(target_notional),
                        "max_abs_notional": str(max_notional),
                    },
                )
                return RiskDecision.reject((v,), scope=RiskScope.SYMBOL, tags=(self.name,))

        return RiskDecision.allow(tags=(self.name,))

    # ---------------------------
    # 对 Order 的评估
    # ---------------------------

    def evaluate_order(self, order: OrderEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        """
        Order 层是“硬限制主战场”：
        - 基于当前仓位 + 本次订单，判断是否超出上限
        - 若允许自动缩量，输出 REDUCE + adjustment.max_qty
        - 若本次订单是减仓/ReduceOnly，则尽量放行
        """
        max_qty, max_notional = self._limits_for(order.symbol)
        sym = _norm_symbol(order.symbol)

        cur_qty = _get_current_position_qty(meta, venue=order.venue, symbol=order.symbol)
        delta = _signed_delta(order.side, _d(order.qty.value))
        projected = cur_qty + delta

        # 如果 reduce_only 或本次确实减少绝对仓位：放行
        if order.reduce_only or self._is_reducing_exposure(cur_qty, delta):
            return RiskDecision.allow(tags=(self.name, "reduce_only_or_reducing"))

        violations: list[RiskViolation] = []

        # 1) qty 上限
        if max_qty is not None and _abs(projected) > _d(max_qty):
            violations.append(
                RiskViolation(
                    code=RiskCode.MAX_POSITION,
                    message="订单触发最大仓位上限（按数量）",
                    scope=RiskScope.SYMBOL,
                    symbol=sym,
                    severity="error",
                    details={
                        "cur_qty": str(cur_qty),
                        "order_qty": str(order.qty.value),
                        "delta_qty": str(delta),
                        "projected_qty": str(projected),
                        "max_abs_qty": str(max_qty),
                    },
                )
            )

        # 2) notional 上限（需要价格）
        if max_notional is not None:
            px: Optional[Decimal] = None
            if order.limit_price is not None:
                px = _d(order.limit_price.value)
            else:
                px = _get_market_price(meta)

            if px is None:
                if self.reject_if_missing_price_for_notional:
                    violations.append(
                        RiskViolation(
                            code=RiskCode.MAX_NOTIONAL,
                            message="无法评估 notional 上限：缺少价格（limit_price/market_price）",
                            scope=RiskScope.SYMBOL,
                            symbol=sym,
                            severity="error",
                            details={"max_abs_notional": str(max_notional)},
                        )
                    )
                # 否则仅跳过 notional 检查
            else:
                projected_notional = _abs(projected) * px
                if projected_notional > _d(max_notional):
                    violations.append(
                        RiskViolation(
                            code=RiskCode.MAX_NOTIONAL,
                            message="订单触发最大仓位名义金额上限（按 notional）",
                            scope=RiskScope.SYMBOL,
                            symbol=sym,
                            severity="error",
                            details={
                                "cur_qty": str(cur_qty),
                                "projected_qty": str(projected),
                                "price": str(px),
                                "projected_notional": str(projected_notional),
                                "max_abs_notional": str(max_notional),
                            },
                        )
                    )

        if not violations:
            return RiskDecision.allow(tags=(self.name,))

        # 如果不允许自动缩量：直接 REJECT
        if not self.allow_auto_reduce:
            return RiskDecision.reject(tuple(violations), scope=RiskScope.SYMBOL, tags=(self.name,))

        # 尝试给出缩量建议（仅在 qty 上限场景可做得“确定”）
        # 计算允许的最大“新增绝对仓位”：
        #   max_abs_qty - abs(cur_qty)
        if max_qty is not None:
            headroom = _d(max_qty) - _abs(cur_qty)
            if headroom <= 0:
                # 当前已经满仓/超限：本次不是减仓，直接 REJECT
                return RiskDecision.reject(tuple(violations), scope=RiskScope.SYMBOL, tags=(self.name, "no_headroom"))

            # 本次订单最大可下数量（正数）
            # 注意：headroom 是“绝对仓位空间”，对 BUY/SELL 都一样
            max_order_qty = headroom
            if max_order_qty <= 0:
                return RiskDecision.reject(tuple(violations), scope=RiskScope.SYMBOL, tags=(self.name, "no_headroom"))

            adj = RiskAdjustment(
                max_qty=float(max_order_qty),  # 合同里是 float；执行层应再做精度/步进约束
                tags=(self.name, "auto_reduce"),
            )
            return RiskDecision.reduce(tuple(violations), adjustment=adj, scope=RiskScope.SYMBOL, tags=(self.name,))

        # 只有 notional 超限且无法精确换算 qty（缺 price / 或复杂合约）：保守 REJECT
        return RiskDecision.reject(tuple(violations), scope=RiskScope.SYMBOL, tags=(self.name, "notional_only"))
