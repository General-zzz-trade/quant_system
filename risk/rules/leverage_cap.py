# quant_system/risk/rules/leverage_cap.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Optional

from event.types import IntentEvent, OrderEvent, Side, Symbol, Venue
from risk.decisions import (
    RiskAdjustment,
    RiskCode,
    RiskDecision,
    RiskScope,
    RiskViolation,
)


class LeverageCapRuleError(RuntimeError):
    pass


def _d(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def _abs(x: Decimal) -> Decimal:
    return x if x >= 0 else -x


def _norm_symbol(s: Symbol | str) -> str:
    return s.normalized if hasattr(s, "normalized") else str(s)


def _signed_delta_qty(side: Side, qty: Decimal) -> Decimal:
    # 约定：多头仓位为正，空头仓位为负
    return qty if side == Side.BUY else -qty


def _get_from_meta(meta: Mapping[str, Any], *keys: str, default=None):
    for k in keys:
        if k in meta:
            return meta[k]
    return default


def _get_market_price(meta: Mapping[str, Any]) -> Optional[Decimal]:
    v = _get_from_meta(meta, "market_price", "mark_price", "last_price", "px", default=None)
    if v is None:
        return None
    return _d(v)


def _get_equity(meta: Mapping[str, Any]) -> Optional[Decimal]:
    v = _get_from_meta(meta, "equity", "account_equity", "nav", default=None)
    return None if v is None else _d(v)


def _get_position_qty(meta: Mapping[str, Any], *, venue: Optional[Venue], symbol: Symbol) -> Decimal:
    """
    读取当前仓位数量（signed qty）
    支持（任选其一）：
      - meta["position_qty"] / "cur_position_qty"
      - meta["positions_qty"][(venue, "BTCUSDT")] = qty
      - meta["positions_qty"]["BINANCE:BTCUSDT"] = qty
      - meta["account_positions"][(venue, "BTCUSDT")] = {"qty": ...}
    """
    v = _get_from_meta(meta, "position_qty", "cur_position_qty", default=None)
    if v is not None:
        return _d(v)

    sym = _norm_symbol(symbol)
    positions_qty = _get_from_meta(meta, "positions_qty", "positions", default=None)
    if isinstance(positions_qty, Mapping):
        if venue is not None and (venue, sym) in positions_qty:
            return _d(positions_qty[(venue, sym)])
        if venue is not None and f"{venue.value}:{sym}" in positions_qty:
            return _d(positions_qty[f"{venue.value}:{sym}"])
        # 无 venue 时，退化按 symbol 匹配（不推荐，但允许先跑通）
        for k, val in positions_qty.items():
            if isinstance(k, tuple) and len(k) == 2 and k[1] == sym:
                return _d(val)
            if isinstance(k, str) and k.endswith(f":{sym}"):
                return _d(val)

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

    return Decimal("0")


def _get_current_gross_notional(meta: Mapping[str, Any]) -> Optional[Decimal]:
    """
    优先读取 meta 里已经计算好的 gross_notional（最推荐）。
    若没有，则尝试用 positions + mark_price 计算（尽量保守）。
    """
    v = _get_from_meta(meta, "gross_notional", "gross_exposure", default=None)
    if v is not None:
        return _d(v)

    # 退化：从 positions 估算
    positions = _get_from_meta(meta, "positions_exposure", "positions_mark", "positions", default=None)
    if not isinstance(positions, Mapping):
        return None

    total = Decimal("0")
    for _, rec in positions.items():
        # rec 支持 dict: {"qty":..., "mark_price":..., "multiplier":...}
        if isinstance(rec, Mapping):
            if "qty" in rec and ("mark_price" in rec or "price" in rec):
                qty = _abs(_d(rec["qty"]))
                px = _d(rec.get("mark_price", rec.get("price")))
                mult = _d(rec.get("multiplier", "1"))
                total += qty * px * mult
    return total


def _notional_for_delta_qty(*, delta_qty: Decimal, price: Decimal, multiplier: Decimal = Decimal("1")) -> Decimal:
    return _abs(delta_qty) * price * multiplier


@dataclass(frozen=True, slots=True)
class LeverageCapRule:
    """
    顶级机构级：杠杆上限（Leverage Cap）规则

    定义（机构常用口径之一）：
      leverage = gross_notional / equity
    - gross_notional：所有持仓名义金额绝对值之和（不区分多空）
    - equity：账户权益（NAV）

    规则输出：
      - ALLOW：未超过杠杆上限
      - REDUCE：可以对“新增风险订单”缩量，使其不超过上限（需要价格）
      - REJECT：无法安全缩量（缺价格/缺权益/已无 headroom 等）
    """

    name: str = "leverage_cap"

    max_leverage: Decimal = Decimal("3")  # 默认 3x（示例；应由 config 驱动）
    per_symbol_max_leverage: Mapping[str, Decimal] = field(default_factory=dict)

    allow_auto_reduce: bool = True
    reject_if_missing_price: bool = True

    def _cap_for(self, symbol: Symbol) -> Decimal:
        sym = _norm_symbol(symbol)
        return _d(self.per_symbol_max_leverage.get(sym, self.max_leverage))

    def _is_reducing_exposure(self, cur_qty: Decimal, delta_qty: Decimal) -> bool:
        if cur_qty == 0:
            return False
        return (cur_qty > 0 and delta_qty < 0) or (cur_qty < 0 and delta_qty > 0)

    # ---------------------------
    # Intent 评估（保守预检）
    # ---------------------------

    def evaluate_intent(self, intent: IntentEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        cap = self._cap_for(intent.symbol)
        sym = _norm_symbol(intent.symbol)

        equity = _get_equity(meta)
        if equity is None or equity <= 0:
            v = RiskViolation(
                code=RiskCode.MAX_LEVERAGE,
                message="无法评估杠杆：缺少或无效 equity",
                scope=RiskScope.ACCOUNT,
                symbol=sym,
                severity="error",
                details={"equity": None if equity is None else str(equity)},
            )
            return RiskDecision.reject((v,), scope=RiskScope.ACCOUNT, tags=(self.name,))

        gross = _get_current_gross_notional(meta)
        if gross is None:
            # Intent 阶段允许缺 gross（不强依赖），直接放行；订单阶段再硬限制
            return RiskDecision.allow(tags=(self.name, "intent_skip_missing_gross"))

        cur_lev = gross / equity if equity > 0 else Decimal("0")
        if cur_lev > cap:
            v = RiskViolation(
                code=RiskCode.MAX_LEVERAGE,
                message="当前账户杠杆已超过上限（intent 预检）",
                scope=RiskScope.ACCOUNT,
                symbol=sym,
                severity="error",
                details={"gross_notional": str(gross), "equity": str(equity), "leverage": str(cur_lev),
                    "cap": str(cap)},
            )
            return RiskDecision.reject((v,), scope=RiskScope.ACCOUNT, tags=(self.name,))

        return RiskDecision.allow(tags=(self.name,))

    # ---------------------------
    # Order 评估（硬限制主战场）
    # ---------------------------

    def evaluate_order(self, order: OrderEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        cap = self._cap_for(order.symbol)
        sym = _norm_symbol(order.symbol)

        equity = _get_equity(meta)
        if equity is None or equity <= 0:
            v = RiskViolation(
                code=RiskCode.MAX_LEVERAGE,
                message="无法评估杠杆：缺少或无效 equity",
                scope=RiskScope.ACCOUNT,
                symbol=sym,
                severity="error",
                details={"equity": None if equity is None else str(equity)},
            )
            return RiskDecision.reject((v,), scope=RiskScope.ACCOUNT, tags=(self.name,))

        # 当前 gross_notional：强烈建议上层 meta 直接给；否则退化估算
        gross = _get_current_gross_notional(meta)
        if gross is None:
            v = RiskViolation(
                code=RiskCode.MAX_LEVERAGE,
                message="无法评估杠杆：缺少 gross_notional/positions_exposure",
                scope=RiskScope.ACCOUNT,
                symbol=sym,
                severity="error",
                details={"hint": "meta 提供 gross_notional 或 positions_exposure"},
            )
            return RiskDecision.reject((v,), scope=RiskScope.ACCOUNT, tags=(self.name,))

        # 若是减仓/ReduceOnly：放行（即使当前杠杆已超）
        cur_qty = _get_position_qty(meta, venue=order.venue, symbol=order.symbol)
        delta_qty = _signed_delta_qty(order.side, _d(order.qty.value))
        if getattr(order, "reduce_only", False) or self._is_reducing_exposure(cur_qty, delta_qty):
            return RiskDecision.allow(tags=(self.name, "reduce_only_or_reducing"))

        # 价格：limit_price 优先，否则用 meta 的 market_price
        px: Optional[Decimal] = None
        if getattr(order, "limit_price", None) is not None:
            px = _d(order.limit_price.value)
        else:
            px = _get_market_price(meta)

        if px is None:
            if self.reject_if_missing_price:
                v = RiskViolation(
                    code=RiskCode.MAX_LEVERAGE,
                    message="无法评估订单增量杠杆：缺少价格（limit_price/market_price）",
                    scope=RiskScope.ACCOUNT,
                    symbol=sym,
                    severity="error",
                    details={"cap": str(cap)},
                )
                return RiskDecision.reject((v,), scope=RiskScope.ACCOUNT, tags=(self.name,))
            # 不建议走这里：缺价格就无法保守判断新增杠杆，直接放行会很危险
            return RiskDecision.allow(tags=(self.name, "skip_missing_price"))

        # multiplier：若上层提供合约乘数，使用之；否则默认 1
        mult = _d(_get_from_meta(meta, "multiplier", "contract_multiplier", default="1"))
        delta_notional = _notional_for_delta_qty(delta_qty=delta_qty, price=px, multiplier=mult)

        projected_gross = gross + delta_notional
        projected_lev = projected_gross / equity if equity > 0 else Decimal("0")

        if projected_lev <= cap:
            return RiskDecision.allow(tags=(self.name,))

        # 超限：生成 violation
        violations = (
            RiskViolation(
                code=RiskCode.MAX_LEVERAGE,
                message="订单触发杠杆上限（gross_notional/equity）",
                scope=RiskScope.ACCOUNT,
                symbol=sym,
                severity="error",
                details={
                    "gross_notional": str(gross),
                    "delta_notional": str(delta_notional),
                    "projected_gross": str(projected_gross),
                    "equity": str(equity),
                    "projected_leverage": str(projected_lev),
                    "cap": str(cap),
                    "price": str(px),
                    "multiplier": str(mult),
                },
            ),
        )

        if not self.allow_auto_reduce:
            return RiskDecision.reject(violations, scope=RiskScope.ACCOUNT, tags=(self.name,))

        # 尝试自动缩量（REDUCE）：
        # 允许的最大 gross_notional = cap * equity
        max_gross = cap * equity
        headroom = max_gross - gross  # 还能增加的名义金额
        if headroom <= 0:
            return RiskDecision.reject(violations, scope=RiskScope.ACCOUNT, tags=(self.name, "no_headroom"))

        # 允许的最大订单 qty（按名义金额反推）：qty <= headroom / (price * multiplier)
        denom = px * mult
        if denom <= 0:
            return RiskDecision.reject(violations, scope=RiskScope.ACCOUNT, tags=(self.name, "bad_price_or_mult"))

        max_qty = headroom / denom
        if max_qty <= 0:
            return RiskDecision.reject(violations, scope=RiskScope.ACCOUNT, tags=(self.name, "no_headroom"))

        adj = RiskAdjustment(
            max_qty=float(max_qty),  # 执行层必须再按 step_size/precision 裁剪
            tags=(self.name, "auto_reduce"),
        )
        return RiskDecision.reduce(violations, adjustment=adj, scope=RiskScope.ACCOUNT, tags=(self.name,))
