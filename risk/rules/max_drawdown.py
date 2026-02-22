# quant_system/risk/rules/max_drawdown.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Optional, Tuple

from event.types import IntentEvent, OrderEvent, Side, Symbol, Venue
from risk.decisions import (
    RiskAction,
    RiskCode,
    RiskDecision,
    RiskScope,
    RiskViolation,
)


class MaxDrawdownRuleError(RuntimeError):
    pass


def _d(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def _abs(x: Decimal) -> Decimal:
    return x if x >= 0 else -x


def _norm_symbol(s: Symbol | str) -> str:
    return s.normalized if hasattr(s, "normalized") else str(s)


def _get_from_meta(meta: Mapping[str, Any], *keys: str, default=None):
    for k in keys:
        if k in meta:
            return meta[k]
    return default


def _get_equity(meta: Mapping[str, Any]) -> Optional[Decimal]:
    v = _get_from_meta(meta, "equity", "account_equity", "nav", default=None)
    return None if v is None else _d(v)


def _get_peak_equity(meta: Mapping[str, Any]) -> Optional[Decimal]:
    """
    推荐上层维护峰值并注入 meta（回测/实盘都一样）
    支持 keys：
      - peak_equity
      - high_watermark
      - hwm_equity
    """
    v = _get_from_meta(meta, "peak_equity", "high_watermark", "hwm_equity", default=None)
    return None if v is None else _d(v)


def _get_drawdown_pct(meta: Mapping[str, Any]) -> Optional[Decimal]:
    """
    如果上层已算好 drawdown_pct，优先使用（最推荐）。
    允许 keys：
      - drawdown_pct   (0.12 表示回撤 12%)
      - dd_pct
    """
    v = _get_from_meta(meta, "drawdown_pct", "dd_pct", default=None)
    return None if v is None else _d(v)


def _get_position_qty(meta: Mapping[str, Any], *, venue: Optional[Venue], symbol: Symbol) -> Decimal:
    """
    用于判断“是否减仓/收敛风险”。
    规则不依赖 AccountState 类型，从 meta 读取当前仓位（signed qty）。

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


def _signed_delta_qty(side: Side, qty: Decimal) -> Decimal:
    # 约定：多头仓位为正，空头仓位为负
    return qty if side == Side.BUY else -qty


@dataclass(frozen=True, slots=True)
class MaxDrawdownRule:
    """
    顶级机构级：最大回撤限制（Max Drawdown）

    重要原则（机构实务）：
    1) max_drawdown 是“账户级硬风险边界”，通常触发后进入：
         - HARD_KILL（停止新增风险）
         - 或 REDUCE_ONLY（只允许减仓）
       这里由 action_on_breach 决定：KILL 或 REJECT（更保守：KILL）

    2) 触发后必须允许“风险收敛行为”：
         - reduce_only 订单必须放行
         - 或者订单确实在减少绝对仓位，也应放行
       否则会出现“越亏越无法减仓”的灾难性锁死。

    3) 规则不维护峰值，不修改状态：
         - peak_equity / drawdown_pct 应由上层（context/monitoring）维护并注入 meta
    """

    name: str = "max_drawdown"

    # 例如 0.20 表示最大回撤 20%
    max_drawdown_pct: Decimal = Decimal("0.20")

    # 可选：按策略覆盖（如果你上层能提供 strategy_id 维度的 drawdown_pct）
    per_strategy_max_dd: Mapping[str, Decimal] = field(default_factory=dict)

    # 触发时动作：默认 KILL（更符合机构“止血”习惯）
    action_on_breach: RiskAction = RiskAction.KILL

    # 触发后是否允许减仓/收敛风险订单通过（强烈建议 True）
    allow_reduce_after_breach: bool = True

    # 若无法评估回撤（缺 peak/drawdown），订单阶段默认拒绝（更安全）
    reject_if_cannot_evaluate: bool = True

    # 若 action_on_breach=KILL，可附带建议 ttl（供上层 kill_switch 使用）
    # 注意：KillSwitch 不在本规则里调用；这里只在 meta/tags 中给出建议
    suggested_kill_ttl_seconds: Optional[int] = 3600

    def _cap_for_strategy(self, strategy_id: Optional[str]) -> Decimal:
        if strategy_id and strategy_id in self.per_strategy_max_dd:
            return _d(self.per_strategy_max_dd[strategy_id])
        return _d(self.max_drawdown_pct)

    def _is_reducing_exposure(self, cur_qty: Decimal, delta_qty: Decimal) -> bool:
        if cur_qty == 0:
            return False
        return (cur_qty > 0 and delta_qty < 0) or (cur_qty < 0 and delta_qty > 0)

    def _compute_drawdown_pct(self, *, equity: Decimal, peak: Decimal) -> Optional[Decimal]:
        if peak <= 0:
            return None
        if equity >= peak:
            return Decimal("0")
        return (peak - equity) / peak

    def _evaluate_drawdown(self, *, meta: Mapping[str, Any]) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        返回 (equity, peak_equity, drawdown_pct)
        drawdown_pct 优先从 meta 读取，否则用 equity/peak 计算。
        """
        equity = _get_equity(meta)
        peak = _get_peak_equity(meta)
        dd = _get_drawdown_pct(meta)

        if dd is not None:
            # dd 已由上层计算（最推荐）
            return equity, peak, dd

        if equity is None or peak is None:
            return equity, peak, None

        return equity, peak, self._compute_drawdown_pct(equity=equity, peak=peak)

    # ---------------------------
    # Intent 评估（保守预检）
    # ---------------------------

    def evaluate_intent(self, intent: IntentEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        sym = _norm_symbol(intent.symbol)
        strategy_id = _get_from_meta(meta, "strategy_id", "strategy", default=None)
        cap = self._cap_for_strategy(strategy_id)

        equity, peak, dd = self._evaluate_drawdown(meta=meta)
        if dd is None:
            # intent 阶段允许缺失（不误杀）；订单阶段再硬限制
            return RiskDecision.allow(tags=(self.name, "intent_skip_missing_dd"))

        if dd <= cap:
            return RiskDecision.allow(tags=(self.name,))

        v = RiskViolation(
            code=RiskCode.MAX_DRAWDOWN,
            message="触发最大回撤限制（intent 预检）",
            scope=RiskScope.ACCOUNT,
            symbol=sym,
            strategy_id=str(strategy_id) if strategy_id is not None else None,
            severity="error",
            details={
                "equity": None if equity is None else str(equity),
                "peak_equity": None if peak is None else str(peak),
                "drawdown_pct": str(dd),
                "cap": str(cap),
                "action_on_breach": self.action_on_breach.value,
            },
        )

        # intent 阶段一般直接 REJECT（避免在 intent 层引入复杂“减仓放行”判定）
        return RiskDecision.reject((v,), scope=RiskScope.ACCOUNT, tags=(self.name,))

    # ---------------------------
    # Order 评估（硬限制主战场）
    # ---------------------------

    def evaluate_order(self, order: OrderEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        sym = _norm_symbol(order.symbol)
        strategy_id = _get_from_meta(meta, "strategy_id", "strategy", default=None)
        cap = self._cap_for_strategy(strategy_id)

        equity, peak, dd = self._evaluate_drawdown(meta=meta)

        if dd is None:
            if not self.reject_if_cannot_evaluate:
                return RiskDecision.allow(tags=(self.name, "skip_missing_dd"))
            v = RiskViolation(
                code=RiskCode.MAX_DRAWDOWN,
                message="无法评估最大回撤：缺少 drawdown_pct 或 (equity, peak_equity)",
                scope=RiskScope.ACCOUNT,
                symbol=sym,
                strategy_id=str(strategy_id) if strategy_id is not None else None,
                severity="error",
                details={
                    "equity": None if equity is None else str(equity),
                    "peak_equity": None if peak is None else str(peak),
                    "hint": "meta 提供 drawdown_pct 或同时提供 equity + peak_equity",
                },
            )
            return RiskDecision.reject((v,), scope=RiskScope.ACCOUNT, tags=(self.name,))

        # 未触发：放行
        if dd <= cap:
            return RiskDecision.allow(tags=(self.name,))

        # 触发后：默认进入“止血模式”
        # 1) 如果订单是减仓/收敛风险，且允许减仓，则放行
        if self.allow_reduce_after_breach:
            cur_qty = _get_position_qty(meta, venue=order.venue, symbol=order.symbol)
            delta_qty = _signed_delta_qty(order.side, _d(order.qty.value))
            if getattr(order, "reduce_only", False) or self._is_reducing_exposure(cur_qty, delta_qty):
                return RiskDecision.allow(tags=(self.name, "reduce_only_after_dd"))

        # 2) 否则：输出 KILL 或 REJECT（由配置决定）
        details = {
            "equity": None if equity is None else str(equity),
            "peak_equity": None if peak is None else str(peak),
            "drawdown_pct": str(dd),
            "cap": str(cap),
            "action_on_breach": self.action_on_breach.value,
        }
        if self.action_on_breach == RiskAction.KILL and self.suggested_kill_ttl_seconds is not None:
            details["suggested_kill_ttl_seconds"] = int(self.suggested_kill_ttl_seconds)

        v = RiskViolation(
            code=RiskCode.MAX_DRAWDOWN,
            message="触发最大回撤限制（账户级止血）",
            scope=RiskScope.ACCOUNT,
            symbol=sym,
            strategy_id=str(strategy_id) if strategy_id is not None else None,
            severity="fatal" if self.action_on_breach == RiskAction.KILL else "error",
            details=details,
        )

        if self.action_on_breach == RiskAction.KILL:
            # scope=ACCOUNT：由上层 kill_switch 决定是 account/global/strategy 的具体落点
            return RiskDecision.kill((v,), scope=RiskScope.ACCOUNT, tags=(self.name, "kill_on_dd"))
        return RiskDecision.reject((v,), scope=RiskScope.ACCOUNT, tags=(self.name, "reject_on_dd"))
