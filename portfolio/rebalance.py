# quant_system/portfolio/rebalance.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Iterable



# ============================================================
# 顶级机构级 portfolio/rebalance.py
# ============================================================
# 设计定位：
# - rebalance 是 portfolio -> execution 的桥梁
# - 输入：AllocationPlan（目标仓位）、当前持仓、价格、交易规则（step/tick/min_notional）
# - 输出：一组“交易意图”（TradeIntent），由 policy/risk 再次 gate，然后交给 execution/oms
#
# 关键原则（机构实务）：
# 1) 永远优先“减仓/风险收敛”：
#    - 当目标仓位绝对值 < 当前仓位绝对值：产生 reduce_only intent
# 2) 控制噪声与过度交易：
#    - 有 deadband（最小偏离带）
#    - 有 min_notional（过小订单直接忽略）
# 3) 规则与市场微结构：
#    - qty_step / price_tick / min_qty / min_notional / lot_size
# 4) 输出必须可审计：
#    - 每个 intent 附带原因、计算过程关键字段
#
# 注意：
# - rebalance 不是 risk：不做一票否决，只做“生成合理意图”
# - risk 层会对 intent/order 再做最终裁决（REDUCE/REJECT/KILL）
# ============================================================


# =========================
# Protocol：输入解耦
# =========================

class PriceProvider(Protocol):
    def price(self, symbol: str) -> Decimal:
        ...


class AccountSnapshot(Protocol):
    @property
    def equity(self) -> Decimal: ...
    @property
    def positions_qty(self) -> Mapping[str, Decimal]: ...


# =========================
# 交易规则（微结构/交易所约束）
# =========================

@dataclass(frozen=True, slots=True)
class InstrumentRules:
    """
    单标的交易规则（来自 exchange/broker 的 normalization 层）
    """
    symbol: str
    qty_step: Decimal                      # 数量步进（例如 0.001 BTC）
    min_qty: Decimal = Decimal("0")        # 最小下单数量
    min_notional: Decimal = Decimal("0")   # 最小名义金额（例如 5 USDT）
    max_qty: Optional[Decimal] = None      # 最大下单数量（如交易所限制）
    allow_short: bool = True               # 现货 false，合约 true


@dataclass(frozen=True, slots=True)
class RebalanceConfig:
    """
    调仓配置（组合层偏好）
    """
    # deadband：目标和当前差距的最小触发阈值（按名义/权益比例）
    # 例如 0.002 表示偏离 < 0.2% equity 的就不动
    deadband_notional_pct: Decimal = Decimal("0.002")

    # 每次调仓的最大名义变化比例（防止一次性巨量调仓）
    # 例如 0.10 表示单次最多调整 10% equity 的名义
    per_symbol_delta_cap_pct: Optional[Decimal] = Decimal("0.10")

    # 订单方向生成策略：默认按“目标仓位 - 当前仓位”决定
    # 是否允许把目标拆成多笔（这里最小冻结版：不拆单）
    allow_split: bool = False

    # rounding：数量取整方式（默认向下，避免超预算）
    qty_rounding: str = "down"  # "down" or "nearest"


# =========================
# 输出：TradeIntent（给 event/risk/execution）
# =========================

@dataclass(frozen=True, slots=True)
class TradeIntent:
    """
    调仓产生的交易意图（portfolio -> event）
    qty_delta：signed quantity（>0 买入，<0 卖出）
    reduce_only：是否只允许减仓（关键）
    """
    symbol: str
    qty_delta: Decimal
    notional_delta_abs: Decimal
    reduce_only: bool
    reason: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def side(self) -> str:
        return "BUY" if self.qty_delta > 0 else "SELL"


@dataclass(frozen=True, slots=True)
class RebalanceDiagnostics:
    """
    本次 rebalance 诊断信息（可审计）
    """
    equity: Decimal
    total_intents: int
    total_notional_delta: Decimal
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RebalancePlan:
    ts: Any
    intents: Tuple[TradeIntent, ...]
    diagnostics: RebalanceDiagnostics
    tags: Tuple[str, ...] = ()


# =========================
# 工具函数
# =========================

def _d(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def _abs(x: Decimal) -> Decimal:
    return x if x >= 0 else -x


def _sign(x: Decimal) -> Decimal:
    if x > 0:
        return Decimal("1")
    if x < 0:
        return Decimal("-1")
    return Decimal("0")


def _round_qty(qty: Decimal, step: Decimal, *, mode: str) -> Decimal:
    """
    将 qty 量化到 step 网格上（绝对值量化后再恢复符号）
    - down：向下取整（保守）
    - nearest：最近邻（更贴近目标，但可能略超预算）
    """
    if step <= 0:
        return qty

    s = _sign(qty)
    q = _abs(qty)

    # 计算 q / step 并量化
    n = (q / step)
    if mode == "nearest":
        n_int = n.to_integral_value(rounding=ROUND_HALF_UP)
    else:
        n_int = n.to_integral_value(rounding=ROUND_DOWN)

    return s * (n_int * step)


def _clamp_qty(qty: Decimal, *, min_qty: Decimal, max_qty: Optional[Decimal]) -> Decimal:
    q = qty
    # min_qty 针对绝对值
    if _abs(q) < min_qty:
        return Decimal("0")
    if max_qty is not None and _abs(q) > max_qty:
        q = _sign(q) * max_qty
    return q


# =========================
# 核心：Rebalancer
# =========================

class RebalanceError(RuntimeError):
    pass


class Rebalancer:
    """
    顶级机构级 Rebalancer（可冻结版）

    输入：
      - targets_qty：目标仓位（signed qty）
      - current_qty：当前仓位（signed qty）
      - rules：交易所规则（step/min_notional）
      - prices：价格源

    输出：
      - intents：给 policy/risk/execution 的“交易意图”
    """

    def __init__(self, *, cfg: Optional[RebalanceConfig] = None) -> None:
        self._cfg = cfg or RebalanceConfig()

    def build_plan(
        self,
        *,
        ts: Any,
        symbols: Sequence[str],
        targets_qty: Mapping[str, Decimal],
        account: AccountSnapshot,
        prices: PriceProvider,
        rules: Mapping[str, InstrumentRules],
        tags: Tuple[str, ...] = (),
    ) -> RebalancePlan:
        equity = _d(account.equity)
        if equity <= 0:
            raise RebalanceError("account.equity 必须为正")

        cfg = self._cfg
        notes: list[str] = []
        intents: list[TradeIntent] = []

        deadband_notional = cfg.deadband_notional_pct * equity

        for s in symbols:
            if s not in rules:
                raise RebalanceError(f"缺少交易规则：{s}")

            r = rules[s]
            px = _d(prices.price(s))
            if px <= 0:
                raise RebalanceError(f"无效价格：{s} price={px}")

            cur = _d(account.positions_qty.get(s, 0))
            tgt = _d(targets_qty.get(s, 0))
            delta = tgt - cur

            # 不允许做空：目标与 delta 都要裁剪
            if not r.allow_short:
                if tgt < 0:
                    tgt = Decimal("0")
                delta = tgt - cur
                if delta < 0 and cur <= 0:
                    # 已经是 0 或空头，且不允许做空 -> 不动
                    delta = Decimal("0")

            # deadband：按名义金额
            delta_notional_abs = _abs(delta) * px
            if delta_notional_abs < deadband_notional:
                continue

            # per-symbol delta cap：限制单标的本次调整的最大名义
            if cfg.per_symbol_delta_cap_pct is not None:
                cap_notional = _d(cfg.per_symbol_delta_cap_pct) * equity
                if delta_notional_abs > cap_notional:
                    # 按比例缩放 delta
                    scale = cap_notional / delta_notional_abs
                    notes.append(f"delta_cap: {s} {delta_notional_abs} -> {cap_notional}, scaled_by={scale}")
                    delta = delta * scale

            # 数量取整（按 step）
            delta = _round_qty(delta, _d(r.qty_step), mode=cfg.qty_rounding)

            # min_qty / max_qty 限制
            delta = _clamp_qty(delta, min_qty=_d(r.min_qty), max_qty=r.max_qty)

            # 再次计算名义（取整后）
            delta_notional_abs = _abs(delta) * px
            if delta == 0:
                continue

            # min_notional：过小订单直接忽略
            if _d(r.min_notional) > 0 and delta_notional_abs < _d(r.min_notional):
                notes.append(f"min_notional_skip: {s} {delta_notional_abs} < {r.min_notional}")
                continue

            # reduce_only 判定（机构级关键逻辑）
            # 规则：只要此订单会减少“绝对仓位”，就标记 reduce_only=True
            # 当前绝对仓位 > 目标绝对仓位 => 任何朝向目标的 delta 都应 reduce_only
            reduce_only = _abs(tgt) < _abs(cur)

            # 如果当前有仓位，且 delta 与 cur 方向相反，也属于减仓（更严格）
            if cur != 0 and (cur > 0 > delta or cur < 0 < delta):
                reduce_only = True

            intents.append(
                TradeIntent(
                    symbol=s,
                    qty_delta=delta,
                    notional_delta_abs=delta_notional_abs,
                    reduce_only=reduce_only,
                    reason="rebalance_to_target",
                    meta={
                        "price": str(px),
                        "cur_qty": str(cur),
                        "tgt_qty": str(tgt),
                        "raw_delta_qty": str(tgt - cur),
                        "final_delta_qty": str(delta),
                        "deadband_notional": str(deadband_notional),
                        "min_notional": str(r.min_notional),
                    },
                )
            )

        total_notional_delta = sum(i.notional_delta_abs for i in intents)
        diag = RebalanceDiagnostics(
            equity=equity,
            total_intents=len(intents),
            total_notional_delta=total_notional_delta,
            notes=tuple(notes),
        )

        # 稳定排序：优先 reduce_only，其次按名义变化从大到小（先减仓/先大单）
        intents_sorted = sorted(
            intents,
            key=lambda x: (0 if x.reduce_only else 1, -x.notional_delta_abs, x.symbol),
        )

        return RebalancePlan(
            ts=ts,
            intents=tuple(intents_sorted),
            diagnostics=diag,
            tags=tags + ("rebalancer",),
        )


# ============================================================
# 工具：从 allocator 的 targets 转成 targets_qty
# ============================================================

def targets_to_qty(targets: Iterable[Any]) -> Dict[str, Decimal]:
    """
    兼容 TargetPosition（allocator.py）：
      - t.symbol
      - t.target_qty
    """
    out: Dict[str, Decimal] = {}
    for t in targets:
        s = getattr(t, "symbol")
        q = getattr(t, "target_qty")
        out[str(s)] = _d(q)
    return out
