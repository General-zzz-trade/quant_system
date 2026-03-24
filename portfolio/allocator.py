# quant_system/portfolio/allocator.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple


# ============================================================
# 顶级机构级 portfolio/allocator.py（核心：目标仓位生成器）
# ============================================================
# 设计定位：
# - Allocator 属于 portfolio 层：回答"资金如何分配/仓位如何配"
# - 它不下单、不做撮合、不做风控裁决（风控在 risk 层）
# - 它输出"目标仓位/目标权重/目标名义金额"（Target），给 rebalance/execution 去实现
#
# 核心能力（最小可冻结版本）：
# 1) 支持多种分配策略：
#    - TargetWeightAllocator：按目标权重
#    - EqualWeightAllocator：等权
#    - VolTargetAllocator：波动率目标（风险预算雏形）
# 2) 约束处理（组合层约束，不是风控）：
#    - 单标的 max_weight / max_notional
#    - 组合级 max_gross_leverage（组合预算）
#    - turnover_cap（换手上限，控制调仓幅度）
# 3) 输出可审计：解释每一步如何得到 target
#
# 注意：
# - 真实顶级机构还会加：协方差/风险模型、优化器（QP/二阶锥）、成本模型联动
# - 但 allocator 的"契约/边界/可复现"必须先正确（这版就是可冻结核心）
# ============================================================


# =========================
# 基础工具
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


def _safe_div(a: Decimal, b: Decimal) -> Decimal:
    if b == 0:
        return Decimal("0")
    return a / b


def _clamp(x: Decimal, lo: Optional[Decimal], hi: Optional[Decimal]) -> Decimal:
    if lo is not None and x < lo:
        return lo
    if hi is not None and x > hi:
        return hi
    return x


# =========================
# 输入协议（解耦 Context/AccountState）
# =========================

class PriceProvider(Protocol):
    """提供标的价格（mark/last）。"""
    def price(self, symbol: str) -> Decimal:
        ...


class AccountSnapshot(Protocol):
    """
    账户/组合事实快照（Allocator 输入）
    - equity：权益/NAV（用于计算 target_notional）
    - positions_qty：当前持仓（signed qty，多正空负）
    """
    @property
    def equity(self) -> Decimal: ...
    @property
    def positions_qty(self) -> Mapping[str, Decimal]: ...


# =========================
# 约束（组合层约束，不是 risk 的硬闸）
# =========================

@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    """
    组合层约束（Allocator 可用的"预算/偏好"）
    注意：这些不是 risk 层的一票否决，而是"组合目标要尽量满足"的软/硬约束。
    """
    # 单标的权重上限（例如 0.25 表示最多 25% 权重）
    max_weight: Optional[Decimal] = None

    # 单标的名义金额上限（例如 20000 表示最多 20000 USDT 名义）
    max_notional_per_symbol: Optional[Decimal] = None

    # 组合最大毛杠杆（gross_notional / equity），例如 2.0
    max_gross_leverage: Optional[Decimal] = None

    # 调仓换手上限（turnover cap）：限制每次调仓的"绝对名义增量 / equity"
    # 例如 0.10 表示本次调仓最多动用 10% 权益的名义增量
    turnover_cap: Optional[Decimal] = None

    # 是否允许做空（spot 账户通常 False；合约通常 True）
    allow_short: bool = True


# =========================
# 目标/计划输出（给 rebalance 执行）
# =========================

@dataclass(frozen=True, slots=True)
class TargetPosition:
    """
    输出：某标的目标仓位（以 qty 表示）
    - target_qty：signed qty，多正空负
    - target_notional：目标名义金额绝对值（可用于审计/排序）
    - target_weight：目标权重（signed weight，多正空负）
    """
    symbol: str
    target_qty: Decimal
    target_notional: Decimal
    target_weight: Decimal
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AllocationDiagnostics:
    """
    审计信息：用于回放 allocator 是如何得到结果的
    """
    equity: Decimal
    gross_before: Decimal
    gross_after: Decimal
    leverage_before: Decimal
    leverage_after: Decimal
    turnover_used: Decimal
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AllocationPlan:
    """
    Allocator 输出的完整计划
    """
    ts: Any
    targets: Tuple[TargetPosition, ...]
    diagnostics: AllocationDiagnostics
    tags: Tuple[str, ...] = ()


# =========================
# Allocator 基类与实现
# =========================

class AllocatorError(RuntimeError):
    pass


class BaseAllocator(Protocol):
    """
    Allocator 最小契约
    """
    name: str

    def allocate(
        self,
        *,
        ts: Any,
        symbols: Sequence[str],
        account: AccountSnapshot,
        prices: PriceProvider,
        constraints: PortfolioConstraints,
        # 可选输入：信号/权重/波动率等（由具体 allocator 使用）
        inputs: Mapping[str, Any] | None = None,
        tags: Tuple[str, ...] = (),
    ) -> AllocationPlan:
        ...


# -------------------------
# 核心：按目标权重分配
# -------------------------

@dataclass(frozen=True, slots=True)
class TargetWeightAllocator:
    """
    机构级核心：Target Weight Allocator

    输入 inputs:
      - target_weights: Mapping[str, float|Decimal]，例如 {"BTCUSDT": 0.3, "ETHUSDT": -0.2}
        * 权重可正可负（空头），是否允许空头由 constraints.allow_short 决定
      - weight_residual_to_cash: bool（默认 True）
        * 未分配的权重残差视为现金/不使用杠杆
    """
    name: str = "target_weight_allocator"

    def allocate(
        self,
        *,
        ts: Any,
        symbols: Sequence[str],
        account: AccountSnapshot,
        prices: PriceProvider,
        constraints: PortfolioConstraints,
        inputs: Mapping[str, Any] | None = None,
        tags: Tuple[str, ...] = (),
    ) -> AllocationPlan:
        if inputs is None or "target_weights" not in inputs:
            raise AllocatorError("TargetWeightAllocator 需要 inputs['target_weights']")

        equity = _d(account.equity)
        if equity <= 0:
            raise AllocatorError("account.equity 必须为正")

        raw_w: Mapping[str, Any] = inputs["target_weights"]

        # 1) 取出权重（未提供的 symbol 视为 0）
        w: Dict[str, Decimal] = {}
        for s in symbols:
            if s in raw_w:
                w[s] = _d(raw_w[s])
            else:
                w[s] = Decimal("0")

        # Validate prices up-front (both paths need valid prices)
        for s in symbols:
            px = _d(prices.price(s))
            if px <= 0:
                raise AllocatorError(f"价格无效：{s} price={px}")

        # --- Rust fast path: delegate weight->constraint->qty to Rust ---
        if _RUST_ALLOCATOR_AVAILABLE:
            final_targets = _rust_allocate_targets(
                symbols=symbols,
                target_weights=w,
                equity=equity,
                prices=prices,
                current_qty=account.positions_qty,
                constraints=constraints,
            )

            gross_before = _gross_notional(account.positions_qty, prices)
            gross_after = _gross_notional(
                {t.symbol: t.target_qty for t in final_targets.values()}, prices,
            )

            diag = AllocationDiagnostics(
                equity=equity,
                gross_before=gross_before,
                gross_after=gross_after,
                leverage_before=_safe_div(gross_before, equity),
                leverage_after=_safe_div(gross_after, equity),
                turnover_used=Decimal("0"),  # Rust handles turnover internally
                notes=("rust_delegated",),
            )

            return AllocationPlan(
                ts=ts,
                targets=tuple(final_targets[s] for s in symbols),
                diagnostics=diag,
                tags=tags + (self.name,),
            )

        # --- Python fallback path (unchanged) ---
        weight_residual_to_cash = bool(inputs.get("weight_residual_to_cash", True))

        # 2) 空头权限
        if not constraints.allow_short:
            for s in list(w.keys()):
                if w[s] < 0:
                    w[s] = Decimal("0")

        # 3) 单标的 max_weight（绝对权重上限）
        notes: list[str] = []
        if constraints.max_weight is not None:
            cap = _d(constraints.max_weight)
            for s in list(w.keys()):
                if _abs(w[s]) > cap:
                    notes.append(f"max_weight: {s} {w[s]} -> {_sign(w[s]) * cap}")
                    w[s] = _sign(w[s]) * cap

        # 4) 若要求"剩余为现金"，则不强制归一化；否则可选择归一化（这里保持机构常用：不强行归一化）
        #    解释：顶级机构常把权重当"目标风险/资金占用"，不一定要 sum=1
        if not weight_residual_to_cash:
            # 可选：归一化到 sum(abs)=1（更像 risk-parity 后处理）；但默认不做
            pass

        # 5) 将权重 -> 名义金额 -> qty
        targets_pre: Dict[str, TargetPosition] = {}
        for s in symbols:
            px = _d(prices.price(s))

            tw = w[s]
            # 目标名义金额：|w| * equity
            t_notional_abs = _abs(tw) * equity
            # qty = signed_notional / price
            t_qty = (tw * equity) / px

            targets_pre[s] = TargetPosition(
                symbol=s,
                target_qty=t_qty,
                target_notional=t_notional_abs,
                target_weight=tw,
                meta={"price": str(px)},
            )

        # 6) 组合级预算：max_gross_leverage
        targets_scaled, scale_note = _apply_gross_leverage_cap(
            targets_pre,
            equity=equity,
            max_gross_leverage=constraints.max_gross_leverage,
        )
        if scale_note:
            notes.append(scale_note)

        # 7) 单标的名义金额上限
        targets_capped, cap_notes = _apply_max_notional_per_symbol(
            targets_scaled,
            equity=equity,
            max_notional_per_symbol=constraints.max_notional_per_symbol,
            prices=prices,
        )
        notes.extend(cap_notes)

        # 8) turnover cap（换手上限）
        final_targets, turnover_used, turn_notes = _apply_turnover_cap(
            targets_capped,
            equity=equity,
            current_qty=account.positions_qty,
            prices=prices,
            turnover_cap=constraints.turnover_cap,
        )
        notes.extend(turn_notes)

        # 9) diagnostics
        gross_before = _gross_notional(account.positions_qty, prices)
        gross_after = _gross_notional({t.symbol: t.target_qty for t in final_targets.values()}, prices)

        diag = AllocationDiagnostics(
            equity=equity,
            gross_before=gross_before,
            gross_after=gross_after,
            leverage_before=_safe_div(gross_before, equity),
            leverage_after=_safe_div(gross_after, equity),
            turnover_used=turnover_used,
            notes=tuple(notes),
        )

        return AllocationPlan(
            ts=ts,
            targets=tuple(final_targets[s] for s in symbols),
            diagnostics=diag,
            tags=tags + (self.name,),
        )


# -------------------------
# 等权（在没信号时的 baseline）
# -------------------------

@dataclass(frozen=True, slots=True)
class EqualWeightAllocator:
    """
    等权分配（baseline）
    - 默认：对 symbols 等权分配到 sum(abs(w)) = 1（全资金使用）
    - 可配 gross_leverage 由 constraints 控制
    """
    name: str = "equal_weight_allocator"

    def allocate(
        self,
        *,
        ts: Any,
        symbols: Sequence[str],
        account: AccountSnapshot,
        prices: PriceProvider,
        constraints: PortfolioConstraints,
        inputs: Mapping[str, Any] | None = None,
        tags: Tuple[str, ...] = (),
    ) -> AllocationPlan:
        n = len(symbols)
        if n == 0:
            raise AllocatorError("symbols 不能为空")

        w = Decimal("1") / Decimal(str(n))
        tw = {s: w for s in symbols}

        return TargetWeightAllocator().allocate(
            ts=ts,
            symbols=symbols,
            account=account,
            prices=prices,
            constraints=constraints,
            inputs={"target_weights": tw, "weight_residual_to_cash": False},
            tags=tags + (self.name,),
        )


# -------------------------
# 波动率目标（风险预算雏形）
# -------------------------

@dataclass(frozen=True, slots=True)
class VolTargetAllocator:
    """
    波动率目标分配（风险预算雏形）

    输入 inputs:
      - target_vol: 目标组合年化波动（例如 0.15）
      - vols: Mapping[str, float|Decimal] 每个标的年化波动（例如 0.60）
      - base_weights: 可选，若提供则先按 base_weights 分配，再按目标波动缩放整体名义
        否则默认按 1/vol 等风险权重（近似 risk parity 的最小版本）

    输出：
      - 本质仍输出 target_weights（相对），然后用 max_gross_leverage 约束防止过度杠杆
    """
    name: str = "vol_target_allocator"

    def allocate(
        self,
        *,
        ts: Any,
        symbols: Sequence[str],
        account: AccountSnapshot,
        prices: PriceProvider,
        constraints: PortfolioConstraints,
        inputs: Mapping[str, Any] | None = None,
        tags: Tuple[str, ...] = (),
    ) -> AllocationPlan:
        if inputs is None:
            raise AllocatorError("VolTargetAllocator 需要 inputs")
        if "target_vol" not in inputs or "vols" not in inputs:
            raise AllocatorError("VolTargetAllocator 需要 inputs['target_vol'] 与 inputs['vols']")

        target_vol = _d(inputs["target_vol"])
        vols: Mapping[str, Any] = inputs["vols"]
        base_weights: Optional[Mapping[str, Any]] = inputs.get("base_weights")

        if target_vol <= 0:
            raise AllocatorError("target_vol 必须为正")

        # 1) 生成相对权重（未缩放）
        rel_w: Dict[str, Decimal] = {}
        if base_weights is not None:
            for s in symbols:
                rel_w[s] = _d(base_weights.get(s, 0))
        else:
            # 近似：w ~ 1/vol（vol 越大权重越小）
            for s in symbols:
                v = _d(vols.get(s))
                if v is None or v <= 0:
                    raise AllocatorError(f"缺少或无效波动率：{s} vol={v}")
                rel_w[s] = Decimal("1") / v

        # 2) 归一化到 sum(abs)=1（便于解释）
        denom = sum(_abs(x) for x in rel_w.values())
        if denom == 0:
            raise AllocatorError("权重全为 0")
        for s in list(rel_w.keys()):
            rel_w[s] = rel_w[s] / denom

        # 3) 估算组合波动（极简版本：假设相关性=0）
        #    port_vol ≈ sqrt(sum(w^2 * vol^2))
        port_var = Decimal("0")
        for s in symbols:
            v = _d(vols.get(s))
            port_var += (rel_w[s] * v) * (rel_w[s] * v)
        port_vol = port_var.sqrt() if port_var > 0 else Decimal("0")
        if port_vol <= 0:
            raise AllocatorError("组合波动估算为 0，检查 vols/base_weights")

        # 4) 缩放因子：scale = target_vol / port_vol
        scale = target_vol / port_vol

        # 5) 将 scale 作用到权重（意味着使用更多/更少名义）
        #    注意：最终仍会被 max_gross_leverage 截断，保证实盘安全
        scaled_w = {s: rel_w[s] * scale for s in symbols}

        return TargetWeightAllocator().allocate(
            ts=ts,
            symbols=symbols,
            account=account,
            prices=prices,
            constraints=constraints,
            inputs={"target_weights": scaled_w, "weight_residual_to_cash": True},
            tags=tags + (self.name,),
        )


# ============================================================
# 内部：约束与度量（组合层）
# ============================================================

def _gross_notional(positions_qty: Mapping[str, Decimal], prices: PriceProvider) -> Decimal:
    total = Decimal("0")
    for s, q in positions_qty.items():
        px = _d(prices.price(s))
        total += _abs(_d(q)) * px
    return total


def _apply_gross_leverage_cap(
    targets: Mapping[str, TargetPosition],
    *,
    equity: Decimal,
    max_gross_leverage: Optional[Decimal],
) -> Tuple[Dict[str, TargetPosition], Optional[str]]:
    if max_gross_leverage is None:
        return dict(targets), None

    cap = _d(max_gross_leverage)
    if cap <= 0:
        return dict(targets), "max_gross_leverage<=0，忽略"

    gross = sum(t.target_notional for t in targets.values())
    # gross_notional = sum(abs(w)*equity)（本实现里 target_notional 已是 abs 名义）
    # leverage = gross / equity
    lev = _safe_div(gross, equity)
    if lev <= cap or gross == 0:
        return dict(targets), None

    scale = cap / lev  # < 1
    out: Dict[str, TargetPosition] = {}
    for s, t in targets.items():
        out[s] = TargetPosition(
            symbol=s,
            target_qty=t.target_qty * scale,
            target_notional=t.target_notional * scale,
            target_weight=t.target_weight * scale,
            meta=dict(t.meta) | {"scaled_by_gross_leverage": str(scale)},
        )
    return out, f"gross_leverage_cap: leverage {lev} -> {cap}, scaled_by={scale}"


def _apply_max_notional_per_symbol(
    targets: Mapping[str, TargetPosition],
    *,
    equity: Decimal,
    max_notional_per_symbol: Optional[Decimal],
    prices: PriceProvider,
) -> Tuple[Dict[str, TargetPosition], Tuple[str, ...]]:
    if max_notional_per_symbol is None:
        return dict(targets), ()

    cap = _d(max_notional_per_symbol)
    if cap <= 0:
        return dict(targets), ("max_notional_per_symbol<=0，忽略",)

    notes: list[str] = []
    out: Dict[str, TargetPosition] = {}
    for s, t in targets.items():
        if t.target_notional <= cap:
            out[s] = t
            continue

        px = _d(prices.price(s))
        if px <= 0:
            out[s] = t
            continue

        # 限制名义金额后反推 qty/weight
        # signed_notional = sign(weight) * cap
        signed_notional = _sign(t.target_weight) * cap
        new_qty = signed_notional / px
        new_w = signed_notional / equity

        notes.append(f"max_notional_per_symbol: {s} {t.target_notional} -> {cap}")
        out[s] = TargetPosition(
            symbol=s,
            target_qty=new_qty,
            target_notional=cap,
            target_weight=new_w,
            meta=dict(t.meta) | {"capped_notional": str(cap)},
        )

    return out, tuple(notes)


def _apply_turnover_cap(
    targets: Mapping[str, TargetPosition],
    *,
    equity: Decimal,
    current_qty: Mapping[str, Decimal],
    prices: PriceProvider,
    turnover_cap: Optional[Decimal],
) -> Tuple[Dict[str, TargetPosition], Decimal, Tuple[str, ...]]:
    """
    turnover 定义（机构常用之一）：
      turnover = sum(|target_notional - current_notional|) / equity
    这里用 qty 与价格计算 current_notional，并按名义差的总和约束。
    """
    if turnover_cap is None:
        return dict(targets), Decimal("0"), ()

    cap = _d(turnover_cap)
    if cap <= 0:
        return dict(targets), Decimal("0"), ("turnover_cap<=0，忽略",)

    # 计算名义变化总量
    total_delta = Decimal("0")
    cur_notional: Dict[str, Decimal] = {}
    for s, t in targets.items():
        px = _d(prices.price(s))
        cq = _d(current_qty.get(s, 0))
        cn = _abs(cq) * px
        cur_notional[s] = cn
        total_delta += _abs(t.target_notional - cn)

    turnover = _safe_div(total_delta, equity)
    if turnover <= cap or total_delta == 0:
        return dict(targets), turnover, ()

    # 缩放名义变化（线性缩放到 cap）
    # 解释：不改变目标方向，只缩小"调仓步长"
    scale = cap / turnover  # < 1

    out: Dict[str, TargetPosition] = {}
    for s, t in targets.items():
        px = _d(prices.price(s))
        cq = _d(current_qty.get(s, 0))

        # 当前 signed notional（用 qty 推断方向；若当前 0，则用目标方向）
        cur_signed_notional = cq * px
        tgt_signed_notional = t.target_qty * px

        # delta_signed = tgt - cur，然后按 scale 缩小
        delta_signed = tgt_signed_notional - cur_signed_notional
        new_signed_notional = cur_signed_notional + delta_signed * scale

        # 得到新的 qty/weight/notional(abs)
        new_qty = _safe_div(new_signed_notional, px)
        new_notional_abs = _abs(new_signed_notional)
        new_weight = _safe_div(new_signed_notional, equity)

        out[s] = TargetPosition(
            symbol=s,
            target_qty=new_qty,
            target_notional=new_notional_abs,
            target_weight=new_weight,
            meta=dict(t.meta) | {"turnover_scaled_by": str(scale)},
        )

    return out, cap, (f"turnover_cap: turnover {turnover} -> {cap}, scaled_by={scale}",)


# ============================================================
# 可选：工具函数（把 AllocationPlan 变成"目标名义/目标权重字典"）
# ============================================================

def plan_to_target_qty(plan: AllocationPlan) -> Dict[str, Decimal]:
    return {t.symbol: t.target_qty for t in plan.targets}


def plan_to_target_weight(plan: AllocationPlan) -> Dict[str, Decimal]:
    return {t.symbol: t.target_weight for t in plan.targets}


def plan_to_target_notional(plan: AllocationPlan) -> Dict[str, Decimal]:
    return {t.symbol: t.target_notional for t in plan.targets}


# --- Rust acceleration ---
try:
    from _quant_hotpath import RustPortfolioAllocator  # noqa: F401
    from _quant_hotpath import (
        cpp_compute_exposures,
        cpp_factor_model_covariance,
        cpp_estimate_specific_risk,
        cpp_black_litterman_posterior,
        rust_allocate_portfolio,
        rust_strategy_weights,
    )
    _RUST_ALLOCATOR_AVAILABLE = True

    # Rust-accelerated portfolio analytics
    compute_exposures = cpp_compute_exposures
    factor_model_covariance = cpp_factor_model_covariance
    estimate_specific_risk = cpp_estimate_specific_risk
    black_litterman_posterior = cpp_black_litterman_posterior
    allocate_portfolio = rust_allocate_portfolio
    strategy_weights = rust_strategy_weights
except ImportError:
    _RUST_ALLOCATOR_AVAILABLE = False


# ============================================================
# Rust-delegated allocation (used by TargetWeightAllocator)
# ============================================================

def _rust_allocate_targets(
    symbols: Sequence[str],
    target_weights: Dict[str, Decimal],
    equity: Decimal,
    prices: PriceProvider,
    current_qty: Mapping[str, Decimal],
    constraints: PortfolioConstraints,
) -> Dict[str, TargetPosition]:
    """Delegate weight->constraint->qty math to rust_allocate_portfolio.

    Returns dict[symbol, TargetPosition] in the same shape as Python path.
    """
    # Convert Decimal inputs to float for Rust FFI
    w_f = {s: float(target_weights.get(s, 0)) for s in symbols}
    px_f = {s: float(prices.price(s)) for s in symbols}
    eq_f = float(equity)
    cq_f = {s: float(current_qty.get(s, 0)) for s in symbols}

    result = rust_allocate_portfolio(
        target_weights=w_f,
        prices=px_f,
        equity=eq_f,
        current_qty=cq_f,
        max_weight=(
            float(constraints.max_weight) if constraints.max_weight is not None else None
        ),
        max_notional_per_symbol=(
            float(constraints.max_notional_per_symbol)
            if constraints.max_notional_per_symbol is not None else None
        ),
        max_gross_leverage=(
            float(constraints.max_gross_leverage)
            if constraints.max_gross_leverage is not None else None
        ),
        turnover_cap=(
            float(constraints.turnover_cap)
            if constraints.turnover_cap is not None else None
        ),
        allow_short=constraints.allow_short,
    )

    # Wrap Rust output back into TargetPosition dataclasses
    targets: Dict[str, TargetPosition] = {}
    for s in symbols:
        if s not in result:
            targets[s] = TargetPosition(
                symbol=s,
                target_qty=Decimal("0"),
                target_notional=Decimal("0"),
                target_weight=Decimal("0"),
            )
            continue
        r = result[s]
        targets[s] = TargetPosition(
            symbol=s,
            target_qty=Decimal(str(r["qty"])),
            target_notional=Decimal(str(r["notional"])),
            target_weight=Decimal(str(r["weight"])),
            meta={"price": str(px_f[s]), "rust_delegated": "true"},
        )
    return targets
