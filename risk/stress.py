# quant_system/risk/stress.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


# ============================================================
# 顶级机构级 stress.py（压力测试 / 情景分析）
# ============================================================
# 设计目标：
# 1) stress 是“评估工具”，不是风控实时决策器（不直接下 KILL）
# 2) 既可用于离线研究（研究/回测），也可用于线上监控（定时跑）
# 3) 输入必须是“账户事实快照 + 市场价格”，不依赖 Context/AccountState 的具体实现
# 4) 输出必须可审计（可落库、可复现）
#
# 典型使用方式：
# - 盘中：每 N 秒/分钟对当前仓位做一组 stress 场景，输出风险边界指标给 monitoring
# - 盘前/盘后：跑更重的场景库（多标的联动、极端跳空）并生成报表
#
# 注意：
# - 此文件不做复杂期权 Greeks、VaR/ES（那是 risk_model 或 research 的事）
# - 但它必须提供“足够稳定可复现”的 worst-case 评估框架


# =========================
# 基础工具
# =========================

def _d(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def _abs(x: Decimal) -> Decimal:
    return x if x >= 0 else -x


def _clamp(x: Decimal, lo: Optional[Decimal], hi: Optional[Decimal]) -> Decimal:
    if lo is not None and x < lo:
        return lo
    if hi is not None and x > hi:
        return hi
    return x


# =========================
# 输入结构：账户快照与仓位暴露
# =========================

@dataclass(frozen=True, slots=True)
class PositionExposure:
    """
    单一标的暴露（尽量保持“事实化”）

    qty:
      - 建议用“有符号数量”：多头为正，空头为负
      - 若你的系统内部是 Side + qty，也可在上层 meta_builder 里转换成 signed qty

    entry_price: 建仓均价（可选，用于更精细的 realized/unrealized 拆分；压力测试核心不依赖它）
    mark_price: 现价（必须有，或者通过 market_prices 提供）
    multiplier: 合约乘数（现货=1；合约可用面值/张数乘数）
    """
    symbol: str
    qty: Decimal
    mark_price: Decimal
    multiplier: Decimal = Decimal("1")
    entry_price: Optional[Decimal] = None


@dataclass(frozen=True, slots=True)
class AccountExposure:
    """
    账户暴露快照（Stress 输入）
    """
    equity: Decimal
    balance: Decimal
    used_margin: Decimal = Decimal("0")

    # 费用/融资等（可选：如果你有 funding / fees 的实时估计，可在这里加入）
    extra_liabilities: Decimal = Decimal("0")

    # key 为 normalized symbol
    positions: Mapping[str, PositionExposure] = field(default_factory=dict)


# =========================
# 场景定义：价格冲击
# =========================

@dataclass(frozen=True, slots=True)
class PriceShock:
    """
    价格冲击（两种写法任选其一）：
    - pct: 以百分比冲击，例如 -0.10 表示下跌 10%
    - abs: 以绝对价格改变量冲击，例如 -500 表示价格减 500

    clamp_* 用于限制冲击后的价格区间（可选）
    """
    pct: Optional[Decimal] = None
    abs: Optional[Decimal] = None
    clamp_min_price: Optional[Decimal] = None
    clamp_max_price: Optional[Decimal] = None

    def apply(self, price: Decimal) -> Decimal:
        shocked = price
        if self.pct is not None:
            shocked = shocked * (Decimal("1") + self.pct)
        if self.abs is not None:
            shocked = shocked + self.abs
        # 价格不能为负：给一个硬下限
        shocked = _clamp(shocked, lo=Decimal("0"), hi=None)
        shocked = _clamp(shocked, lo=self.clamp_min_price, hi=self.clamp_max_price)
        return shocked


@dataclass(frozen=True, slots=True)
class StressScenario:
    """
    压力测试场景

    - shocks: 指定 symbol 的冲击
    - global_shock: 未被 shocks 覆盖的标的统一冲击（可选）
    """
    name: str
    shocks: Mapping[str, PriceShock] = field(default_factory=dict)
    global_shock: Optional[PriceShock] = None
    tags: Tuple[str, ...] = ()

    def shocked_price_for(self, symbol: str, price: Decimal) -> Decimal:
        if symbol in self.shocks:
            return self.shocks[symbol].apply(price)
        if self.global_shock is not None:
            return self.global_shock.apply(price)
        return price


# =========================
# 输出结构：结果与判定
# =========================

@dataclass(frozen=True, slots=True)
class StressThresholds:
    """
    压力测试判定阈值（机构级：阈值与场景解耦）
    - max_drawdown_pct: 最大可接受权益跌幅（例如 0.30 = 跌 30%）
    - min_margin_ratio: 最小保证金比率（equity / used_margin）
    """
    max_drawdown_pct: Optional[Decimal] = None
    min_margin_ratio: Optional[Decimal] = None
    min_equity: Optional[Decimal] = None


@dataclass(frozen=True, slots=True)
class StressViolation:
    """
    stress 的违规记录（注意：这不是 RiskDecision；stress 是评估工具）
    """
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StressResult:
    """
    单场景输出（可落库、可复现）
    """
    scenario: str
    equity_before: Decimal
    equity_after: Decimal
    pnl: Decimal
    drawdown_pct: Decimal

    used_margin: Decimal
    margin_ratio: Optional[Decimal]

    # 每个标的在场景下的价格（便于复现与排查）
    shocked_prices: Mapping[str, Decimal] = field(default_factory=dict)

    violations: Tuple[StressViolation, ...] = ()
    tags: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return len(self.violations) == 0


@dataclass(frozen=True, slots=True)
class StressReport:
    """
    一次批量场景运行的报告
    """
    ts: Any
    results: Tuple[StressResult, ...]
    tags: Tuple[str, ...] = ()

    @property
    def worst_by_equity(self) -> StressResult:
        return min(self.results, key=lambda r: r.equity_after)

    @property
    def worst_by_drawdown(self) -> StressResult:
        return max(self.results, key=lambda r: r.drawdown_pct)


# =========================
# Stress 引擎
# =========================

class StressEngineError(RuntimeError):
    pass


class StressEngine:
    """
    顶级机构级 StressEngine（最小可冻结版本）

    核心做法：
    - 对每个场景，构造 shocked_prices
    - 基于仓位（signed qty）计算情景下的 PnL
    - 得到 equity_after，并做阈值判定
    """

    def __init__(
        self,
        *,
        thresholds: Optional[StressThresholds] = None,
    ) -> None:
        self._thresholds = thresholds or StressThresholds()

    def run(
        self,
        *,
        account: AccountExposure,
        scenarios: Sequence[StressScenario],
        ts: Any = None,
        report_tags: Tuple[str, ...] = (),
    ) -> StressReport:
        if not scenarios:
            raise StressEngineError("scenarios 不能为空")

        results: list[StressResult] = []
        for sc in scenarios:
            results.append(self._run_one(account=account, scenario=sc))

        return StressReport(ts=ts, results=tuple(results), tags=report_tags)

    # -------------------------
    # 单场景计算
    # -------------------------

    def _run_one(self, *, account: AccountExposure, scenario: StressScenario) -> StressResult:
        equity_before = _d(account.equity)
        used_margin = _d(account.used_margin)

        shocked_prices: Dict[str, Decimal] = {}
        pnl = Decimal("0")

        # 计算每个仓位在情景下的 PnL
        for sym, pos in account.positions.items():
            cur_px = _d(pos.mark_price)
            new_px = scenario.shocked_price_for(sym, cur_px)
            shocked_prices[sym] = new_px

            # signed qty：多正空负
            qty = _d(pos.qty)
            mult = _d(pos.multiplier)
            # PnL = (new - cur) * qty * multiplier
            pnl += (new_px - cur_px) * qty * mult

        # 额外负债（例如预估 funding/fees）在情景下也要计入（保守）
        liabilities = _d(account.extra_liabilities)

        equity_after = equity_before + pnl - liabilities

        # drawdown_pct：按权益基准
        if equity_before <= 0:
            drawdown_pct = Decimal("0")
        else:
            drawdown_pct = _abs(equity_after - equity_before) / equity_before
            # 只关心“下跌”时的回撤，盈利不算回撤
            if equity_after >= equity_before:
                drawdown_pct = Decimal("0")

        margin_ratio: Optional[Decimal]
        if used_margin > 0:
            margin_ratio = equity_after / used_margin
        else:
            margin_ratio = None

        violations: list[StressViolation] = []
        thr = self._thresholds

        # 阈值判定（机构习惯：阈值缺省不判定）
        if thr.min_equity is not None and equity_after < thr.min_equity:
            violations.append(
                StressViolation(
                    code="min_equity",
                    message="情景下权益低于最小权益阈值",
                    details={"equity_after": str(equity_after), "min_equity": str(thr.min_equity)},
                )
            )

        if thr.max_drawdown_pct is not None and drawdown_pct > thr.max_drawdown_pct:
            violations.append(
                StressViolation(
                    code="max_drawdown",
                    message="情景下回撤超过阈值",
                    details={"drawdown_pct": str(drawdown_pct), "max_drawdown_pct": str(thr.max_drawdown_pct)},
                )
            )

        if thr.min_margin_ratio is not None and margin_ratio is not None and margin_ratio < thr.min_margin_ratio:
            violations.append(
                StressViolation(
                    code="min_margin_ratio",
                    message="情景下保证金比率低于阈值",
                    details={"margin_ratio": str(margin_ratio), "min_margin_ratio": str(thr.min_margin_ratio)},
                )
            )

        return StressResult(
            scenario=scenario.name,
            equity_before=equity_before,
            equity_after=equity_after,
            pnl=pnl,
            drawdown_pct=drawdown_pct,
            used_margin=used_margin,
            margin_ratio=margin_ratio,
            shocked_prices=shocked_prices,
            violations=tuple(violations),
            tags=scenario.tags,
        )


# =========================
# 场景库工厂（顶级机构常用的最小集合）
# =========================

def build_default_stress_scenarios(
    *,
    symbols: Sequence[str],
    base_down_pct: Decimal = Decimal("-0.10"),
    base_up_pct: Decimal = Decimal("0.10"),
    crash_pct: Decimal = Decimal("-0.30"),
    squeeze_pct: Decimal = Decimal("0.30"),
) -> Tuple[StressScenario, ...]:
    """
    默认压力场景库（最小可用）
    - 全市场统一下跌/上涨
    - 单标的 crash / squeeze（其余不变）
    """
    scenarios: list[StressScenario] = []

    scenarios.append(
        StressScenario(
            name=f"global_{int(_abs(base_down_pct) * 100)}pct_down",
            global_shock=PriceShock(pct=base_down_pct),
            tags=("global", "down"),
        )
    )
    scenarios.append(
        StressScenario(
            name=f"global_{int(_abs(base_up_pct) * 100)}pct_up",
            global_shock=PriceShock(pct=base_up_pct),
            tags=("global", "up"),
        )
    )

    for sym in symbols:
        scenarios.append(
            StressScenario(
                name=f"{sym}_crash_{int(_abs(crash_pct) * 100)}pct",
                shocks={sym: PriceShock(pct=crash_pct)},
                tags=("single", "crash"),
            )
        )
        scenarios.append(
            StressScenario(
                name=f"{sym}_squeeze_{int(_abs(squeeze_pct) * 100)}pct",
                shocks={sym: PriceShock(pct=squeeze_pct)},
                tags=("single", "squeeze"),
            )
        )

    return tuple(scenarios)
