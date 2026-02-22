# quant_system/context/account_state.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Mapping, Optional, Tuple

from event.types import (
    Symbol,
    Venue,
    Side,
    FillEvent,
)

# ============================================================
# 只读快照（对外）
# ============================================================

@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """
    单一标的的只读仓位快照（值语义）
    """
    symbol: Symbol
    venue: Venue
    side: Side

    qty: Decimal
    avg_price: Optional[Decimal]

    realized_pnl: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """
    账户状态只读快照（给 risk / strategy / monitoring 使用）
    """
    equity: Decimal                 # 账户权益
    balance: Decimal                # 账户余额（可用现金）
    used_margin: Decimal            # 已用保证金（v1 固定为 0）
    free_margin: Decimal            # 可用保证金（= equity - used_margin）

    realized_pnl: Decimal           # 已实现盈亏
    unrealized_pnl: Decimal         # 未实现盈亏

    positions: Mapping[Tuple[Venue, str], PositionSnapshot]


# ============================================================
# 内部可变记录（仅 Context / Reducer 可写）
# ============================================================

@dataclass(slots=True)
class _PositionRecord:
    symbol: Symbol
    venue: Venue
    side: Side

    qty: Decimal = Decimal("0")
    avg_price: Optional[Decimal] = None

    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")


@dataclass(slots=True)
class _AccountRecord:
    balance: Decimal
    equity: Decimal

    used_margin: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # ⚠️ 机构级要求：可变字段必须用 default_factory
    positions: Dict[Tuple[Venue, str], _PositionRecord] = field(default_factory=dict)


# ============================================================
# AccountState 主体
# ============================================================

class AccountStateError(RuntimeError):
    pass


class AccountState:
    """
    顶级机构级 AccountState（v1.0 冻结版）

    核心原则：
    1) 只记录“账户事实”，不做风控判断
    2) 不做执行决策，不依赖 Strategy
    3) 所有写操作只能来自 Reducer
    4) 不允许产生“物理不可能”的账户状态
    """

    def __init__(
        self,
        *,
        initial_balance: Decimal,
    ) -> None:
        if not isinstance(initial_balance, Decimal):
            raise AccountStateError("initial_balance 必须是 Decimal")
        if initial_balance <= Decimal("0"):
            raise AccountStateError("initial_balance 必须大于 0")

        self._record = _AccountRecord(
            balance=initial_balance,
            equity=initial_balance,
        )

    # ------------------------------------------------------------
    # 核心更新入口（仅 Reducer 调用）
    # ------------------------------------------------------------

    def apply_fill(self, fill: FillEvent) -> None:
        """
        用真实成交（FillEvent）更新账户状态（v1.0）

        v1.0 约束（刻意严格）：
        - 只支持同向成交（加仓）
        - 反向成交 / 平仓 → 直接抛异常（fail-fast）
        - 不支持手续费 / 杠杆 / 保证金模型
        """
        key = (fill.venue, fill.symbol.normalized)
        pos = self._record.positions.get(key)

        # 成交数量与价格（Decimal）
        qty = fill.qty.value
        price = fill.price.value
        notional = qty * price

        if qty <= Decimal("0") or price <= Decimal("0"):
            raise AccountStateError("成交数量或价格非法（<= 0）")

        # 余额防线：AccountState 不允许负余额
        if self._record.balance < notional:
            raise AccountStateError(
                f"余额不足：balance={self._record.balance}, notional={notional}"
            )

        if pos is None:
            # 新建仓位
            pos = _PositionRecord(
                symbol=fill.symbol,
                venue=fill.venue,
                side=fill.side,
            )
            self._record.positions[key] = pos

        # v1.0 明确拒绝反向成交 / 平仓
        if pos.qty != Decimal("0") and fill.side != pos.side:
            raise AccountStateError(
                "v1.0 AccountState 不支持反向成交 / 平仓（请在 v2 实现）"
            )

        # 同向加仓 / 首次建仓
        if pos.qty == Decimal("0"):
            pos.avg_price = price
            pos.qty = qty
        else:
            # 加权平均价
            pos.avg_price = (
                (pos.avg_price * pos.qty + price * qty) / (pos.qty + qty)
            )
            pos.qty += qty

        # 扣减余额（逐仓、无手续费假设）
        self._record.balance -= notional

        # 账户权益更新（未实现盈亏稍后由 mark_to_market 更新）
        self._record.equity = self._record.balance + self._record.unrealized_pnl

    # ------------------------------------------------------------
    # 未实现盈亏刷新（仅 Reducer 调用）
    # ------------------------------------------------------------

    def mark_to_market(
        self,
        *,
        symbol: Symbol,
        venue: Venue,
        price: Decimal,
    ) -> None:
        """
        按最新市价刷新未实现盈亏（v1.0）

        约束：
        - 只能由 Reducer 调用
        - price 必须来自 MarketSnapshot
        """
        if price <= Decimal("0"):
            raise AccountStateError("mark_to_market 价格非法（<= 0）")

        key = (venue, symbol.normalized)
        pos = self._record.positions.get(key)
        if pos is None or pos.avg_price is None or pos.qty == Decimal("0"):
            return

        if pos.side == Side.BUY:
            pos.unrealized_pnl = (price - pos.avg_price) * pos.qty
        else:
            pos.unrealized_pnl = (pos.avg_price - price) * pos.qty

        # 汇总账户层面未实现盈亏
        self._record.unrealized_pnl = sum(
            p.unrealized_pnl for p in self._record.positions.values()
        )

        # 更新权益
        self._record.equity = self._record.balance + self._record.unrealized_pnl

    # ------------------------------------------------------------
    # 只读快照
    # ------------------------------------------------------------

    def snapshot(self) -> AccountSnapshot:
        free_margin = self._record.equity - self._record.used_margin

        positions = {
            k: PositionSnapshot(
                symbol=v.symbol,
                venue=v.venue,
                side=v.side,
                qty=v.qty,
                avg_price=v.avg_price,
                realized_pnl=v.realized_pnl,
                unrealized_pnl=v.unrealized_pnl,
            )
            for k, v in self._record.positions.items()
        }

        return AccountSnapshot(
            equity=self._record.equity,
            balance=self._record.balance,
            used_margin=self._record.used_margin,
            free_margin=free_margin,
            realized_pnl=self._record.realized_pnl,
            unrealized_pnl=self._record.unrealized_pnl,
            positions=positions,
        )
