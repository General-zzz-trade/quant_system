# execution/models/instruments.py
"""Canonical instrument (symbol) metadata."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence


@dataclass(frozen=True, slots=True)
class InstrumentInfo:
    """
    交易对/合约的标准化静态元数据。

    来自交易所的 exchangeInfo，在启动时缓存。
    用于下单前精度校验和数量/价格圆整。
    """
    venue: str
    symbol: str

    base_asset: str
    quote_asset: str

    # 精度
    price_precision: int = 8
    qty_precision: int = 8

    # 步长
    tick_size: Decimal = Decimal("0.01")
    lot_size: Decimal = Decimal("0.001")

    # 限制
    min_qty: Decimal = Decimal("0")
    max_qty: Optional[Decimal] = None
    min_notional: Decimal = Decimal("0")

    # 合约信息（仅永续/期货）
    contract_type: Optional[str] = None  # "perpetual" / "quarterly" / None(现货)
    margin_asset: Optional[str] = None

    # 状态
    trading_enabled: bool = True

    def round_price(self, price: Decimal) -> Decimal:
        """按 tick_size 圆整价格（向下）。"""
        if self.tick_size <= 0:
            return price
        return (price // self.tick_size) * self.tick_size

    def round_qty(self, qty: Decimal) -> Decimal:
        """按 lot_size 圆整数量（向下）。"""
        if self.lot_size <= 0:
            return qty
        return (qty // self.lot_size) * self.lot_size

    def validate_qty(self, qty: Decimal) -> tuple[bool, str]:
        """检查数量是否合法。"""
        if qty < self.min_qty:
            return False, f"qty {qty} < min_qty {self.min_qty}"
        if self.max_qty is not None and qty > self.max_qty:
            return False, f"qty {qty} > max_qty {self.max_qty}"
        return True, ""

    def validate_notional(self, qty: Decimal, price: Decimal) -> tuple[bool, str]:
        """检查名义价值是否达到最低要求。"""
        notional = qty * price
        if notional < self.min_notional:
            return False, f"notional {notional} < min_notional {self.min_notional}"
        return True, ""


@dataclass(frozen=True, slots=True)
class InstrumentRegistry:
    """交易对注册表 — 保存所有可交易品种的元数据。"""
    _instruments: tuple[InstrumentInfo, ...] = ()

    def get(self, venue: str, symbol: str) -> Optional[InstrumentInfo]:
        v = venue.lower()
        s = symbol.upper()
        for inst in self._instruments:
            if inst.venue == v and inst.symbol == s:
                return inst
        return None

    def symbols(self, venue: Optional[str] = None) -> Sequence[str]:
        if venue is None:
            return [i.symbol for i in self._instruments]
        v = venue.lower()
        return [i.symbol for i in self._instruments if i.venue == v]

    def add(self, inst: InstrumentInfo) -> InstrumentRegistry:
        existing = [i for i in self._instruments
                    if not (i.venue == inst.venue and i.symbol == inst.symbol)]
        return InstrumentRegistry(_instruments=tuple(existing) + (inst,))


# 兼容别名
Instrument = InstrumentInfo
