# execution/config/reconcile_config.py
"""Reconciliation configuration."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class ReconcileConfig:
    """
    对账配置。

    控制对账行为：容差、频率、自动修复等。
    """
    # 对账间隔
    interval_sec: float = 60.0

    # 数量容差（绝对值）
    qty_tolerance: Decimal = Decimal("0.00001")

    # 价格容差（百分比，如 0.001 = 0.1%）
    price_tolerance_pct: Decimal = Decimal("0.001")

    # 是否自动修复漂移
    auto_fix_info: bool = False        # INFO 级自动修复
    auto_fix_warning: bool = False     # WARNING 级自动修复
    auto_fix_critical: bool = False    # CRITICAL 级不自动修复（需人工）

    # 对账失败时是否暂停交易
    halt_on_critical: bool = True

    # 最大连续失败次数（超过后触发告警）
    max_consecutive_failures: int = 3

    # 对账超时
    timeout_sec: float = 30.0
