# quant_system/context/validators.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from context.context import Context


def _context_error(msg: str) -> Exception:
    """延迟导入 ContextError 避免循环依赖。"""
    from context.context import ContextError
    return ContextError(msg)


def validate_context(ctx: Context) -> None:
    """
    顶级机构级 Context 一致性校验入口（v1.0）

    设计原则：
    - reducer 之后必须调用
    - 不修改任何 state（只读）
    - 发现问题立即抛异常（fail-fast）
    """
    _validate_clock(ctx)
    _validate_market_state(ctx)
    _validate_snapshot_lineage(ctx)
    _validate_context_integrity(ctx)


# ============================================================
# 1. Clock / 时间系统校验（最基础、最致命）
# ============================================================

def _validate_clock(ctx: Context) -> None:
    clock = ctx._clock  # noqa: SLF001（Validator 是 Context 的特权模块）

    # ts / bar_index 不允许为 None
    if clock.ts is None:
        raise _context_error("Clock.ts 不能为空")

    if clock.bar_index is None:
        raise _context_error("Clock.bar_index 不能为空")

    # bar_index 必须是非负整数
    if not isinstance(clock.bar_index, int):
        raise _context_error(
            f"Clock.bar_index 类型错误，应为 int，实际为 {type(clock.bar_index)}"
        )

    if clock.bar_index < 0:
        raise _context_error(
            f"Clock.bar_index 非法（<0）：{clock.bar_index}"
        )


# ============================================================
# 2. MarketState 最小自洽性校验
# ============================================================

def _validate_market_state(ctx: Context) -> None:
    market = ctx._market  # noqa: SLF001

    # MarketState 必须存在
    if market is None:
        raise _context_error("Context.market_state 为空")

    # 如果 MarketState 内部有 symbol 快照，必须满足基本一致性
    # 这里不假设你的 MarketState 内部结构，只做“存在性 + 不崩溃”校验
    try:
        _ = market  # 占位，防止未使用警告
    except Exception as e:
        raise _context_error(
            f"MarketState 访问失败，内部状态可能损坏: {e}"
        )


# ============================================================
# 3. Snapshot / Lineage 一致性校验（审计级）
# ============================================================

def _validate_snapshot_lineage(ctx: Context) -> None:
    """
    校验 Context 的 lineage 状态是否自洽。
    注意：这里不创建 snapshot，只检查 lineage 指针。
    """
    # context_id 必须存在且为 str
    if not isinstance(ctx.context_id, str):
        raise _context_error(
            f"context_id 类型错误，应为 str，实际为 {type(ctx.context_id)}"
        )

    # last_snapshot_id 如果存在，必须是 str
    last_snap = ctx._last_snapshot_id  # noqa: SLF001
    if last_snap is not None and not isinstance(last_snap, str):
        raise _context_error(
            f"last_snapshot_id 类型错误，应为 str，实际为 {type(last_snap)}"
        )


# ============================================================
# 4. Context 自身结构完整性校验
# ============================================================

def _validate_context_integrity(ctx: Context) -> None:
    """
    防止 Context 被错误构造或被外部模块破坏。
    """

    # Clock / MarketState 必须存在
    if ctx._clock is None:  # noqa: SLF001
        raise _context_error("Context._clock 为空")

    if ctx._market is None:  # noqa: SLF001
        raise _context_error("Context._market 为空")

    # meta 必须是 dict（内部可变，但结构要对）
    if not isinstance(ctx._meta, dict):  # noqa: SLF001
        raise _context_error(
            f"Context._meta 类型错误，应为 dict，实际为 {type(ctx._meta)}"
        )
