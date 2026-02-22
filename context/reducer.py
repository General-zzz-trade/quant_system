# quant_system/context/reducer.py
from __future__ import annotations

from typing import Any

from event.types import BaseEvent, EventType, MarketEvent
from context.context import Context, ContextError


def reduce_context(ctx: Context, event: BaseEvent) -> None:
    """
    顶级机构级 Reducer（Context 状态演化规则）

    设计原则：
    1) Reducer 是唯一允许修改 Context 内部状态的地方
    2) Reducer 必须是确定性的（同事件序列 -> 同状态）
    3) Reducer 内禁止 IO / 网络 / logging
    4) Reducer 失败 = 状态非法，必须抛异常（fail-fast）
    """

    # ============================================================
    # 0. 只接受 Context 认可的事件类型
    # ============================================================

    if event.type != EventType.MARKET:
        # 非 MARKET 事件：Context 选择忽略
        ctx._ignored_events += 1  # noqa: SLF001（Reducer 是 Context 的特权模块）
        return

    if not isinstance(event, MarketEvent):
        raise ContextError(
            "EventType=MARKET 但事件对象不是 MarketEvent，事件定义不一致"
        )

    # ============================================================
    # 1. 推进 Clock（世界时间演化）
    # ============================================================

    prev_ts = ctx._clock.ts
    prev_bar_index = ctx._clock.bar_index

    # 1.1 提取事件时间
    et = event.header.event_time
    if et is None and getattr(event, "bar", None) is not None:
        et = event.bar.ts

    if et is None:
        raise ContextError(
            "MarketEvent 缺少 event_time（header.event_time 或 bar.ts）"
        )

    # 1.2 判断是否是 bar 事件（兼容 enum / str）
    kind = getattr(event, "kind", None)
    kind_value = getattr(kind, "value", kind)
    is_bar = (kind_value == "bar")

    # 1.3 更新 Clock
    ctx._clock.update_from_event_time(et, is_bar=is_bar)

    # ============================================================
    # 2. 顶级机构安全防线：时间不可倒流
    # ============================================================

    if prev_ts is not None and ctx._clock.ts is not None:
        try:
            if ctx._clock.ts < prev_ts:
                raise ContextError(
                    f"Clock 时间倒流：prev_ts={prev_ts} -> new_ts={ctx._clock.ts}"
                )
        except TypeError:
            raise ContextError(
                f"Clock ts 不可比较：{type(prev_ts)} vs {type(ctx._clock.ts)}"
            )

    if ctx._clock.bar_index < prev_bar_index:
        raise ContextError(
            f"bar_index 回退：prev={prev_bar_index} -> new={ctx._clock.bar_index}"
        )

    # ============================================================
    # 3. 写入 MarketState（外部世界映射）
    # ============================================================

    ctx._market.update_from_market_event(
        event,
        ts=ctx._clock.ts,
        bar_index=ctx._clock.bar_index,
    )

    # ============================================================
    # 4. 未来扩展点（占位，不实现）
    # ============================================================

    # - account_state 更新
    # - portfolio_state 更新
    # - execution_state 更新
    # 这些都应该逐步加在 reducer 里，而不是 Context
