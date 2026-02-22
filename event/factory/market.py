# event/factory/market.py
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from event.header import EventHeader
from event.types import MarketEvent, EventType
from event.errors import EventFatalError


class MarketEventFactory:
    """
    MarketEventFactory —— 市场类事件的唯一合法构造入口

    约束：
    - 必须创建 EventHeader
    - event_type / version 由制度注入
    - 不处理 runtime / schema / store
    """

    @staticmethod
    def bar(
        *,
        ts: datetime,
        symbol: str,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
        source: str = "system.replay",
        run_id: str | None = None,
        correlation_id: str | None = None,
    ) -> MarketEvent:
        # ============================================================
        # 1. 参数合法性校验（Factory 级）
        # ============================================================

        if ts.tzinfo is None:
            raise EventFatalError("ts must be timezone-aware datetime")

        ts_utc = ts.astimezone(timezone.utc)

        if high < low:
            raise EventFatalError("high < low")

        if not (low <= open <= high):
            raise EventFatalError("open out of range")

        if not (low <= close <= high):
            raise EventFatalError("close out of range")

        # ============================================================
        # 2. 构造 EventHeader（制度级）
        # ============================================================

        header = EventHeader.new_root(
            event_type=EventType.MARKET,
            version=MarketEvent.VERSION,
            source=source,
            run_id=run_id,
            correlation_id=correlation_id,
        )

        # ============================================================
        # 3. 构造 MarketEvent（事实本体）
        # ============================================================

        return MarketEvent(
            header=header,
            ts=ts_utc,
            symbol=symbol,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
