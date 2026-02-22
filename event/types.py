from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar, Dict, Mapping, Type
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timezone
# ============================================================
# EventType —— 事件类型枚举（制度真理源）
# ============================================================

class EventType(Enum):
    MARKET = "market"
    SIGNAL = "signal"
    INTENT = "intent"
    ORDER = "order"
    FILL = "fill"
    RISK = "risk"
    CONTROL = "control"



# ============================================================
# BaseEvent —— 所有事件的抽象基类
# ============================================================

@dataclass(frozen=True, slots=True)
class BaseEvent(ABC):
    """
    BaseEvent —— 事件事实的唯一抽象

    约束：
    - header 必须存在
    - version 由事件类 VERSION 决定
    """

    event_type: ClassVar[EventType]
    header: Any

    # schema / runtime 使用的制度版本
    VERSION: ClassVar[int] = 1

    @property
    def version(self) -> int:
        return int(self.__class__.VERSION)

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        将事件事实转换为 schema 校验用的 body
        """
        ...

    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        *,
        header: Any,
        body: Mapping[str, Any],
    ) -> "BaseEvent":
        """
        从持久化 / replay 数据恢复事件事实
        """
        ...


# ============================================================
# MarketEvent —— 市场行情事件
# ============================================================

@dataclass(frozen=True, slots=True)
class MarketEvent(BaseEvent):
    """
    MarketEvent —— K 线 / 行情类事件
    """

    event_type: ClassVar[EventType] = EventType.MARKET
    VERSION: ClassVar[int] = 1

    # 业务时间（K 线时间）
    ts: datetime

    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(
        cls,
        *,
        header: Any,
        body: Mapping[str, Any],
    ) -> "MarketEvent":
        raw_ts = body["ts"]
        if isinstance(raw_ts, str):
            s = raw_ts.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            try:
                ts = datetime.fromisoformat(s)
            except ValueError as e:
                raise ValueError(f"invalid ts isoformat: {raw_ts}") from e
        elif isinstance(raw_ts, datetime):
            ts = raw_ts
        else:
            raise ValueError(f"invalid ts type: {type(raw_ts).__name__}")

        if ts.tzinfo is None:
            raise ValueError("ts must be tz-aware")
        ts = ts.astimezone(timezone.utc)

        return cls(
            header=header,
            ts=ts,
            symbol=str(body["symbol"]),
            open=Decimal(body["open"]),
            high=Decimal(body["high"]),
            low=Decimal(body["low"]),
            close=Decimal(body["close"]),
            volume=Decimal(body["volume"]),
        )


# ============================================================
# SignalEvent —— 策略信号（可被拒绝）
# ============================================================

@dataclass(frozen=True, slots=True)
class SignalEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.SIGNAL

    signal_id: str
    symbol: str
    side: str  # "long" | "short"
    strength: Decimal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side,
            "strength": self.strength,
        }

    @classmethod
    def from_dict(cls, *, header: Any, body: Mapping[str, Any]) -> SignalEvent:
        return cls(
            header=header,
            signal_id=str(body["signal_id"]),
            symbol=str(body["symbol"]),
            side=str(body["side"]),
            strength=Decimal(body["strength"]),
        )


# ============================================================
# IntentEvent —— 交易意图（制度级）
# ============================================================

@dataclass(frozen=True, slots=True)
class IntentEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.INTENT

    intent_id: str
    symbol: str
    side: str          # "buy" | "sell"
    target_qty: Decimal
    reason_code: str   # signal | rebalance | risk | manual
    origin: str        # strategy_id / model_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "side": self.side,
            "target_qty": self.target_qty,
            "reason_code": self.reason_code,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, *, header: Any, body: Mapping[str, Any]) -> IntentEvent:
        return cls(
            header=header,
            intent_id=str(body["intent_id"]),
            symbol=str(body["symbol"]),
            side=str(body["side"]),
            target_qty=Decimal(body["target_qty"]),
            reason_code=str(body["reason_code"]),
            origin=str(body["origin"]),
        )


# ============================================================
# OrderEvent —— 执行指令
# ============================================================

@dataclass(frozen=True, slots=True)
class OrderEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.ORDER

    order_id: str
    intent_id: str
    symbol: str
    side: str
    qty: Decimal
    price: Decimal | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
        }

    @classmethod
    def from_dict(cls, *, header: Any, body: Mapping[str, Any]) -> OrderEvent:
        price_raw = body.get("price")
        return cls(
            header=header,
            order_id=str(body["order_id"]),
            intent_id=str(body["intent_id"]),
            symbol=str(body["symbol"]),
            side=str(body["side"]),
            qty=Decimal(body["qty"]),
            price=None if price_raw is None else Decimal(price_raw),
        )


# ============================================================
# FillEvent —— 成交事实
# ============================================================

@dataclass(frozen=True, slots=True)
class FillEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.FILL

    fill_id: str
    order_id: str
    symbol: str
    qty: Decimal
    price: Decimal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "qty": self.qty,
            "price": self.price,
        }

    @classmethod
    def from_dict(cls, *, header: Any, body: Mapping[str, Any]) -> FillEvent:
        return cls(
            header=header,
            fill_id=str(body["fill_id"]),
            order_id=str(body["order_id"]),
            symbol=str(body["symbol"]),
            qty=Decimal(body["qty"]),
            price=Decimal(body["price"]),
        )


# ============================================================
# RiskEvent —— 风控裁决（制度事件）
# ============================================================

@dataclass(frozen=True, slots=True)
class RiskEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.RISK

    rule_id: str
    level: str          # info | warn | block
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "level": self.level,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, *, header: Any, body: Mapping[str, Any]) -> RiskEvent:
        return cls(
            header=header,
            rule_id=str(body["rule_id"]),
            level=str(body["level"]),
            message=str(body["message"]),
        )


# ============================================================
# ControlEvent —— 系统控制（制度级）
# ============================================================

@dataclass(frozen=True, slots=True)
class ControlEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.CONTROL

    command: str    # halt / resume / flush / shutdown
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, *, header: Any, body: Mapping[str, Any]) -> ControlEvent:
        return cls(
            header=header,
            command=str(body["command"]),
            reason=str(body["reason"]),
        )


# ============================================================
# 冻结版说明
# ============================================================
# - 本文件为 event 层“制度真理源”
# - 不允许在 codec / runtime / reducer 中兜底字段
# - 若 IDE 出现红线，只能修改本文件
# - 新增事件类型 = 新制度版本（不可偷偷加字段）
