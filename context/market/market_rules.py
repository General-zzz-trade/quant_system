# context/market/market_rules.py
"""Market rules — trading session, circuit breakers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class TradingSession:
    """交易时段。"""
    name: str
    start_hour: int     # UTC
    end_hour: int       # UTC
    active: bool = True


@dataclass(frozen=True, slots=True)
class MarketRules:
    """市场规则配置。"""
    venue: str
    symbol: str
    is_24h: bool = True    # 永续合约默认 24 小时交易
    sessions: tuple[TradingSession, ...] = ()
    max_price_change_pct: Optional[float] = None   # 涨跌幅限制
    circuit_breaker_enabled: bool = False

    def is_trading_active(self, hour_utc: int) -> bool:
        if self.is_24h:
            return True
        return any(
            s.active and s.start_hour <= hour_utc < s.end_hour
            for s in self.sessions
        )
