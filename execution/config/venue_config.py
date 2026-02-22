# execution/config/venue_config.py
"""Per-venue configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class VenueConfig:
    """
    单个交易所的运行时配置。

    包含连接、限速、安全参数等。
    """
    name: str                           # "binance"
    enabled: bool = True

    # 连接
    rest_url: str = ""
    ws_url: str = ""
    testnet: bool = False

    # 认证（密钥从环境变量/密钥管理器加载，这里只存 key 名称）
    api_key_env: str = ""
    api_secret_env: str = ""

    # 限速
    rest_rate_per_sec: float = 10.0
    ws_rate_per_sec: float = 5.0
    order_rate_per_sec: float = 10.0
    burst: float = 20.0

    # 可交易品种白名单（空=全部允许）
    symbols: tuple[str, ...] = ()

    # 特性开关
    reduce_only_supported: bool = True
    post_only_supported: bool = True
    hedge_mode: bool = False

    # 超时
    connect_timeout_sec: float = 10.0
    read_timeout_sec: float = 5.0

    def is_symbol_allowed(self, symbol: str) -> bool:
        if not self.symbols:
            return True
        return symbol.upper() in self.symbols
