"""Configuration read from the environment. This package is the only place that reads it."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from trading.adapters.binance.adapter import DEFAULT_LEVERAGE, MARGIN_TYPES, MAX_LEVERAGE
from trading.adapters.binance.config import BinanceConfig


def binance_config_from_env(*, testnet: bool) -> BinanceConfig:
    """API credentials for the chosen environment; missing credentials are a start-up error."""
    if testnet:
        key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    else:
        key = os.environ.get("BINANCE_API_KEY", "")
        secret = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError("Binance API key/secret not set in environment")
    return BinanceConfig(api_key=key, api_secret=secret, testnet=testnet)


def exchange_settings(env: Mapping[str, str] | None = None) -> tuple[int, str | None]:
    """(exchange leverage setting, margin type or None) from BINANCE_EXCHANGE_LEVERAGE / BINANCE_MARGIN_TYPE.

    Defaults: leverage 3, margin type left as it is. These change margin per position and
    the liquidation distance only; actual exposure stays capped at 3x equity."""
    env = os.environ if env is None else env
    raw = env.get("BINANCE_EXCHANGE_LEVERAGE", "").strip()
    leverage = int(raw) if raw else DEFAULT_LEVERAGE
    if not 1 <= leverage <= MAX_LEVERAGE:
        raise RuntimeError(f"BINANCE_EXCHANGE_LEVERAGE must be in [1, {MAX_LEVERAGE}]")
    margin = env.get("BINANCE_MARGIN_TYPE", "").strip().upper() or None
    if margin is not None and margin not in MARGIN_TYPES:
        raise RuntimeError(f"BINANCE_MARGIN_TYPE must be one of {sorted(MARGIN_TYPES)}")
    return leverage, margin


def execution_scope(*, testnet: bool, env: Mapping[str, str] | None = None) -> str:
    """RISK_SCOPE for an execution run: it must name binance, and say testnet exactly on the testnet."""
    env = os.environ if env is None else env
    scope = env.get("RISK_SCOPE", "").strip()
    if not scope or "binance" not in scope or (("testnet" in scope) != testnet):
        raise RuntimeError("RISK_SCOPE must name binance and contain 'testnet' exactly when BINANCE_TESTNET=1")
    return scope


@dataclass(frozen=True)
class RiskLimits:
    """Limits a person sets in the environment; nothing in the code raises them."""

    daily_drawdown_pct: float
    weekly_drawdown_pct: float
    max_order_notional: Decimal
    max_daily_notional: Decimal
    max_daily_orders: int
    max_leverage: Decimal


def risk_limits(env: Mapping[str, str] | None = None) -> RiskLimits:
    """Drawdown, order and exposure limits of the target runtime, with the defaults it has always had."""
    env = os.environ if env is None else env
    return RiskLimits(daily_drawdown_pct=float(env.get("MAX_DAILY_DRAWDOWN_PCT", "5")),
                      weekly_drawdown_pct=float(env.get("MAX_WEEKLY_DRAWDOWN_PCT", "12")),
                      max_order_notional=Decimal(env.get("BINANCE_MAX_ORDER_NOTIONAL", "5000")),
                      max_daily_notional=Decimal(env.get("BINANCE_MAX_DAILY_NOTIONAL", "50000")),
                      max_daily_orders=int(env.get("BINANCE_MAX_DAILY_ORDERS", "50")),
                      max_leverage=Decimal(env.get("BINANCE_MAX_LEVERAGE", "3")))
