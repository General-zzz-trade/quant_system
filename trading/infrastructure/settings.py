"""Configuration read from the environment. This package is the only place that reads it."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from trading.adapters.binance.adapter import DEFAULT_LEVERAGE, MARGIN_TYPES, MAX_LEVERAGE
from trading.adapters.binance.config import BinanceConfig
from trading.adapters.notify.log import LogNotifier
from trading.adapters.notify.telegram import TelegramNotifier


def testnet_from_env(env: Mapping[str, str] | None = None) -> bool:
    """BINANCE_TESTNET: on unless set to 0, false, no or off."""
    env = os.environ if env is None else env
    return env.get("BINANCE_TESTNET", "1").strip().lower() not in ("0", "false", "no", "off")


def binance_config_from_env(*, testnet: bool, env: Mapping[str, str] | None = None) -> BinanceConfig:
    """API credentials for the chosen environment; missing credentials are a start-up error."""
    env = os.environ if env is None else env
    if testnet:
        key = env.get("BINANCE_TESTNET_API_KEY", "")
        secret = env.get("BINANCE_TESTNET_API_SECRET", "")
    else:
        key = env.get("BINANCE_API_KEY", "")
        secret = env.get("BINANCE_API_SECRET", "")
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


def notifier_from_env(env: Mapping[str, str] | None = None) -> TelegramNotifier | LogNotifier:
    """Telegram when both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set, the process log when neither is.
    One without the other is a start-up error. The values are never echoed."""
    env = os.environ if env is None else env
    token, chat_id = env.get("TELEGRAM_BOT_TOKEN", "").strip(), env.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        return TelegramNotifier(token, chat_id, background=True)
    if token or chat_id:
        raise RuntimeError("set both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or neither")
    return LogNotifier()


def risk_limits(env: Mapping[str, str] | None = None) -> RiskLimits:
    """Drawdown, order and exposure limits of the target runtime, with the defaults it has always had."""
    env = os.environ if env is None else env
    return RiskLimits(daily_drawdown_pct=float(env.get("MAX_DAILY_DRAWDOWN_PCT", "5")),
                      weekly_drawdown_pct=float(env.get("MAX_WEEKLY_DRAWDOWN_PCT", "12")),
                      max_order_notional=Decimal(env.get("BINANCE_MAX_ORDER_NOTIONAL", "5000")),
                      max_daily_notional=Decimal(env.get("BINANCE_MAX_DAILY_NOTIONAL", "50000")),
                      max_daily_orders=int(env.get("BINANCE_MAX_DAILY_ORDERS", "50")),
                      max_leverage=Decimal(env.get("BINANCE_MAX_LEVERAGE", "3")))
