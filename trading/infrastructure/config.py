"""The configuration, read from the environment once at start-up (ADR-0017). This is the only module in trading/
and runner/ that reads the environment (tests/architecture/test_config_rules.py).

``load_config`` returns a frozen ``RuntimeConfig`` - every value the runtime, the composition root and the tools
use, and the account identity (ADR-0016 point 5) - and, apart, the ``Secrets`` only the adapters are built with.
The config is recorded at start-up without any secret; its digest is in the status file. The parsers below were
moved here from settings.py, which re-exports them.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal

from trading.adapters.binance.adapter import DEFAULT_LEVERAGE, MARGIN_TYPES, MAX_LEVERAGE
from trading.adapters.binance.config import BinanceConfig
from trading.adapters.notify.heartbeat import HttpHeartbeat
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
    max_order_notional: Decimal | None      # None: no single-order cap (paper, decided 2026-11-16)
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


# Hard ceilings of the drawdown limits (tech lead, 2026-11-15): a person may set less, never more.
DRAWDOWN_CEILING_PCT = {"daily": 10.0, "weekly": 25.0}


def risk_limits(env: Mapping[str, str] | None = None) -> RiskLimits:
    """Drawdown, order and exposure limits of the target runtime, with the defaults it has always had. A
    drawdown limit outside (0, ceiling] is a start-up error."""
    env = os.environ if env is None else env
    limits = RiskLimits(daily_drawdown_pct=float(env.get("MAX_DAILY_DRAWDOWN_PCT", "5")),
                      weekly_drawdown_pct=float(env.get("MAX_WEEKLY_DRAWDOWN_PCT", "12")),
                      max_order_notional=Decimal(env.get("BINANCE_MAX_ORDER_NOTIONAL", "5000")),
                      max_daily_notional=Decimal(env.get("BINANCE_MAX_DAILY_NOTIONAL", "50000")),
                      max_daily_orders=int(env.get("BINANCE_MAX_DAILY_ORDERS", "50")),
                      max_leverage=Decimal(env.get("BINANCE_MAX_LEVERAGE", "3")))
    for kind, value in (("daily", limits.daily_drawdown_pct), ("weekly", limits.weekly_drawdown_pct)):
        if not math.isfinite(value) or not 0 < value <= DRAWDOWN_CEILING_PCT[kind]:
            raise RuntimeError(f"{kind} drawdown limit {value}% is outside (0, {DRAWDOWN_CEILING_PCT[kind]}%]")
    return limits


@dataclass(frozen=True, repr=False)
class Secrets:
    """API credentials and the Telegram token: only for building adapters, never recorded or printed."""

    binance: BinanceConfig | None
    telegram: tuple[str, str] | None
    heartbeat_url: str | None = None   # the external heartbeat's address (ADR-0016 point 3): a secret too

    def __repr__(self) -> str:
        return "Secrets(<hidden>)"


@dataclass(frozen=True)
class RuntimeConfig:
    testnet: bool
    execution: bool
    scope: str
    stop_pct: Decimal
    limits: RiskLimits | None          # execution only; paper uses the code defaults (S10 decision)
    leverage: int | None               # the exchange's leverage setting the start-up check expects
    margin: str | None                 # the margin mode it expects, or None for "as the account has it"
    alerts: str | None                 # "telegram" or "log" in execution mode
    identity: str | None               # sha256(network:API key) when credentials are loaded

    def record(self) -> dict[str, object]:
        """Every value, as plain JSON, without any secret: what this run uses."""
        limits = None if self.limits is None else {k: str(v) for k, v in self.limits.__dict__.items()}
        return {"testnet": self.testnet, "execution": self.execution, "scope": self.scope,
                "stop_pct": str(self.stop_pct), "limits": limits, "leverage": self.leverage, "margin": self.margin,
                "alerts": self.alerts, "identity": self.identity}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True).encode()).hexdigest()


def account_identity(*, testnet: bool, api_key: str) -> str:
    """ADR-0016 point 5: sha256(network + ":" + API key); the key only, never the secret."""
    return hashlib.sha256(f"{'testnet' if testnet else 'mainnet'}:{api_key}".encode()).hexdigest()


def load_config(env: Mapping[str, str] | None = None, *, execution: bool, stop_pct: Decimal,
                testnet: bool | None = None, require_credentials: bool = False) -> tuple[RuntimeConfig, Secrets]:
    """Read and check every setting once. A missing or invalid one refuses the start and names it. ``testnet``
    overrides BINANCE_TESTNET (tests); credentials are read only when ``require_credentials``."""
    env = os.environ if env is None else env
    net = testnet_from_env(env) if testnet is None else testnet
    if not execution:
        return (RuntimeConfig(testnet=net, execution=False, scope="paper-target", stop_pct=Decimal(stop_pct),
                              limits=None, leverage=None, margin=None, alerts=None, identity=None),
                Secrets(binance=None, telegram=None))
    scope = execution_scope(testnet=net, env=env)
    limits = risk_limits(env)
    leverage, margin = exchange_settings(env)
    token, chat_id = env.get("TELEGRAM_BOT_TOKEN", "").strip(), env.get("TELEGRAM_CHAT_ID", "").strip()
    if bool(token) != bool(chat_id):
        raise RuntimeError("set both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or neither")
    heartbeat = env.get("HEARTBEAT_URL", "").strip() or None
    if heartbeat is not None and not heartbeat.startswith("https://"):
        raise RuntimeError("HEARTBEAT_URL must be an https:// address")       # never echoed: it is a secret
    binance = binance_config_from_env(testnet=net, env=env) if require_credentials else None
    identity = account_identity(testnet=net, api_key=binance.api_key) if binance is not None else None
    config = RuntimeConfig(testnet=net, execution=True, scope=scope, stop_pct=Decimal(stop_pct), limits=limits,
                           leverage=leverage, margin=margin, alerts="telegram" if token else "log", identity=identity)
    return config, Secrets(binance=binance, telegram=(token, chat_id) if token else None, heartbeat_url=heartbeat)


def require_operator_channels(config: RuntimeConfig, secrets: Secrets) -> None:
    """ADR-0016 points 1 and 3: execution mode (testnet and mainnet) refuses to start without Telegram or without a
    heartbeat address. Paper needs neither."""
    if not config.execution:
        return
    if secrets.telegram is None:
        raise RuntimeError("execution mode needs alerts: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    if not secrets.heartbeat_url:
        raise RuntimeError("execution mode needs an external heartbeat: set HEARTBEAT_URL")


def make_heartbeat(config: RuntimeConfig, secrets: Secrets) -> HttpHeartbeat | None:
    """The external heartbeat of an execution run; None in paper mode."""
    if not config.execution or not secrets.heartbeat_url:
        return None
    return HttpHeartbeat(secrets.heartbeat_url)


def with_credentials(config: RuntimeConfig, secrets: Secrets,
                     env: Mapping[str, str] | None = None) -> tuple[RuntimeConfig, Secrets]:
    """The Binance credentials, read after every local start-up check and before the account is connected (the
    order the entry point has always had), and the account identity derived from them."""
    env = os.environ if env is None else env
    binance = binance_config_from_env(testnet=config.testnet, env=env)
    return (replace(config, identity=account_identity(testnet=config.testnet, api_key=binance.api_key)),
            replace(secrets, binance=binance))


def make_notifier(secrets: Secrets) -> TelegramNotifier | LogNotifier:
    """Telegram when both its values were set, the process log otherwise (load_config refused one alone)."""
    if secrets.telegram is not None:
        return TelegramNotifier(*secrets.telegram, background=True)
    return LogNotifier()
