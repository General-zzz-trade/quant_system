"""The configuration, read from the environment once at start-up (ADR-0017). This is the only module in trading/
and runner/ that reads the environment (tests/architecture/test_config_rules.py).

``load_config`` returns a frozen ``RuntimeConfig`` - every value the runtime, the composition root and the tools
use, and the account identity (ADR-0016 point 5) - and, apart, the ``Secrets`` only the adapters are built with.
The config is recorded at start-up without any secret; its digest is in the status file. The parsers below were
moved here from settings.py (S4); that re-export shell was deleted on 2026-11-16.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path

from trading.adapters.binance.adapter import DEFAULT_LEVERAGE, MARGIN_TYPES, MAX_LEVERAGE
from trading.adapters.binance.config import BinanceConfig
from trading.adapters.notify.heartbeat import HttpHeartbeat
from trading.adapters.notify.log import LogNotifier
from trading.adapters.notify.file import FileNotifier
from trading.application.execution.stops import MAX_STOP_PCT, MIN_STOP_PCT
from trading.domain.risk.exposure import MAX_TOTAL_LEVERAGE
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


def alert_file(env: Mapping[str, str], *, telegram: bool) -> Path | None:
    """ALERT_FILE (owner 2026-11-16): the file alerts are appended to in place of Telegram. An absolute path; with
    the Telegram pair set as well the start is refused: one channel, so it is clear who is told."""
    raw = env.get("ALERT_FILE", "").strip()
    if not raw:
        return None
    if telegram:
        raise RuntimeError("set ALERT_FILE or the Telegram pair (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID), not both")
    if not Path(raw).is_absolute():
        raise RuntimeError("ALERT_FILE must be an absolute path")
    return Path(raw)


def notifier_from_env(env: Mapping[str, str] | None = None) -> TelegramNotifier | FileNotifier | LogNotifier:
    """Telegram when both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set, the alert file when ALERT_FILE is, the
    process log when none is. Half a Telegram pair, or both channels, is a start-up error. Values are never echoed."""
    env = os.environ if env is None else env
    token, chat_id = env.get("TELEGRAM_BOT_TOKEN", "").strip(), env.get("TELEGRAM_CHAT_ID", "").strip()
    file = alert_file(env, telegram=bool(token or chat_id))
    if token and chat_id:
        return TelegramNotifier(token, chat_id, background=True)
    if token or chat_id:
        raise RuntimeError("set both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or neither")
    return FileNotifier(file) if file is not None else LogNotifier()


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
    alert_file: Path | None = None     # owner 2026-11-16: alerts appended here in place of Telegram (not a secret)

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
    alerts: str | None                 # "telegram", "file" or "log" in execution mode
    identity: str | None               # sha256(network:API key) when credentials are loaded
    # The owner's strategy capital (2026-11-16): the strategy sizes from min(account equity, this + its own
    # realized PnL); required on mainnet, optional on the testnet (whole account), unused in paper.
    strategy_capital: Decimal | None = None
    strategy_loss_stop: Decimal | None = None    # a fraction of the strategy capital; None without a capital
    # Owner 2026-11-16: a full position is this many times the strategy capital (1 to 3; 1 without a capital).
    strategy_leverage: Decimal = Decimal(1)

    def record(self) -> dict[str, object]:
        """Every value, as plain JSON, without any secret: what this run uses."""
        limits = None if self.limits is None else {k: str(v) for k, v in self.limits.__dict__.items()}
        return {"testnet": self.testnet, "execution": self.execution, "scope": self.scope,
                "stop_pct": str(self.stop_pct), "limits": limits, "leverage": self.leverage, "margin": self.margin,
                "alerts": self.alerts, "identity": self.identity,
                "strategy_capital": None if self.strategy_capital is None else str(self.strategy_capital),
                "strategy_loss_stop": None if self.strategy_loss_stop is None else str(self.strategy_loss_stop),
                "strategy_leverage": str(self.strategy_leverage)}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True).encode()).hexdigest()


def account_identity(*, testnet: bool, api_key: str) -> str:
    """ADR-0016 point 5: sha256(network + ":" + API key); the key only, never the secret."""
    return hashlib.sha256(f"{'testnet' if testnet else 'mainnet'}:{api_key}".encode()).hexdigest()


def load_config(env: Mapping[str, str] | None = None, *, execution: bool, stop_pct: Decimal | None,
                testnet: bool | None = None, require_credentials: bool = False) -> tuple[RuntimeConfig, Secrets]:
    """Read and check every setting once. A missing or invalid one refuses the start and names it. ``testnet``
    overrides BINANCE_TESTNET (tests); credentials are read only when ``require_credentials``. ``stop_pct`` is the
    command line's ``--stop-pct``; None leaves it to ``STOP_PCT`` (execution) or the default."""
    env = os.environ if env is None else env
    net = testnet_from_env(env) if testnet is None else testnet
    if not execution:
        stop = DEFAULT_STOP_PCT if stop_pct is None else Decimal(stop_pct)
        return (RuntimeConfig(testnet=net, execution=False, scope="paper-target", stop_pct=stop,
                              limits=None, leverage=None, margin=None, alerts=None, identity=None),
                Secrets(binance=None, telegram=None))
    scope = execution_scope(testnet=net, env=env)
    limits = risk_limits(env)
    leverage, margin = exchange_settings(env)
    token, chat_id = env.get("TELEGRAM_BOT_TOKEN", "").strip(), env.get("TELEGRAM_CHAT_ID", "").strip()
    if bool(token) != bool(chat_id):
        raise RuntimeError("set both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or neither")
    file = alert_file(env, telegram=bool(token))
    heartbeat = env.get("HEARTBEAT_URL", "").strip() or None
    if heartbeat is not None and not heartbeat.startswith("https://"):
        raise RuntimeError("HEARTBEAT_URL must be an https:// address")       # never echoed: it is a secret
    capital = strategy_capital(env)
    binance = binance_config_from_env(testnet=net, env=env) if require_credentials else None
    identity = account_identity(testnet=net, api_key=binance.api_key) if binance is not None else None
    config = RuntimeConfig(testnet=net, execution=True, scope=scope, stop_pct=stop_distance(env, stop_pct),
                           limits=limits, leverage=leverage, margin=margin, identity=identity,
                           alerts="telegram" if token else "file" if file else "log",
                           strategy_capital=capital, strategy_loss_stop=strategy_loss_stop(env, capital),
                           strategy_leverage=strategy_leverage(env, capital, leverage))
    return config, Secrets(binance=binance, telegram=(token, chat_id) if token else None, heartbeat_url=heartbeat,
                           alert_file=file)


DEFAULT_STOP_PCT = Decimal("0.08")


def stop_distance(env: Mapping[str, str], given: Decimal | None) -> Decimal:
    """The exchange stop's distance: the command line's ``--stop-pct`` when given, else STOP_PCT (owner 2026-11-16:
    the pilot's 4.5 %, and the mainnet unit passes no flag), else 8 %; within the stop manager's bounds."""
    raw = env.get("STOP_PCT", "").strip()
    try:
        value = Decimal(given) if given is not None else Decimal(raw) if raw else DEFAULT_STOP_PCT
    except (ArithmeticError, ValueError):
        value = Decimal("nan")
    if not value.is_finite() or not MIN_STOP_PCT <= value <= MAX_STOP_PCT:
        raise RuntimeError(f"STOP_PCT (or --stop-pct) must be a fraction between {MIN_STOP_PCT} and {MAX_STOP_PCT}")
    return value


def strategy_leverage(env: Mapping[str, str], capital: Decimal | None, exchange_leverage: int) -> Decimal:
    """STRATEGY_LEVERAGE (owner 2026-11-16): a full position is this many times the strategy capital. 1 unless set;
    above 1 only with a strategy capital; at most MAX_TOTAL_LEVERAGE (the exposure cap) and at most the exchange
    leverage setting the account has (BINANCE_EXCHANGE_LEVERAGE). A person writes it; nothing changes it."""
    raw = env.get("STRATEGY_LEVERAGE", "").strip()
    if not raw:
        return Decimal(1)
    try:
        value = Decimal(raw)
    except (ArithmeticError, ValueError):
        value = Decimal("nan")
    if not value.is_finite() or not 1 <= value <= MAX_TOTAL_LEVERAGE:
        raise RuntimeError(f"STRATEGY_LEVERAGE must be between 1 and {MAX_TOTAL_LEVERAGE}")
    if value > 1 and capital is None:
        raise RuntimeError("STRATEGY_LEVERAGE above 1 is a multiple of the strategy capital: set STRATEGY_CAPITAL_USDT")
    if value > exchange_leverage:
        raise RuntimeError("STRATEGY_LEVERAGE may not exceed the exchange leverage setting BINANCE_EXCHANGE_LEVERAGE")
    return value


def strategy_capital(env: Mapping[str, str]) -> Decimal | None:
    """STRATEGY_CAPITAL_USDT: the owner's strategy capital in USDT, a positive amount; unset or blank means none."""
    raw = env.get("STRATEGY_CAPITAL_USDT", "").strip()
    if not raw:
        return None
    try:
        value = Decimal(raw)
    except (ArithmeticError, ValueError):
        raise RuntimeError("STRATEGY_CAPITAL_USDT must be a positive amount of USDT") from None
    if not value.is_finite() or value <= 0:
        raise RuntimeError("STRATEGY_CAPITAL_USDT must be a positive amount of USDT")
    return value


STRATEGY_LOSS_STOP_MAX_PCT = Decimal(15)     # the owner's limit (2026-11-16); a person can only tighten it


def strategy_loss_stop(env: Mapping[str, str], capital: Decimal | None) -> Decimal | None:
    """STRATEGY_LOSS_STOP_PCT: the cumulative loss, in per cent of the strategy capital, at which every position is
    closed and entries stop. 15 unless set; above 0 and at most 15. Without a strategy capital there is none."""
    raw = env.get("STRATEGY_LOSS_STOP_PCT", "").strip()
    if capital is None:
        if raw:
            raise RuntimeError("STRATEGY_LOSS_STOP_PCT is a share of the strategy capital: set STRATEGY_CAPITAL_USDT")
        return None
    if not raw:
        return STRATEGY_LOSS_STOP_MAX_PCT / 100
    try:
        value = Decimal(raw)
    except (ArithmeticError, ValueError):
        value = Decimal("nan")
    if not value.is_finite() or not 0 < value <= STRATEGY_LOSS_STOP_MAX_PCT:
        raise RuntimeError(f"STRATEGY_LOSS_STOP_PCT must be above 0 and at most {STRATEGY_LOSS_STOP_MAX_PCT}")
    return value / 100


def require_strategy_capital(config: RuntimeConfig) -> None:
    """Owner, 2026-11-16: the strategy shares the main account, so mainnet execution sizes from a capital the owner
    sets and refuses to start without one. The testnet may use the whole account; paper needs none."""
    if config.execution and not config.testnet and config.strategy_capital is None:
        raise RuntimeError("mainnet execution needs the strategy capital: set STRATEGY_CAPITAL_USDT")


def require_operator_channels(config: RuntimeConfig, secrets: Secrets) -> None:
    """ADR-0016 points 1 and 3: execution mode (testnet and mainnet) refuses to start without an alert channel -
    the alert file (owner 2026-11-16) or Telegram - or without a heartbeat address. Paper needs neither."""
    if not config.execution:
        return
    if secrets.telegram is None and secrets.alert_file is None:
        raise RuntimeError("execution mode needs alerts: set ALERT_FILE, or TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
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


def make_notifier(secrets: Secrets) -> TelegramNotifier | FileNotifier | LogNotifier:
    """Telegram when both its values were set, the alert file when that was, the process log otherwise
    (load_config refused half a pair and both channels)."""
    if secrets.telegram is not None:
        return TelegramNotifier(*secrets.telegram, background=True)
    if secrets.alert_file is not None:
        return FileNotifier(secrets.alert_file)
    return LogNotifier()
