"""The settings report of the account check (tech lead 2026-11-16): which settings the service's file holds and what
the runtime's own configuration checks say about them, without anyone reading the file by hand.

    python3 -m scripts.account_check --settings [--env-file PATH]

The file is read in memory only. Secrets are reported as set or not set, never shown. Plain settings are shown with
their values. Then the runtime's configuration is built from them exactly as a mainnet execution start builds it
(``trading.infrastructure.config``) and the start's local checks run: what it would refuse is named. Nothing is sent
anywhere; the account is not contacted (the rest of the account check does that).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

SECRET = ("BINANCE_API_KEY", "BINANCE_API_SECRET", "BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET",
          "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "HEARTBEAT_URL", "QUANT_MODEL_SIGN_KEY")
PLAIN = ("BINANCE_TESTNET", "RISK_SCOPE", "STRATEGY_CAPITAL_USDT", "STRATEGY_LEVERAGE", "STOP_PCT",
         "STRATEGY_LOSS_STOP_PCT", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE", "ALERT_FILE",
         "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT", "BINANCE_MAX_ORDER_NOTIONAL",
         "BINANCE_MAX_DAILY_NOTIONAL", "BINANCE_MAX_DAILY_ORDERS", "QUANT_ALLOW_UNSIGNED_MODELS")


def read_all(path: Path) -> dict[str, str]:
    """Every NAME=value line, the way systemd's EnvironmentFile reads them closely enough for this report."""
    found: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        if sep and name.strip():
            found[name.strip()] = value.strip().strip("'\"")
    return found


def report(values: Mapping[str, str]) -> tuple[list[str], bool]:
    """The report's lines and whether a mainnet execution start would pass its local configuration checks."""
    lines = ["== settings (secrets: set or not set only)"]
    for name in SECRET:
        lines.append(f"  {name}: {'set' if values.get(name, '').strip() else 'not set'}")
    for name in PLAIN:
        value = values.get(name, "").strip()
        lines.append(f"  {name}: {value if value else 'not set'}")
    others = sorted(n for n in values if n not in SECRET and n not in PLAIN)
    if others:
        lines.append(f"  other names in the file: {', '.join(others)}")
    lines.append("== the runtime's configuration checks (a mainnet execution start, before any account request)")
    try:
        from trading.domain.risk.liquidation import check_stop_inside_liquidation
        from trading.infrastructure.config import (load_config, require_operator_channels,
                                                   require_strategy_capital)
        env = {**values, "BINANCE_TESTNET": values.get("BINANCE_TESTNET", "0")}
        config, secrets = load_config(env, execution=True, stop_pct=None)
        require_strategy_capital(config)
        require_operator_channels(config, secrets)
        check_stop_inside_liquidation(float(config.stop_pct), config.leverage or 0, config.margin)
    except (RuntimeError, ValueError) as exc:
        lines.append(f"  refused: {_without(str(exc), values)}")
        return lines, False
    lines.append("  the runtime's configuration checks: passed")
    lines.append("  " + json.dumps(config.record(), sort_keys=True))
    return lines, True


def _without(text: str, values: Mapping[str, str]) -> str:
    """A check's message, with any secret value it might quote taken out (none does today)."""
    for name in SECRET:
        secret = values.get(name, "").strip()
        if secret:
            text = text.replace(secret, "<hidden>")
    return text


__all__ = ["PLAIN", "SECRET", "read_all", "report"]
