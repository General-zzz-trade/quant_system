"""Config schema definitions and validation for quant_system.

Defines the expected structure of configuration files and provides
validation against that schema.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from infra.config.loader import validate_config

# Schema: key -> (type, required, description)
SCHEMA: Dict[str, Dict[str, Any]] = {
    # Trading
    "trading.symbol": {"type": str, "required": True, "description": "Primary trading symbol"},
    "trading.exchange": {"type": str, "required": True, "description": "Exchange name"},
    "trading.mode": {"type": str, "required": False, "description": "Trading mode: live|paper|backtest"},

    # Strategy
    "strategy.name": {"type": str, "required": True, "description": "Strategy identifier"},
    "strategy.ma_window": {"type": int, "required": False, "description": "Moving average window"},
    "strategy.order_qty": {"type": str, "required": False, "description": "Default order quantity"},

    # Risk
    "risk.max_position_notional": {"type": float, "required": False, "description": "Max position notional (USD)"},
    "risk.max_leverage": {"type": float, "required": False, "description": "Maximum leverage"},
    "risk.max_drawdown_pct": {"type": float, "required": False, "description": "Max drawdown percentage"},
    "risk.max_orders_per_minute": {"type": int, "required": False, "description": "Order frequency limit"},

    # Execution
    "execution.fee_bps": {"type": float, "required": False, "description": "Fee in basis points"},
    "execution.slippage_bps": {"type": float, "required": False, "description": "Estimated slippage in bps"},

    # Credentials (env var names, not actual secrets)
    "credentials.api_key_env": {"type": str, "required": False, "description": "Env var name for API key"},
    "credentials.api_secret_env": {"type": str, "required": False, "description": "Env var name for API secret"},

    # Logging
    "logging.level": {"type": str, "required": False, "description": "Log level: DEBUG|INFO|WARNING|ERROR"},
    "logging.structured": {"type": bool, "required": False, "description": "Use JSON structured logging"},
    "logging.file": {"type": str, "required": False, "description": "Log file path"},

    # Data
    "data.store_path": {"type": str, "required": False, "description": "Path to data store directory"},
    "data.format": {"type": str, "required": False, "description": "Storage format: parquet|csv"},

    # Monitoring
    "monitoring.health_check_interval": {"type": float, "required": False, "description": "Health check interval (s)"},
    "monitoring.health_port": {"type": int, "required": False, "description": "Health/control API port"},
    "monitoring.health_host": {"type": str, "required": False, "description": "Health/control API bind host"},
    "monitoring.health_auth_token_env": {"type": str, "required": False, "description": "Env var for health/control API bearer token"},
}


def validate_trading_config(config: Dict[str, Any]) -> List[str]:
    """Validate a trading configuration against the schema.

    Returns a list of error messages (empty = valid).
    """
    required_keys = [k for k, v in SCHEMA.items() if v.get("required")]
    type_checks = {k: v["type"] for k, v in SCHEMA.items() if "type" in v}

    return validate_config(
        config,
        required_keys=required_keys,
        type_checks=type_checks,
    )


def get_schema_docs() -> str:
    """Generate human-readable schema documentation."""
    lines = ["# Configuration Schema", ""]
    current_section = ""

    for key, spec in SCHEMA.items():
        section = key.split(".")[0]
        if section != current_section:
            current_section = section
            lines.append(f"\n## {section.title()}")

        required = " (required)" if spec.get("required") else ""
        type_name = spec.get("type", str).__name__
        desc = spec.get("description", "")
        lines.append(f"  {key}: {type_name}{required} — {desc}")

    return "\n".join(lines)
