"""Infrastructure config — bridges file-based config with core.config.ConfigService.

Usage::

    from infra.config import load_config, get_config_service

    # Load raw dict (standalone)
    raw = load_config("config.json")

    # Get a ConfigService wired with file + env + defaults
    svc = get_config_service("config.json", env_prefix="QS_")
    port = svc.get("server.port", int)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from infra.config.loader import load_config

from core.config import ConfigService

__all__ = ["load_config", "get_config_service", "ConfigService"]


def get_config_service(
    config_file: Optional[str] = None,
    *,
    defaults: Optional[Dict[str, Any]] = None,
    env_prefix: str = "QS_",
) -> ConfigService:
    """Create a ConfigService wired with file, env, and defaults."""
    return ConfigService(
        defaults=defaults or {},
        file_path=config_file,
        env_prefix=env_prefix,
    )
