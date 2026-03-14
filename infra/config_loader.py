"""Legacy config loader -- prefer infra.config.loader instead.

DEPRECATED: This module is a standalone legacy loader. Production code uses
infra.config.loader which provides additional features (validation, secrets
management, secure credential resolution via environment variables).

This file is kept for backward compatibility with any external scripts that
may import from infra.config_loader directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load JSON or YAML config.

    YAML support is optional.

    .. deprecated::
        Use ``infra.config.loader.load_config`` or ``load_config_secure`` instead.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() in (".json",):
        return json.loads(p.read_text(encoding="utf-8"))

    if p.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires pyyaml installed") from e
        return yaml.safe_load(p.read_text(encoding="utf-8"))

    raise ValueError(f"unsupported config format: {p.suffix}")
