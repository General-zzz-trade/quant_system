from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON or YAML config file.

    YAML support is optional. If PyYAML is not installed, YAML files will raise.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix in {".json"}:
        return json.loads(text)

    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML") from e
        return yaml.safe_load(text) or {}

    raise ValueError(f"unsupported config type: {suffix}")
