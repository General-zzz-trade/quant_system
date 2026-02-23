from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load JSON or YAML config.

    YAML support is optional.
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
