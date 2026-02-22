from __future__ import annotations

import json
from typing import Any, Mapping


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(s: str) -> Any:
    return json.loads(s)
