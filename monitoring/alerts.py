from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Protocol


class AlertSink(Protocol):
    def emit(self, *, level: str, message: str, meta: Optional[Dict[str, Any]] = None) -> None:
        ...


@dataclass
class PrintAlertSink:
    def emit(self, *, level: str, message: str, meta: Optional[Dict[str, Any]] = None) -> None:
        now = datetime.utcnow().isoformat()
        meta_s = "" if not meta else f" {meta}"
        print(f"[{now}] {level.upper()}: {message}{meta_s}")
