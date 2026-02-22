from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional


Mode = Literal["live", "paper", "backtest", "replay"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class DecisionContext:
    mode: Mode = "live"
    now: datetime = field(default_factory=utc_now)
    run_id: str = "run-0"
    actor: str = "decision"
    # optional: for experiments / canary
    version: Optional[str] = None
