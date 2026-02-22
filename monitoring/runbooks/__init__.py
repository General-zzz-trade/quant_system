# monitoring/runbooks
"""Runbooks — automated incident response procedures."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass(frozen=True)
class RunbookStep:
    """运维手册步骤。"""
    description: str
    action: str          # check / restart / alert / escalate
    auto_executable: bool = False


@dataclass(frozen=True)
class Runbook:
    """运维手册。"""
    name: str
    trigger: str
    severity: Severity
    steps: tuple[RunbookStep, ...]
    description: str = ""


# 预定义运维手册
EXCHANGE_DISCONNECT = Runbook(
    name="exchange_disconnect",
    trigger="websocket_disconnected",
    severity=Severity.CRITICAL,
    steps=(
        RunbookStep("Check exchange status page", "check"),
        RunbookStep("Verify network connectivity", "check"),
        RunbookStep("Attempt reconnection", "restart", auto_executable=True),
        RunbookStep("Alert on-call if reconnection fails after 3 attempts", "escalate"),
    ),
    description="交易所WebSocket断连处理流程",
)

RISK_LIMIT_BREACH = Runbook(
    name="risk_limit_breach",
    trigger="risk_limit_exceeded",
    severity=Severity.FATAL,
    steps=(
        RunbookStep("Halt all new order submissions", "restart", auto_executable=True),
        RunbookStep("Log current positions and exposure", "check", auto_executable=True),
        RunbookStep("Notify risk manager", "alert", auto_executable=True),
        RunbookStep("Manual review required before resuming", "escalate"),
    ),
    description="风控限额突破处理流程",
)
