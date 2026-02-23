# infra/ci
"""CI/CD utilities — health checks, smoke tests, version info."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class HealthCheck:
    """健康检查结果。"""
    name: str
    healthy: bool
    detail: str = ""


def run_smoke_checks() -> list[HealthCheck]:
    """运行基本健康检查 — 验证核心模块可导入。"""
    checks = []
    modules = [
        "state.snapshot",
        "decision.engine",
        "features.technical",
        "monitoring.metrics",
    ]
    for mod_name in modules:
        try:
            __import__(mod_name)
            checks.append(HealthCheck(mod_name, True))
        except Exception as exc:
            checks.append(HealthCheck(mod_name, False, str(exc)))
    return checks


def get_version() -> str:
    """获取系统版本号。"""
    return "0.1.0"
