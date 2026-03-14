# runner/builders/monitoring.py
"""Monitoring subsystem builders — alert rules and health server.

Extracted from live_runner.py for organization. These are called by
LiveRunner.build() during assembly.
"""
from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from runner.config import LiveRunnerConfig

logger = logging.getLogger(__name__)


def _build_alert_rules(
    alert_manager: Any,
    health: Optional[Any],
    kill_switch: Any,
    latency_tracker: Optional[Any],
    alpha_health_monitor: Optional[Any],
    correlation_computer: Any,
    config: "LiveRunnerConfig",
    report: Any,
) -> None:
    """Add default alert rules to the AlertManager."""
    from monitoring.alerts.base import Severity
    from monitoring.alerts.manager import AlertRule

    # Stale market data
    if health is not None:
        def _stale_data_condition(h=health, cfg=config) -> bool:
            age = h.get_status().data_age_sec
            return age is not None and age > cfg.health_stale_data_sec

        alert_manager.add_rule(AlertRule(
            name="stale_data",
            condition=_stale_data_condition,
            severity=Severity.WARNING,
            message_template="Market data is stale — check feed connectivity",
            cooldown_sec=120.0,
        ))

    # High drawdown (>15%)
    if health is not None:
        def _high_drawdown_condition(h=health) -> bool:
            dd = h.get_status().drawdown_pct
            return dd is not None and dd > 15.0

        alert_manager.add_rule(AlertRule(
            name="high_drawdown",
            condition=_high_drawdown_condition,
            severity=Severity.ERROR,
            message_template="Portfolio drawdown exceeds 15%",
            cooldown_sec=300.0,
        ))

    # Kill switch triggered
    def _kill_switch_condition(ks=kill_switch) -> bool:
        return ks.is_killed() is not None

    alert_manager.add_rule(AlertRule(
        name="kill_switch_triggered",
        condition=_kill_switch_condition,
        severity=Severity.CRITICAL,
        message_template="Kill switch has been triggered — trading halted",
        cooldown_sec=60.0,
    ))

    # Latency SLA breach
    if latency_tracker is not None:
        from execution.latency.report import LatencyReporter
        _reporter = LatencyReporter(latency_tracker)

        def _latency_sla_condition(reporter=_reporter, thresh=config.latency_p99_threshold_ms) -> bool:
            stats = reporter.compute_stats()
            for s in stats:
                if s.metric == "signal_to_fill" and s.count >= 10 and s.p99_ms > thresh:
                    return True
            return False

        alert_manager.add_rule(AlertRule(
            name="latency_sla_breach",
            condition=_latency_sla_condition,
            severity=Severity.WARNING,
            message_template="Latency SLA breach — signal_to_fill P99 exceeds threshold",
            cooldown_sec=300.0,
        ))

    # Alpha health degradation
    if alpha_health_monitor is not None:
        def _alpha_degradation_condition(
            ahm=alpha_health_monitor, syms=config.symbols,
        ) -> bool:
            return any(ahm.position_scale(sym) < 1.0 for sym in syms)

        alert_manager.add_rule(AlertRule(
            name="alpha_degradation",
            condition=_alpha_degradation_condition,
            severity=Severity.WARNING,
            message_template="Alpha health degraded — position scaling active",
            cooldown_sec=3600.0,
        ))

        def _alpha_retrain_needed_condition(
            ahm=alpha_health_monitor, syms=config.symbols,
        ) -> bool:
            return any(ahm.should_retrain(sym) for sym in syms)

        alert_manager.add_rule(AlertRule(
            name="alpha_retrain_needed",
            condition=_alpha_retrain_needed_condition,
            severity=Severity.ERROR,
            message_template="Alpha IC halted — model retraining required",
            cooldown_sec=86400.0,
        ))

    # High portfolio correlation
    def _high_correlation_condition(
        cc=correlation_computer, syms=config.symbols, thresh=config.max_avg_correlation,
    ) -> bool:
        avg = cc.portfolio_avg_correlation(list(syms))
        return avg is not None and avg > thresh

    alert_manager.add_rule(AlertRule(
        name="high_correlation",
        condition=_high_correlation_condition,
        severity=Severity.WARNING,
        message_template="Portfolio avg correlation exceeds threshold",
        cooldown_sec=300.0,
    ))

    report.record("alert_rules", True)


def _build_health_server(
    config: "LiveRunnerConfig",
    health: Any,
    alpha_health_monitor: Optional[Any],
    regime_sizer: Optional[Any],
    portfolio_allocator: Optional[Any],
    live_signal_tracker: Optional[Any],
    report: Any,
) -> tuple:
    """Build health HTTP server and operator control plane.

    Returns (health_server, control_plane).
    """
    if config.health_port is None or health is None:
        return None, None

    from monitoring.health_server import HealthServer
    from dataclasses import asdict as _asdict
    from runner.control_plane import OperatorControlPlane

    _stale_thresh = config.health_stale_data_sec
    health_token = None
    if config.health_auth_token_env:
        health_token = os.environ.get(config.health_auth_token_env)
        if not health_token:
            raise ValueError(
                "health_auth_token_env is set but env var is missing: "
                f"{config.health_auth_token_env}"
            )

    def _health_status_fn() -> Dict[str, Any]:
        st = health.get_status()
        d = _asdict(st)
        age = st.data_age_sec
        if age is not None and age > _stale_thresh:
            d["status"] = "critical"
        if alpha_health_monitor is not None:
            ah_status = {}
            for sym in config.symbols:
                ah_status[sym] = alpha_health_monitor.get_status(sym)
            d["alpha_health"] = ah_status
        if regime_sizer is not None:
            d["regime_sizer"] = regime_sizer.get_status()
        if portfolio_allocator is not None:
            d["portfolio_allocator"] = portfolio_allocator.get_status()
        return d

    control_plane = OperatorControlPlane(SimpleNamespace())
    control_plane.runner = None

    health_server = HealthServer(
        port=config.health_port,
        status_fn=_health_status_fn,
        operator_fn=lambda: (
            control_plane.runner.operator_status()
            if control_plane.runner is not None
            else {"error": "runner unavailable"}
        ),
        control_history_fn=lambda: [
            {
                "command": rec.command,
                "reason": rec.reason,
                "source": rec.source,
                "result": rec.result,
                "ts": rec.ts.isoformat(),
            }
            for rec in (control_plane.runner.control_history if control_plane.runner is not None else [])
        ],
        control_fn=lambda body: control_plane.execute(body).to_dict() if control_plane.runner is not None else {
            "accepted": False,
            "command": str(body.get("command", "")),
            "outcome": "rejected",
            "reason": str(body.get("reason", "")),
            "source": str(body.get("source", "operator")),
            "status": None,
            "detail": None,
            "error": "runner unavailable",
            "error_code": "runner_unavailable",
        },
        alerts_fn=lambda: control_plane.runner.execution_alert_history() if control_plane.runner is not None else [],
        ops_audit_fn=lambda: control_plane.runner.ops_audit_snapshot() if control_plane.runner is not None else {
            "operator": {"error": "runner unavailable"},
            "control_history": [],
            "execution_alerts": [],
            "model_alerts": [],
            "model_actions": [],
            "model_status": [],
            "timeline": [],
        },
        attribution_fn=lambda: (
            live_signal_tracker.get_status()
            if live_signal_tracker is not None
            else {"error": "attribution tracker unavailable"}
        ),
        host=config.health_host,
        auth_token=health_token,
    )

    report.record("health_server", True)
    return health_server, control_plane
