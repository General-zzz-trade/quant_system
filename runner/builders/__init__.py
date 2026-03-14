"""Runner builders -- assembly components for runner subsystems.

These builders provide reusable components for:
- runner/live_runner.py (production: _build_alert_rules, _build_health_server, _build_multi_tf_ensemble)
- runner/live_paper_runner.py (paper trading with full stack)
- runner/testnet_validation.py (testnet validation workflow)
- Custom runner implementations
"""
# Each module exports a pure function that constructs one subsystem group
# and returns a dataclass of assembled components.
#
# Usage (incremental adoption):
#   from runner.builders import build_risk_subsystem
#   risk = build_risk_subsystem(config, ...)
#   kill_switch = risk.kill_switch
#
# Builders:
#   - inference.py  → build_inference_subsystem
#   - execution.py  → build_execution_subsystem
#   - risk.py       → build_risk_subsystem
#   - market_data.py → build_market_data_subsystem
#   - recovery_builder.py → build_recovery_subsystem

from runner.builders.inference import build_inference_subsystem, _build_multi_tf_ensemble
from runner.builders.execution import build_execution_subsystem
from runner.builders.risk import build_risk_subsystem
from runner.builders.market_data import build_market_data_subsystem
from runner.builders.recovery_builder import build_recovery_subsystem
from runner.builders.monitoring import _build_alert_rules, _build_health_server

__all__ = [
    "build_inference_subsystem",
    "build_execution_subsystem",
    "build_risk_subsystem",
    "build_market_data_subsystem",
    "build_recovery_subsystem",
    "_build_multi_tf_ensemble",
    "_build_alert_rules",
    "_build_health_server",
]
