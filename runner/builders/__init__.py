# runner/builders — extracted subsystem builders for LiveRunner.build().
#
# Each module exports a pure function that constructs one subsystem group
# and returns a dataclass of assembled components.  LiveRunner.build() can
# call these instead of inline construction to shrink the build method.
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

from runner.builders.inference import build_inference_subsystem
from runner.builders.execution import build_execution_subsystem
from runner.builders.risk import build_risk_subsystem
from runner.builders.market_data import build_market_data_subsystem
from runner.builders.recovery_builder import build_recovery_subsystem

__all__ = [
    "build_inference_subsystem",
    "build_execution_subsystem",
    "build_risk_subsystem",
    "build_market_data_subsystem",
    "build_recovery_subsystem",
]
