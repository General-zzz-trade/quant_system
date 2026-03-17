"""Runner builders -- assembly components for runner subsystems.

Phase builders (used by LiveRunner.build() — the production path):
  - core_infra_builder.py → build_core_infra
  - rust_components_builder.py → build_rust_components (Phase 1.5)
  - monitoring_builder.py → build_monitoring
  - portfolio_builder.py → build_portfolio_and_correlation
  - order_infra_builder.py → build_order_infra
  - features_builder.py → build_features_and_inference
  - engine_builder.py → build_coordinator_and_pipeline
  - execution_builder.py → build_execution_phase
  - decision_builder.py → build_decision
  - market_data_builder.py → build_market_data
  - user_stream_builder.py → build_user_stream

Legacy builders (not used in production, retained for backcompat):
  - inference.py → build_inference_subsystem
  - execution.py → build_execution_subsystem
  - risk.py → build_risk_subsystem (also contains RiskSubsystem dataclass)
  - market_data.py → build_market_data_subsystem
  - recovery_builder.py → build_recovery_subsystem
"""

# Legacy builders — retained for backward compatibility
from runner.builders.inference import build_inference_subsystem, _build_multi_tf_ensemble
from runner.builders.execution import build_execution_subsystem
from runner.builders.risk import build_risk_subsystem
from runner.builders.market_data import build_market_data_subsystem
from runner.builders.recovery_builder import build_recovery_subsystem
from runner.builders.monitoring import _build_alert_rules, _build_health_server

from runner.builders.rust_components_builder import build_rust_components, RustComponents

__all__ = [
    # Phase builders
    "build_rust_components",
    "RustComponents",
    # Legacy
    "build_inference_subsystem",
    "build_execution_subsystem",
    "build_risk_subsystem",
    "build_market_data_subsystem",
    "build_recovery_subsystem",
    # Shared utilities
    "_build_multi_tf_ensemble",
    "_build_alert_rules",
    "_build_health_server",
]
