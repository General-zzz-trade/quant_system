# runner/gates/__init__.py
"""Gate implementations for the GateChain order pipeline."""
from runner.gates.adaptive_stop_gate import AdaptiveStopGate
from runner.gates.consensus_scaling_gate import ConsensusScalingGate
from runner.gates.equity_leverage_gate import EquityLeverageGate

# Gates moved to strategy/gates/ — re-export for backward compat
from strategy.gates.carry_cost_gate import CarryCostGate
from strategy.gates.liquidation_cascade_gate import LiquidationCascadeGate
from strategy.gates.multi_tf_confluence_gate import MultiTFConfluenceGate

__all__ = [
    "AdaptiveStopGate",
    "CarryCostGate",
    "ConsensusScalingGate",
    "EquityLeverageGate",
    "LiquidationCascadeGate",
    "MultiTFConfluenceGate",
]

# VPINEntryGate requires Rust _quant_hotpath — optional import
try:
    from runner.gates.vpin_entry_gate import VPINEntryGate, VPINEntryConfig  # noqa: F401
    __all__ += ["VPINEntryGate", "VPINEntryConfig"]
except ImportError:
    pass
