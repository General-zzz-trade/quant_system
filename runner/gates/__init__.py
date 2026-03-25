"""Gate implementations — re-exports from strategy/gates/ for backward compat."""
from strategy.gates.adaptive_stop_gate import AdaptiveStopGate
from strategy.gates.carry_cost_gate import CarryCostGate
from strategy.gates.consensus_scaling_gate import ConsensusScalingGate
from strategy.gates.equity_leverage_gate import EquityLeverageGate
from strategy.gates.liquidation_cascade_gate import LiquidationCascadeGate
from strategy.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
from strategy.gates.evaluator import GateEvaluator

__all__ = [
    "AdaptiveStopGate", "CarryCostGate", "ConsensusScalingGate",
    "EquityLeverageGate", "LiquidationCascadeGate", "MultiTFConfluenceGate",
    "GateEvaluator",
]

try:
    from strategy.gates.vpin_entry_gate import VPINEntryGate, VPINEntryConfig
    __all__ += ["VPINEntryGate", "VPINEntryConfig"]
except ImportError:
    pass
