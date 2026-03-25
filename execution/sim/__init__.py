"""execution.sim — Simulation and backtesting adapters (Domain 4: ops).

NOT imported by production code. Provides realistic execution simulation:
  - SlippageModel: configurable slippage (none / fixed-bps / volume-impact)
  - LatencyModel: simulated latency (fixed / uniform / gaussian)
  - PaperBroker: full order-matching simulation engine
  - VenueEmulator: lightweight venue mock
  - CostModel: realistic fee/funding/slippage breakdown
  - EmbargoExecutionAdapter: prevents look-ahead in backtests
  - ReplayExecutionAdapter: replays recorded venue responses
  - ShadowExecutionAdapter: mirrors live orders to a shadow venue
"""
from execution.sim.slippage import SlippageModel, NoSlippage, FixedBpsSlippage, VolumeImpactSlippage
from execution.sim.latency import LatencyModel, FixedLatency, UniformLatency, GaussianLatency
from execution.sim.paper_broker import PaperBroker, PaperBrokerConfig
from execution.sim.venue_emulator import VenueEmulator
from execution.sim.cost_model import RealisticCostModel, CostBreakdown

__all__ = [
    # Slippage
    "SlippageModel",
    "NoSlippage",
    "FixedBpsSlippage",
    "VolumeImpactSlippage",
    # Latency
    "LatencyModel",
    "FixedLatency",
    "UniformLatency",
    "GaussianLatency",
    # Paper broker
    "PaperBroker",
    "PaperBrokerConfig",
    # Venue emulator
    "VenueEmulator",
    # Cost model
    "RealisticCostModel",
    "CostBreakdown",
]
