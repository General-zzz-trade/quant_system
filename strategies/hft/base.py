"""HFT strategy base protocol.

Defines the interface that all HFT strategies must implement.
"""
from __future__ import annotations

from typing import Protocol

from engine.tick_engine import HFTOrder
from event.tick_types import DepthUpdateEvent, TradeTickEvent
from features.microstructure.streaming import MicrostructureState


class HFTStrategy(Protocol):
    """Protocol for HFT strategies.

    Strategies receive trade ticks and depth updates with current
    microstructure state, and return a list of HFTOrder intents.
    """

    strategy_id: str

    def on_trade(
        self, tick: TradeTickEvent, micro: MicrostructureState
    ) -> list[HFTOrder]: ...

    def on_depth(
        self, event: DepthUpdateEvent, micro: MicrostructureState
    ) -> list[HFTOrder]: ...
