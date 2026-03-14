"""Order book imbalance scalper — HFT strategy.

Opens positions on strong order book imbalance, exits on reversal,
toxicity spike, or timeout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from engine.tick_engine import HFTOrder
from event.tick_types import DepthUpdateEvent, TradeTickEvent
from features.microstructure.streaming import MicrostructureState
from state.shared_position import SharedPositionStore


@dataclass(frozen=True, slots=True)
class ImbalanceScalperConfig:
    imbalance_threshold: float = 0.4
    vpin_max_toxicity: float = 0.7
    max_hold_seconds: float = 30.0
    base_qty: float = 0.01
    max_position: float = 0.1
    min_spread_bps: float = 1.0


@dataclass
class ImbalanceScalper:
    """Scalps on order book imbalance with VPIN toxicity filter.

    Entry: imbalance > threshold AND VPIN < max_toxicity
    Exit: reversal, VPIN spike, or timeout
    """

    strategy_id: str = "imbalance_scalper"
    cfg: ImbalanceScalperConfig = field(default_factory=ImbalanceScalperConfig)
    position_store: Optional[SharedPositionStore] = None
    symbol: str = "BTCUSDT"

    _entry_time: Optional[float] = None
    _entry_side: Optional[str] = None

    def _get_position(self) -> float:
        if self.position_store is None:
            return 0.0
        return float(self.position_store.get_position(self.symbol))

    def on_trade(
        self, tick: TradeTickEvent, micro: MicrostructureState
    ) -> list[HFTOrder]:
        return []

    def on_depth(
        self, event: DepthUpdateEvent, micro: MicrostructureState
    ) -> list[HFTOrder]:
        orders: list[HFTOrder] = []
        now = time.monotonic()
        position = self._get_position()

        # Exit: timeout
        if self._entry_time is not None and self._entry_side is not None:
            elapsed = now - self._entry_time
            if elapsed > self.cfg.max_hold_seconds:
                orders.append(self._close_order(position))
                self._reset()
                return orders

        # Exit: VPIN toxicity spike while in position
        if position != 0.0 and micro.vpin > self.cfg.vpin_max_toxicity:
            orders.append(self._close_order(position))
            self._reset()
            return orders

        # Skip if spread too tight (not worth the cost)
        if micro.spread_bps < self.cfg.min_spread_bps:
            return orders

        # Skip if VPIN too high (toxic flow)
        if micro.vpin > self.cfg.vpin_max_toxicity:
            return orders

        # Entry: strong buy pressure
        if (
            micro.ob_imbalance > self.cfg.imbalance_threshold
            and position < self.cfg.max_position
        ):
            orders.append(
                HFTOrder(
                    symbol=self.symbol,
                    side="buy",
                    qty=self.cfg.base_qty,
                    price=float(micro.weighted_mid) if micro.weighted_mid else None,
                    order_type="LIMIT",
                    strategy_id=self.strategy_id,
                )
            )
            if self._entry_time is None:
                self._entry_time = now
                self._entry_side = "buy"

        # Entry: strong sell pressure
        elif (
            micro.ob_imbalance < -self.cfg.imbalance_threshold
            and position > -self.cfg.max_position
        ):
            orders.append(
                HFTOrder(
                    symbol=self.symbol,
                    side="sell",
                    qty=self.cfg.base_qty,
                    price=float(micro.weighted_mid) if micro.weighted_mid else None,
                    order_type="LIMIT",
                    strategy_id=self.strategy_id,
                )
            )
            if self._entry_time is None:
                self._entry_time = now
                self._entry_side = "sell"

        return orders

    def _close_order(self, position: float) -> HFTOrder:
        """Generate a market order to flatten the position."""
        side = "sell" if position > 0 else "buy"
        return HFTOrder(
            symbol=self.symbol,
            side=side,
            qty=abs(position),
            order_type="MARKET",
            reduce_only=True,
            strategy_id=self.strategy_id,
        )

    def _reset(self) -> None:
        self._entry_time = None
        self._entry_side = None
