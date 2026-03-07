"""Tick-level engine for HFT strategies.

Runs in its own thread, independent of the bar engine.
Processes trade ticks and depth updates at sub-second latency.
"""
from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Full, Queue
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Union

from event.tick_types import DepthUpdateEvent, TradeTickEvent
from features.microstructure.streaming import (
    MicrostructureState,
    StreamingMicrostructureComputer,
)

logger = logging.getLogger(__name__)


class HFTStrategy(Protocol):
    """Protocol for HFT strategies consumed by the TickEngine."""

    strategy_id: str

    def on_trade(
        self, tick: TradeTickEvent, micro: MicrostructureState
    ) -> list: ...

    def on_depth(
        self, event: DepthUpdateEvent, micro: MicrostructureState
    ) -> list: ...


@dataclass(frozen=True, slots=True)
class HFTOrder:
    """Lightweight order intent from HFT strategy."""

    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    price: Optional[float] = None  # None = market order
    order_type: str = "LIMIT"  # LIMIT or MARKET
    reduce_only: bool = False
    strategy_id: str = ""


TickEvent = Union[TradeTickEvent, DepthUpdateEvent]


class TickEngine:
    """Independent tick-level engine for HFT.

    Architecture:
    - WS callbacks push events into tick_queue (non-blocking)
    - run() drains queue → updates microstructure → dispatches to strategies
    - Strategy orders go to order_queue for async execution
    """

    def __init__(
        self,
        *,
        queue_size: int = 50_000,
        order_queue_size: int = 10_000,
        risk_checker: Any = None,
    ) -> None:
        self.tick_queue: Queue[TickEvent] = Queue(maxsize=queue_size)
        self.order_queue: Queue[HFTOrder] = Queue(maxsize=order_queue_size)
        self._micro = StreamingMicrostructureComputer()
        self._strategies: list[HFTStrategy] = []
        self._risk_checker = risk_checker
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats_trades = 0
        self._stats_depths = 0
        self._stats_orders = 0

    def register_strategy(self, strategy: HFTStrategy) -> None:
        self._strategies.append(strategy)

    def on_trade_tick(self, tick: TradeTickEvent) -> None:
        """WS callback — non-blocking put into queue."""
        try:
            self.tick_queue.put_nowait(tick)
        except Full:
            logger.warning("tick_queue full, dropping trade tick")

    def on_depth_update(self, event: DepthUpdateEvent) -> None:
        """WS callback — non-blocking put into queue."""
        try:
            self.tick_queue.put_nowait(event)
        except Full:
            logger.warning("tick_queue full, dropping depth update")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self.run, name="tick-engine", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def run(self) -> None:
        """Main loop: drain queue → compute features → dispatch to strategies."""
        logger.info("TickEngine started")
        while self._running:
            try:
                event = self.tick_queue.get(timeout=0.01)
            except Empty:
                continue

            self._process_event(event)

        logger.info(
            "TickEngine stopped — trades=%d depths=%d orders=%d",
            self._stats_trades,
            self._stats_depths,
            self._stats_orders,
        )

    def _process_event(self, event: TickEvent) -> None:
        if isinstance(event, TradeTickEvent):
            micro = self._micro.on_trade(event)
            self._stats_trades += 1
            for strategy in self._strategies:
                orders = strategy.on_trade(event, micro)
                self._submit_orders(orders)
        elif isinstance(event, DepthUpdateEvent):
            micro = self._micro.on_depth(event)
            self._stats_depths += 1
            for strategy in self._strategies:
                orders = strategy.on_depth(event, micro)
                self._submit_orders(orders)

    def _submit_orders(self, orders: list) -> None:
        for order in orders:
            if not isinstance(order, HFTOrder):
                continue
            if self._risk_checker is not None:
                allowed, reason = self._risk_checker.check(order)
                if not allowed:
                    logger.debug("HFT order rejected: %s", reason)
                    continue
            try:
                self.order_queue.put_nowait(order)
                self._stats_orders += 1
            except Full:
                logger.warning("order_queue full, dropping HFT order")
