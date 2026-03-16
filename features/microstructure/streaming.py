"""Streaming microstructure feature computer for tick-level HFT.

Maintains rolling state and recomputes features on each tick/depth update.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from event.tick_types import DepthUpdateEvent, TradeTickEvent
from features.microstructure.kyle_lambda import KyleLambdaEstimator
from _quant_hotpath import RustVPINCalculator
# order_book module removed with strategies/hft/ cleanup
# Stub types to keep module importable (unused in production)
from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True, slots=True)
class OrderBookSignal:
    imbalance: float = 0.0
    spread_bps: float = 0.0
    weighted_mid: object = None
    signal: str = "neutral"
    depth_ratio: float = 1.0


def analyze_book(snapshot):  # type: ignore[no-untyped-def]
    """Stub — original lived in strategies.hft.order_book (deleted)."""
    return OrderBookSignal()

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MicrostructureState:
    """Current snapshot of all microstructure indicators."""

    vpin: float = 0.0
    kyle_lambda: float = 0.0
    kyle_r_squared: float = 0.0
    ob_imbalance: float = 0.0
    spread_bps: float = 0.0
    weighted_mid: Optional[Decimal] = None
    ob_signal: str = "neutral"
    depth_ratio: float = 1.0
    trade_count: int = 0
    last_price: Optional[Decimal] = None
    last_trade_ts_ms: int = 0


class StreamingMicrostructureComputer:
    """Computes microstructure features incrementally on each tick.

    Maintains a rolling trade buffer and recomputes VPIN + Kyle Lambda
    on each trade. OB features are updated on each depth snapshot.
    """

    def __init__(
        self,
        *,
        trade_buffer_size: int = 200,
        vpin_bucket_volume: Decimal = Decimal("100"),
        vpin_n_buckets: int = 50,
        kyle_window: int = 100,
    ) -> None:
        self._trades: deque[TradeTickEvent] = deque(maxlen=trade_buffer_size)
        self._vpin_calc = RustVPINCalculator(
            bucket_volume=float(vpin_bucket_volume),
            n_buckets=vpin_n_buckets,
        )
        self._kyle_est = KyleLambdaEstimator(window=kyle_window)
        self._last_ob: Optional[OrderBookSignal] = None
        self._last_spread_bps: float = 0.0
        self._last_weighted_mid: Optional[Decimal] = None

    def on_trade(self, tick: TradeTickEvent) -> MicrostructureState:
        """Process a trade tick and return updated microstructure state."""
        self._trades.append(tick)

        ticks_list = list(self._trades)
        vpin_result = self._vpin_calc.calculate(ticks_list)
        kyle_result = self._kyle_est.estimate(ticks_list)

        ob = self._last_ob
        return MicrostructureState(
            vpin=vpin_result.vpin,
            kyle_lambda=kyle_result.kyle_lambda,
            kyle_r_squared=kyle_result.r_squared,
            ob_imbalance=ob.imbalance if ob else 0.0,
            spread_bps=self._last_spread_bps if ob else 0.0,
            weighted_mid=self._last_weighted_mid,
            ob_signal=ob.signal if ob else "neutral",
            depth_ratio=ob.depth_ratio if ob else 1.0,
            trade_count=len(self._trades),
            last_price=tick.price,
            last_trade_ts_ms=tick.ts_ms,
        )

    def on_depth(self, event: DepthUpdateEvent) -> MicrostructureState:
        """Process a depth update and return updated microstructure state."""
        ob = analyze_book(event.snapshot)
        self._last_ob = ob
        self._last_spread_bps = ob.spread_bps
        self._last_weighted_mid = ob.weighted_mid

        last_tick = self._trades[-1] if self._trades else None

        return MicrostructureState(
            vpin=self._last_vpin(),
            kyle_lambda=self._last_kyle(),
            kyle_r_squared=0.0,
            ob_imbalance=ob.imbalance,
            spread_bps=ob.spread_bps,
            weighted_mid=ob.weighted_mid,
            ob_signal=ob.signal,
            depth_ratio=ob.depth_ratio,
            trade_count=len(self._trades),
            last_price=last_tick.price if last_tick else None,
            last_trade_ts_ms=last_tick.ts_ms if last_tick else 0,
        )

    def _last_vpin(self) -> float:
        if not self._trades:
            return 0.0
        return self._vpin_calc.calculate(list(self._trades)).vpin

    def _last_kyle(self) -> float:
        if not self._trades:
            return 0.0
        return self._kyle_est.estimate(list(self._trades)).kyle_lambda
