# execution/algo_adapter.py
"""AlgoExecutionAdapter — routes orders through execution algorithms."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

from execution.algos.twap import TWAPAlgo
from execution.algos.vwap import VWAPAlgo
from execution.algos.iceberg import IcebergAlgo

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlgoConfig:
    """Configuration for algo execution routing."""
    large_order_notional: Decimal = Decimal("10000")
    default_algo: str = "twap"
    twap_slices: int = 10
    twap_duration_sec: float = 600
    vwap_slices: int = 10
    vwap_duration_sec: float = 600
    iceberg_clip_fraction: float = 0.1
    tick_interval_sec: float = 1.0


def _make_fill_event(
    order_event: Any,
    fill_price: Decimal,
    fill_qty: Decimal,
) -> SimpleNamespace:
    """Construct a FillEvent-like object from an algo fill."""
    return SimpleNamespace(
        event_type="fill",
        EVENT_TYPE="fill",
        header=SimpleNamespace(
            event_type="fill",
            ts=None,
            event_id=f"algofill-{uuid.uuid4().hex[:12]}",
        ),
        fill_id=f"algofill-{uuid.uuid4().hex[:12]}",
        order_id=getattr(order_event, "order_id", ""),
        symbol=getattr(order_event, "symbol", ""),
        side=getattr(order_event, "side", ""),
        qty=fill_qty,
        quantity=fill_qty,
        price=fill_price,
        fee=Decimal("0"),
        realized_pnl=Decimal("0"),
    )


@dataclass
class AlgoExecutionAdapter:
    """ExecutionAdapter that routes orders through TWAP/VWAP/Iceberg.

    Small orders (notional < threshold) -> immediate market order via submit_fn.
    Large orders -> delegated to execution algo, ticked by daemon thread.

    Implements ExecutionAdapter protocol: send_order(order_event) -> Iterable[Event]
    """

    submit_fn: Callable[[str, str, Decimal], Optional[Decimal]]
    dispatcher_emit: Callable[[Any], None]
    cfg: AlgoConfig = field(default_factory=AlgoConfig)

    _active_orders: Dict[str, tuple] = field(default_factory=dict, init=False)
    _active_events: Dict[str, Any] = field(default_factory=dict, init=False)
    _algos: Dict[str, Any] = field(default_factory=dict, init=False)
    _running: bool = field(default=False, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._algos = {
            "twap": TWAPAlgo(submit_fn=self.submit_fn),
            "vwap": VWAPAlgo(submit_fn=self.submit_fn),
            "iceberg": IcebergAlgo(submit_fn=self.submit_fn),
        }

    def send_order(self, order_event: Any) -> Iterable[Any]:
        """Route order: small -> direct, large -> algo."""
        symbol = getattr(order_event, "symbol", "")
        side = getattr(order_event, "side", "")
        qty = Decimal(str(getattr(order_event, "qty", 0)))
        price = getattr(order_event, "price", None)

        # Estimate notional for routing decision
        notional = qty * Decimal(str(price)) if price else qty

        if notional < self.cfg.large_order_notional:
            # Direct execution
            fill_price = self.submit_fn(symbol, side, qty)
            if fill_price is not None:
                return [_make_fill_event(order_event, fill_price, qty)]
            return []

        # Large order -> algo
        algo_name = self.cfg.default_algo
        algo = self._algos.get(algo_name)
        if algo is None:
            logger.warning("Unknown algo %s, falling back to direct", algo_name)
            fill_price = self.submit_fn(symbol, side, qty)
            if fill_price is not None:
                return [_make_fill_event(order_event, fill_price, qty)]
            return []

        order_id = getattr(order_event, "order_id", uuid.uuid4().hex[:12])
        algo_order = self._create_algo_order(algo, algo_name, symbol, side, qty)

        with self._lock:
            self._active_orders[order_id] = (algo_name, algo_order)
            self._active_events[order_id] = order_event

        logger.info("Large order %s routed to %s algo", order_id, algo_name)
        self._ensure_ticker_running()
        return []  # fills arrive asynchronously via dispatcher_emit

    def _create_algo_order(
        self, algo: Any, algo_name: str, symbol: str, side: str, qty: Decimal,
    ) -> Any:
        if algo_name == "twap":
            return algo.create(
                symbol, side, qty,
                n_slices=self.cfg.twap_slices,
                duration_sec=self.cfg.twap_duration_sec,
            )
        elif algo_name == "vwap":
            return algo.create(
                symbol, side, qty,
                n_slices=self.cfg.vwap_slices,
                duration_sec=self.cfg.vwap_duration_sec,
            )
        elif algo_name == "iceberg":
            clip_size = qty * Decimal(str(self.cfg.iceberg_clip_fraction))
            return algo.create(symbol, side, qty, clip_size=clip_size)
        raise ValueError(f"Unknown algo: {algo_name}")

    def _ensure_ticker_running(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._tick_loop, name="algo-ticker", daemon=True,
        )
        self._thread.start()

    def _tick_loop(self) -> None:
        while self._running:
            time.sleep(self.cfg.tick_interval_sec)
            self._tick_all()

    def _tick_all(self) -> None:
        with self._lock:
            items = list(self._active_orders.items())

        completed: List[str] = []
        for order_id, (algo_name, algo_order) in items:
            algo = self._algos[algo_name]
            result = algo.tick(algo_order)

            if result is not None:
                fill_price = getattr(result, "fill_price", None)
                fill_qty = getattr(result, "qty", None)
                if fill_price is not None and fill_qty is not None:
                    with self._lock:
                        original = self._active_events.get(order_id)
                    if original is not None:
                        fill_ev = _make_fill_event(original, fill_price, fill_qty)
                        try:
                            self.dispatcher_emit(fill_ev)
                        except Exception:
                            logger.exception("Failed to emit algo fill")

            is_complete = getattr(algo_order, "is_complete", False)
            if is_complete:
                completed.append(order_id)

        if completed:
            with self._lock:
                for oid in completed:
                    self._active_orders.pop(oid, None)
                    self._active_events.pop(oid, None)
                if not self._active_orders:
                    self._running = False

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
