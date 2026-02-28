"""Root conftest — shared fixtures for all tests."""
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Set

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Market event factory ─────────────────────────────────────

@pytest.fixture
def make_market_event():
    """Factory fixture: create MARKET events."""
    def _make(symbol: str = "BTCUSDT", close: float = 40000.0, idx: int = 0):
        return SimpleNamespace(
            event_type="MARKET",
            symbol=symbol,
            open=close, high=close + 5, low=close - 5, close=close,
            volume=100.0,
            ts=datetime(2024, 1, 1, 0, idx),
            header=SimpleNamespace(
                event_id=f"mkt_{symbol}_{idx}",
                ts=datetime(2024, 1, 1, 0, idx),
            ),
        )
    return _make


@pytest.fixture
def make_order_event():
    """Factory fixture: create ORDER events."""
    _seq = [0]

    def _make(symbol: str = "BTCUSDT", side: str = "buy", qty: float = 0.01,
              price: float = 40000.0, intent_id: str = ""):
        _seq[0] += 1
        return SimpleNamespace(
            event_type="order",
            EVENT_TYPE="order",
            symbol=symbol,
            venue="mock",
            order_id=f"ord_{_seq[0]}",
            intent_id=intent_id or f"intent_{_seq[0]}",
            command_id=f"cmd_{_seq[0]}",
            idempotency_key=f"idem_{_seq[0]}",
            side=side,
            qty=qty,
            price=price,
            order_type="limit",
        )
    return _make


@pytest.fixture
def make_fill_event():
    """Factory fixture: create FILL events."""
    _seq = [0]

    def _make(symbol: str = "BTCUSDT", side: str = "buy", qty: float = 0.01,
              price: float = 40000.0, order_id: str = ""):
        _seq[0] += 1
        return SimpleNamespace(
            event_type="fill",
            EVENT_TYPE="fill",
            fill_id=f"fill_{_seq[0]}",
            order_id=order_id or f"ord_{_seq[0]}",
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            fee=0.0,
            realized_pnl=0.0,
            margin_change=0.0,
            header=SimpleNamespace(
                event_id=f"fill_{_seq[0]}",
                ts=datetime(2024, 1, 1, 1, 0),
            ),
        )
    return _make


# ── Mock venue adapter ───────────────────────────────────────

class MockVenueAdapter:
    """Mock venue that returns fill events for every order."""

    def __init__(self) -> None:
        self.orders: List[Any] = []
        self._fill_seq = 0

    def send_order(self, order_event: Any) -> list:
        self.orders.append(order_event)
        self._fill_seq += 1
        return [SimpleNamespace(
            event_type="fill",
            EVENT_TYPE="fill",
            fill_id=f"fill_{self._fill_seq}",
            order_id=getattr(order_event, "order_id", f"ord_{self._fill_seq}"),
            symbol=getattr(order_event, "symbol", "BTCUSDT"),
            side=getattr(order_event, "side", "buy"),
            qty=getattr(order_event, "qty", 0.01),
            price=getattr(order_event, "price", 40000.0),
            fee=0.0,
            realized_pnl=0.0,
            margin_change=0.0,
            header=SimpleNamespace(
                event_id=f"fill_{self._fill_seq}",
                ts=datetime(2024, 1, 1, 1, 0),
            ),
        )]


@pytest.fixture
def mock_venue():
    return MockVenueAdapter()


# ── Mock decision module ─────────────────────────────────────

class MockDecisionModule:
    """Issues a buy order when close > threshold."""

    def __init__(self, threshold: float = 40100.0) -> None:
        self.snapshots: List[Any] = []
        self._ordered_bars: Set[int] = set()
        self.threshold = threshold
        self._seq = 0

    def decide(self, snapshot: Any) -> list:
        self.snapshots.append(snapshot)
        if snapshot.event_type.upper() != "MARKET":
            return []
        if snapshot.bar_index in self._ordered_bars:
            return []
        close = snapshot.markets[snapshot.symbol].close
        if close > self.threshold:
            self._ordered_bars.add(snapshot.bar_index)
            self._seq += 1
            return [SimpleNamespace(
                event_type="order",
                EVENT_TYPE="order",
                symbol=snapshot.symbol,
                venue="mock",
                order_id=f"ord_{self._seq}",
                intent_id=f"intent_{self._seq}",
                command_id=f"cmd_{snapshot.bar_index}",
                idempotency_key=f"idem_{snapshot.bar_index}",
                side="buy",
                qty=0.01,
                price=close,
                order_type="limit",
            )]
        return []


@pytest.fixture
def mock_decision():
    return MockDecisionModule()


# ── Coordinator stack builder ─────────────────────────────────

@pytest.fixture
def build_coordinator_stack():
    """Factory fixture: build a coordinator + decision + execution stack."""
    from engine.coordinator import CoordinatorConfig, EngineCoordinator
    from engine.decision_bridge import DecisionBridge
    from engine.execution_bridge import ExecutionBridge

    def _build(symbols=("BTCUSDT",), decision_module=None, venue_adapter=None):
        venue = venue_adapter or MockVenueAdapter()
        decision = decision_module or MockDecisionModule()

        cfg = CoordinatorConfig(
            symbol_default=symbols[0],
            symbols=symbols,
            currency="USDT",
            starting_balance=10000.0,
        )
        coord = EngineCoordinator(cfg=cfg)

        def _emit(ev: Any) -> None:
            coord.emit(ev, actor="bridge")

        bridge = DecisionBridge(dispatcher_emit=_emit, modules=[decision])
        exec_bridge = ExecutionBridge(adapter=venue, dispatcher_emit=_emit)
        coord.attach_decision_bridge(bridge)
        coord.attach_execution_bridge(exec_bridge)
        coord.start()

        return SimpleNamespace(
            coordinator=coord,
            venue=venue,
            decision=decision,
            decision_bridge=bridge,
            emit=_emit,
        )
    return _build


# ── Snapshot builder ──────────────────────────────────────────

@pytest.fixture
def make_snapshot():
    """Factory fixture: create minimal snapshot objects."""
    def _make(symbol: str = "BTCUSDT", close: float = 40000.0, bar_index: int = 0):
        return SimpleNamespace(
            symbol=symbol,
            bar_index=bar_index,
            event_type="MARKET",
            markets={symbol: SimpleNamespace(
                close=close, open=close, high=close + 5, low=close - 5,
                volume=100.0,
            )},
            account=SimpleNamespace(balance=Decimal("10000"), equity=Decimal("10000")),
            positions={},
            features={},
        )
    return _make
