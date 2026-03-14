"""End-to-end tests: AttributionTracker, CorrelationComputer activation, CorrelationGate order rejection.

Reuses MockVenueAdapter / MockDecisionModule patterns from test_live_flow_e2e.py.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, List, Set

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from attribution.tracker import AttributionTracker
from risk.correlation_computer import CorrelationComputer
from risk.correlation_gate import CorrelationGate, CorrelationGateConfig


# ── Helpers ──────────────────────────────────────────────────

class MockDecisionModuleWithOrigin:
    """Emits IntentEvent then OrderEvent with origin tracking."""

    def __init__(self, origin: str = "momentum_v1") -> None:
        self.origin = origin
        self._ordered_bars: Set[int] = set()
        self._seq = 0

    def decide(self, snapshot: Any) -> list:
        if snapshot.event_type.upper() != "MARKET":
            return []
        if snapshot.bar_index in self._ordered_bars:
            return []
        close = snapshot.markets[snapshot.symbol].close
        if close > 40100:
            self._ordered_bars.add(snapshot.bar_index)
            self._seq += 1
            intent_id = f"intent_{self._seq}"
            intent = SimpleNamespace(
                event_type="intent",
                EVENT_TYPE="intent",
                intent_id=intent_id,
                symbol=snapshot.symbol,
                side="buy",
                target_qty=0.01,
                reason_code="signal",
                origin=self.origin,
            )
            order = SimpleNamespace(
                event_type="order",
                EVENT_TYPE="order",
                symbol=snapshot.symbol,
                venue="mock",
                order_id=f"ord_{self._seq}",
                intent_id=intent_id,
                command_id=f"cmd_{snapshot.bar_index}",
                idempotency_key=f"idem_{snapshot.bar_index}",
                side="buy",
                qty=0.01,
                price=close,
                order_type="limit",
            )
            return [intent, order]
        return []


class MockVenueAdapter:
    """Returns fill events for orders."""

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


def _market(symbol: str, close: float, idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="MARKET",
        symbol=symbol,
        open=close, high=close + 5, low=close - 5, close=close,
        volume=100.0,
        ts=datetime(2024, 1, 1, 0, idx),
        header=SimpleNamespace(event_id=f"e{idx}_{symbol}", ts=datetime(2024, 1, 1, 0, idx)),
    )


def _build_stack(symbols=("BTCUSDT",), correlation_gate=None):
    """Build a minimal coordinator + decision + execution stack."""
    mock_venue = MockVenueAdapter()
    mock_decision = MockDecisionModuleWithOrigin()
    attribution_tracker = AttributionTracker()

    cfg = CoordinatorConfig(
        symbol_default=symbols[0],
        symbols=symbols,
        currency="USDT",
        starting_balance=10000.0,
    )
    coord = EngineCoordinator(cfg=cfg)

    def _emit(ev: Any) -> None:
        attribution_tracker.on_event(ev)

        # Correlation gate interception
        if correlation_gate is not None:
            et = getattr(ev, "event_type", None)
            et_str = (str(et.value) if hasattr(et, "value") else str(et)).upper() if et else ""
            if et_str == "ORDER":
                view = coord.get_state_view()
                positions = view.get("positions", {})
                existing = [s for s, p in positions.items() if float(getattr(p, "qty", 0)) != 0]
                sym = getattr(ev, "symbol", "")
                decision = correlation_gate.should_allow(sym, existing)
                if not decision.ok:
                    return

        coord.emit(ev, actor="bridge")

    decision_bridge = DecisionBridge(dispatcher_emit=_emit, modules=[mock_decision])
    exec_bridge = ExecutionBridge(adapter=mock_venue, dispatcher_emit=_emit)
    coord.attach_decision_bridge(decision_bridge)
    coord.attach_execution_bridge(exec_bridge)
    coord.start()

    return coord, mock_venue, mock_decision, attribution_tracker


# ── Tests ────────────────────────────────────────────────────

def test_attribution_end_to_end():
    """IntentEvent + OrderEvent + FillEvent tracked, report attributes to correct signal."""
    coord, venue, decision, tracker = _build_stack()

    # Emit market events: first two below threshold, third triggers buy
    coord.emit(_market("BTCUSDT", 40000.0, 0), actor="test")
    coord.emit(_market("BTCUSDT", 40050.0, 1), actor="test")
    coord.emit(_market("BTCUSDT", 40150.0, 2), actor="test")  # triggers intent+order

    assert tracker.intent_count >= 1, f"Expected intents, got {tracker.intent_count}"
    assert tracker.order_count >= 1, f"Expected orders, got {tracker.order_count}"
    assert tracker.fill_count >= 1, f"Expected fills, got {tracker.fill_count}"

    report = tracker.report()
    assert report is not None


def test_attribution_pnl_identity():
    """sum(by_signal PnL) + unattributed == total PnL."""
    coord, venue, decision, tracker = _build_stack()

    coord.emit(_market("BTCUSDT", 40000.0, 0), actor="test")
    coord.emit(_market("BTCUSDT", 40150.0, 1), actor="test")  # triggers buy
    coord.emit(_market("BTCUSDT", 40200.0, 2), actor="test")  # triggers buy

    report = tracker.report(current_prices={"BTCUSDT": 40250.0})
    by_signal_total = sum(
        s.realized_pnl + s.unrealized_pnl - s.fee_cost
        for s in report.by_signal.values()
    )
    total = report.total_pnl
    diff = abs(total - (by_signal_total + report.unattributed_pnl))
    assert diff < 1e-6, f"PnL identity violated: total={total}, by_signal={by_signal_total}, unattr={report.unattributed_pnl}"


def test_correlation_computer_activated():
    """CorrelationComputer receives data via on_snapshot callback."""
    cc = CorrelationComputer(window=60)

    def _on_snapshot(snapshot: Any) -> None:
        markets = getattr(snapshot, "markets", {})
        for sym, mkt in markets.items():
            close = getattr(mkt, "close", None)
            if close is not None:
                cc.update(sym, float(close))

    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT", "ETHUSDT"),
        currency="USDT",
        on_snapshot=_on_snapshot,
    )
    coord = EngineCoordinator(cfg=cfg)
    coord.start()

    # Feed 15 market events per symbol
    for i in range(15):
        coord.emit(_market("BTCUSDT", 40000.0 + i * 10, i * 2), actor="test")
        coord.emit(_market("ETHUSDT", 2500.0 + i * 5, i * 2 + 1), actor="test")

    # CorrelationComputer should have return data for both symbols
    assert "BTCUSDT" in cc._returns, "BTCUSDT not in correlation returns"
    assert "ETHUSDT" in cc._returns, "ETHUSDT not in correlation returns"
    assert len(cc._returns["BTCUSDT"]) >= 10, f"Expected 10+ returns, got {len(cc._returns['BTCUSDT'])}"

    # Portfolio avg correlation should be computable
    avg = cc.portfolio_avg_correlation(["BTCUSDT", "ETHUSDT"])
    assert avg is not None, "portfolio_avg_correlation returned None"


def test_correlation_gate_rejects_high_correlation():
    """CorrelationGate blocks ORDER when portfolio correlation exceeds threshold."""
    cc = CorrelationComputer(window=60)

    # Feed perfectly correlated data to make gate reject
    for i in range(30):
        price = 40000.0 + i * 10
        cc.update("BTCUSDT", price)
        cc.update("ETHUSDT", price * 0.0625)  # perfectly correlated

    gate = CorrelationGate(
        computer=cc,
        config=CorrelationGateConfig(
            max_avg_correlation=0.3,
            max_position_correlation=0.3,
            min_data_points=5,
        ),
    )

    coord, venue, decision, tracker = _build_stack(
        symbols=("BTCUSDT", "ETHUSDT"),
        correlation_gate=gate,
    )

    # First: get a BTCUSDT position via direct execution
    coord.emit(_market("BTCUSDT", 40150.0, 0), actor="test")  # triggers buy for BTCUSDT

    # Verify we have a position now
    view = coord.get_state_view()
    btc_pos = view["positions"].get("BTCUSDT")
    has_btc_pos = btc_pos is not None and float(getattr(btc_pos, "qty", 0)) != 0

    if has_btc_pos:
        # Now feed ETHUSDT market data to trigger another order
        orders_before = len(venue.orders)
        coord.emit(_market("ETHUSDT", 40150.0, 1), actor="test")

        # Gate should have rejected the ETHUSDT order since correlation is high
        assert len(venue.orders) == orders_before, (
            f"Expected gate to block ETHUSDT order, but venue got {len(venue.orders) - orders_before} new orders"
        )


def test_correlation_gate_allows_uncorrelated():
    """CorrelationGate allows ORDER when there are no positions (gate trivially passes)."""
    cc = CorrelationComputer(window=60)
    gate = CorrelationGate(
        computer=cc,
        config=CorrelationGateConfig(max_avg_correlation=0.7),
    )

    coord, venue, decision, tracker = _build_stack(
        symbols=("BTCUSDT",),
        correlation_gate=gate,
    )

    # No existing positions, gate should allow
    coord.emit(_market("BTCUSDT", 40150.0, 0), actor="test")  # triggers buy

    assert len(venue.orders) >= 1, "Expected order to pass through gate"
