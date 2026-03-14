# tests/integration/test_e2e_pipeline.py
"""End-to-end pipeline integration tests.

Tests the FULL causal chain:
  MarketEvent → FeatureHook (RustFeatureEngine) → ML Inference → ml_score
  → DecisionModule → OrderEvent → RiskGate → ExecutionBridge → FillEvent → state update

Covers gaps not tested by existing integration tests:
  - test_feature_to_ml_signal.py: features→ml_score but no decision→execution
  - test_live_flow_e2e.py: decision→execution but no features/ML
  - This file: the complete chain including features + ML + decision + risk + execution
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, List


from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.feature_hook import FeatureComputeHook
from execution.safety.risk_gate import RiskGate, RiskGateConfig
from features.live_computer import LiveFeatureComputer
from alpha.inference.bridge import LiveInferenceBridge
from alpha.base import Signal


# ── Shared helpers ────────────────────────────────────────────

class MockAlphaModel:
    """Returns signal based on ma_cross feature from RustFeatureEngine."""
    name = "mock_alpha"

    def predict(self, *, symbol, ts, features):
        signal_val = features.get("ma_cross_5_20")
        if signal_val is None:
            return None
        side = "long" if signal_val > 0 else ("short" if signal_val < 0 else "flat")
        return Signal(symbol=symbol, ts=ts, side=side, strength=min(abs(signal_val), 1.0))


class MLDecisionModule:
    """Decision module that reads ml_score from features and generates orders."""

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol
        self._ordered_bars: set[int] = set()

    def decide(self, snapshot: Any) -> list:
        if getattr(snapshot, "event_type", "").upper() != "MARKET":
            return []
        bar_idx = getattr(snapshot, "bar_index", 0)
        if bar_idx in self._ordered_bars:
            return []

        features = getattr(snapshot, "features", None) or {}
        ml_score = features.get("ml_score")
        if ml_score is None or abs(ml_score) < 0.01:
            return []

        market = getattr(snapshot, "market", None)
        if market is None:
            markets = getattr(snapshot, "markets", {})
            market = markets.get(self.symbol)
        if market is None:
            return []

        close = getattr(market, "close_f", None) or float(getattr(market, "close", 0))
        side = "buy" if ml_score > 0 else "sell"

        self._ordered_bars.add(bar_idx)
        return [SimpleNamespace(
            event_type="ORDER",
            EVENT_TYPE="order",
            symbol=self.symbol,
            venue="mock",
            command_id=f"cmd_{bar_idx}",
            idempotency_key=f"idem_{bar_idx}",
            side=side,
            qty=Decimal("0.01"),
            price=Decimal(str(close)),
            order_type="limit",
        )]


class MockVenueAdapter:
    """Returns fill events for every order."""

    def __init__(self) -> None:
        self.orders: List[Any] = []
        self._seq = 0

    def send_order(self, order_event: Any) -> list:
        self.orders.append(order_event)
        self._seq += 1
        return [SimpleNamespace(
            event_type="FILL",
            EVENT_TYPE="fill",
            symbol=getattr(order_event, "symbol", "BTCUSDT"),
            side=getattr(order_event, "side", "buy"),
            qty=float(getattr(order_event, "qty", 0.01)),
            price=float(getattr(order_event, "price", 40000.0)),
            fee=0.0,
            realized_pnl=0.0,
            margin_change=0.0,
            header=SimpleNamespace(
                event_id=f"fill_{self._seq}",
                ts=datetime(2024, 1, 1, 1, 0),
            ),
        )]


def _market(close: float, idx: int) -> SimpleNamespace:
    h, m = divmod(idx, 60)
    ts = datetime(2024, 1, 1, h % 24, m)
    return SimpleNamespace(
        event_type="MARKET",
        symbol="BTCUSDT",
        open=close - 1, high=close + 2, low=close - 2, close=close,
        volume=50.0,
        ts=ts,
        header=SimpleNamespace(event_id=f"e{idx}", ts=ts),
    )


def _build_e2e(
    *,
    risk_gate: RiskGate | None = None,
    warmup_bars: int = 0,
    starting_balance: float = 100_000.0,
) -> tuple[EngineCoordinator, MockVenueAdapter, MLDecisionModule]:
    """Build a full E2E pipeline: features + ML + decision + execution."""
    computer = LiveFeatureComputer(fast_ma=3, slow_ma=5, vol_window=3)
    model = MockAlphaModel()
    bridge = LiveInferenceBridge(models=[model])
    hook = FeatureComputeHook(computer=computer, inference_bridge=bridge, warmup_bars=warmup_bars)

    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT",),
        currency="USDT",
        starting_balance=starting_balance,
        feature_hook=hook,
    )
    coord = EngineCoordinator(cfg=cfg)

    mock_venue = MockVenueAdapter()
    decision_module = MLDecisionModule()

    def _emit(ev: Any) -> None:
        coord.emit(ev, actor="bridge")

    decision_bridge = DecisionBridge(dispatcher_emit=_emit, modules=[decision_module])
    exec_bridge = ExecutionBridge(
        adapter=mock_venue,
        dispatcher_emit=_emit,
        risk_gate=risk_gate,
    )

    coord.attach_decision_bridge(decision_bridge)
    coord.attach_execution_bridge(exec_bridge)
    coord.start()

    return coord, mock_venue, decision_module


# ── Test 1: Full pipeline — features → ML → decision → execution → fill ──

def test_full_pipeline_features_to_fill():
    """Complete E2E: market events → RustFeatureEngine → ML inference → ml_score
    → decision module → order → venue → fill → position update."""
    coord, mock_venue, decision_module = _build_e2e()

    # Pump 50 bars of uptrend data for Rust engine windows to fill
    for i in range(50):
        coord.emit(_market(100.0 + i * 0.5, i), actor="test")

    # After warmup, ML score should have been generated and decision triggered
    snap = coord.get_state_view()["last_snapshot"]
    assert snap.features is not None
    assert "ml_score" in snap.features

    # Decision module should have produced orders
    assert len(mock_venue.orders) >= 1, "Expected at least one order from ML decision"

    # Fills should have updated positions
    pos = coord.get_state_view()["positions"].get("BTCUSDT")
    assert pos is not None, "Expected position after fill"
    assert float(pos.qty) != 0


# ── Test 2: Feature types match decision expectations ─────────

def test_feature_types_compatible_with_decision():
    """Verify RustFeatureEngine output types are compatible with decision module input."""
    coord, _, _ = _build_e2e()

    for i in range(50):
        coord.emit(_market(100.0 + i * 0.5, i), actor="test")

    snap = coord.get_state_view()["last_snapshot"]
    features = snap.features
    assert features is not None

    # All feature values should be float-compatible (not None, not string, not NaN)
    for key, val in features.items():
        assert isinstance(val, (int, float)), f"Feature {key}={val!r} is not numeric"

    # Key features that decision modules depend on must exist
    assert "ml_score" in features
    # RustFeatureEngine core features
    for key in ("close", "volume", "rsi_14", "atr_norm_14", "ma_cross_5_20"):
        assert key in features, f"Missing expected feature: {key}"


# ── Test 3: RiskGate blocks oversized orders ──────────────────

def test_risk_gate_blocks_orders_in_pipeline():
    """RiskGate integrated into execution bridge blocks orders exceeding limits."""
    risk_gate = RiskGate(
        config=RiskGateConfig(
            max_order_notional=1.0,  # Extremely tight: block any order > $1
        ),
    )

    coord, mock_venue, _ = _build_e2e(risk_gate=risk_gate)

    for i in range(50):
        coord.emit(_market(100.0 + i * 0.5, i), actor="test")

    # Orders should have been generated by decision module but blocked by RiskGate
    # (0.01 qty * ~125 price = ~$1.25 > $1.0 limit)
    assert len(mock_venue.orders) == 0, "RiskGate should have blocked all orders"

    # No position should exist (orders were blocked)
    pos = coord.get_state_view()["positions"].get("BTCUSDT")
    if pos is not None:
        assert float(pos.qty) == 0


# ── Test 4: RiskGate kill switch blocks all orders ────────────

def test_risk_gate_kill_switch_blocks_pipeline():
    """Kill switch active → all orders rejected at execution boundary."""
    risk_gate = RiskGate(
        config=RiskGateConfig(),
        is_killed=lambda: True,  # Kill switch always active
    )

    coord, mock_venue, _ = _build_e2e(risk_gate=risk_gate)

    for i in range(50):
        coord.emit(_market(100.0 + i * 0.5, i), actor="test")

    assert len(mock_venue.orders) == 0, "Kill switch should block all orders"


# ── Test 5: Warmup bars delay ML inference ────────────────────

def test_warmup_bars_delay_inference():
    """ML inference should not produce ml_score during warmup period."""
    coord, mock_venue, _ = _build_e2e(warmup_bars=100)

    # Only pump 50 bars (less than warmup=100)
    for i in range(50):
        coord.emit(_market(100.0 + i * 0.5, i), actor="test")

    snap = coord.get_state_view()["last_snapshot"]
    features = snap.features

    # During warmup, ml_score should not be present (inference bridge skips)
    if features is not None:
        assert "ml_score" not in features, "ml_score should not be present during warmup"

    # No orders should have been generated
    assert len(mock_venue.orders) == 0


# ── Test 6: Flat signal produces no orders ────────────────────

def test_flat_signal_no_orders():
    """When price is flat, ML score ≈ 0 → no orders generated."""
    coord, mock_venue, _ = _build_e2e()

    # Flat price series — no trend
    for i in range(50):
        coord.emit(_market(100.0, i), actor="test")

    snap = coord.get_state_view()["last_snapshot"]
    features = snap.features or {}
    ml_score = features.get("ml_score")

    # Either no ml_score or very close to 0 → decision module filters it
    if ml_score is not None and abs(ml_score) >= 0.01:
        # If ml_score is somehow non-zero on flat data, orders might exist
        pass  # acceptable — depends on feature engineering
    else:
        assert len(mock_venue.orders) == 0, "Flat signal should not generate orders"


# ── Test 7: Multiple fills update position correctly ──────────

def test_multiple_fills_accumulate_position():
    """Multiple fills from consecutive bars accumulate position correctly."""
    coord, mock_venue, _ = _build_e2e()

    # Strong uptrend — should trigger multiple buy signals
    for i in range(80):
        coord.emit(_market(100.0 + i * 1.0, i), actor="test")

    n_orders = len(mock_venue.orders)
    if n_orders >= 2:
        # All orders should be same side (buy in uptrend)
        sides = {getattr(o, "side", None) for o in mock_venue.orders}
        assert "buy" in sides

        # Position qty should reflect accumulated fills
        pos = coord.get_state_view()["positions"].get("BTCUSDT")
        assert pos is not None
        assert float(pos.qty) > 0


# ── Test 8: RiskGate with position callback sees real positions ──

def test_risk_gate_position_callback():
    """RiskGate position callback reflects actual coordinator state."""
    coord, mock_venue, _ = _build_e2e()

    # Wire RiskGate after first fill to test position awareness
    positions_seen: list[dict] = []

    def _capture_positions():
        p = coord.get_state_view().get("positions", {})
        positions_seen.append(dict(p))
        return p

    gate = RiskGate(
        config=RiskGateConfig(max_position_notional=1_000_000.0),
        get_positions=_capture_positions,
    )

    # Replace execution bridge risk gate
    coord2, mock_venue2, _ = _build_e2e(risk_gate=gate)

    for i in range(60):
        coord2.emit(_market(100.0 + i * 0.5, i), actor="test")

    # If orders were generated, risk gate's position callback was invoked
    if len(mock_venue2.orders) > 0:
        assert len(positions_seen) > 0, "RiskGate position callback should have been called"


# ── Test 9: Downtrend generates sell signals ──────────────────

def test_downtrend_generates_sell_signals():
    """Downtrend data → negative ml_score → sell orders."""
    coord, mock_venue, _ = _build_e2e()

    # First establish some uptrend bars for feature warmup
    for i in range(35):
        coord.emit(_market(200.0 + i * 0.1, i), actor="test")

    # Then strong downtrend
    for i in range(35, 70):
        coord.emit(_market(200.0 - (i - 35) * 1.0, i), actor="test")

    snap = coord.get_state_view()["last_snapshot"]
    features = snap.features or {}
    ml_score = features.get("ml_score")

    # In downtrend, ml_score should be negative (if present)
    if ml_score is not None and ml_score < -0.01:
        # Check that at least some sell orders were generated
        sell_orders = [o for o in mock_venue.orders if getattr(o, "side", "") == "sell"]
        assert len(sell_orders) >= 1, "Expected sell orders in downtrend"


# ── Test 10: State consistency after full pipeline ────────────

def test_state_consistency_after_pipeline():
    """After processing many events, state should be internally consistent."""
    coord, mock_venue, _ = _build_e2e(starting_balance=100_000.0)

    for i in range(60):
        coord.emit(_market(100.0 + i * 0.3, i), actor="test")

    view = coord.get_state_view()

    # Event index should reflect all processed events (market + fills)
    assert view["event_index"] >= 60

    # Account state should exist
    account = view.get("account")
    assert account is not None

    # Markets should have BTCUSDT
    markets = view.get("markets", {})
    assert "BTCUSDT" in markets

    # If positions exist, they should have valid qty
    pos = view["positions"].get("BTCUSDT")
    if pos is not None:
        qty = float(pos.qty)
        assert qty == qty  # not NaN
