"""Integration: pipeline state consistency across event sequences."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from engine.coordinator import CoordinatorConfig, EngineCoordinator


def _market(close: float, idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="MARKET",
        symbol="BTCUSDT",
        open=close, high=close + 1, low=close - 1, close=close,
        volume=50.0,
        ts=datetime(2024, 1, 1, 0, idx),
        header=SimpleNamespace(event_id=f"e{idx}", ts=datetime(2024, 1, 1, 0, idx)),
    )


def test_event_index_monotonic():
    """Event index increases monotonically with each fact event."""
    cfg = CoordinatorConfig(symbol_default="BTCUSDT", symbols=("BTCUSDT",), currency="USDT")
    coord = EngineCoordinator(cfg=cfg)
    coord.start()

    indices = []
    for i in range(10):
        coord.emit(_market(40000.0 + i, i), actor="test")
        indices.append(coord.get_state_view()["event_index"])

    assert indices == list(range(1, 11))


def test_snapshot_reflects_latest_state():
    """Last snapshot matches the current market state."""
    cfg = CoordinatorConfig(symbol_default="BTCUSDT", symbols=("BTCUSDT",), currency="USDT")
    coord = EngineCoordinator(cfg=cfg)
    coord.start()

    for i in range(5):
        coord.emit(_market(40000.0 + i * 100, i), actor="test")

    view = coord.get_state_view()
    snap = view["last_snapshot"]
    assert snap is not None
    assert snap.markets["BTCUSDT"].close == view["markets"]["BTCUSDT"].close


def test_features_in_snapshot():
    """Features computed by FeatureComputeHook appear in snapshot."""
    from engine.feature_hook import FeatureComputeHook
    from features.live_computer import LiveFeatureComputer

    computer = LiveFeatureComputer(fast_ma=3, slow_ma=5, vol_window=3)
    hook = FeatureComputeHook(computer=computer)

    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT",),
        currency="USDT",
        feature_hook=hook,
    )
    coord = EngineCoordinator(cfg=cfg)
    coord.start()

    for i in range(40):
        coord.emit(_market(40000.0 + i * 10, i), actor="test")

    snap = coord.get_state_view()["last_snapshot"]
    assert snap.features is not None
    assert "close" in snap.features
    # Rust engine produces ma_cross features after enough bars
    assert "ma_cross_10_30" in snap.features or "ma_cross_5_20" in snap.features
