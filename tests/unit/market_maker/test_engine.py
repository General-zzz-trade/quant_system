"""Tests for MarketMakerEngine."""

import pytest
from unittest.mock import MagicMock
from execution.market_maker.config import MarketMakerConfig
from execution.market_maker.engine import MarketMakerEngine


def _make_snapshot(bid=2000.0, ask=2001.0, n_levels=5):
    """Create a mock OrderBookSnapshot."""
    mock = MagicMock()
    mock.best_bid = bid
    mock.best_ask = ask
    level = MagicMock()
    level.price = bid
    level.qty = 1.0
    mock.bids = [level] * n_levels
    ask_level = MagicMock()
    ask_level.price = ask
    ask_level.qty = 1.0
    mock.asks = [ask_level] * n_levels
    return mock


@pytest.fixture
def cfg():
    return MarketMakerConfig(dry_run=True, quote_update_interval_s=0.0)


@pytest.fixture
def engine(cfg):
    return MarketMakerEngine(cfg=cfg)


class TestMarketMakerEngine:
    def test_start_stop(self, engine):
        engine.start()
        assert engine.is_running
        engine.stop()
        assert not engine.is_running

    def test_on_depth_updates_state(self, engine):
        engine.start()
        snap = _make_snapshot(2000.0, 2001.0)
        engine.on_depth(snap)
        assert engine._mid == 2000.5
        assert engine._best_bid == 2000.0

    def test_on_trade_updates_vol(self, engine):
        engine.start()
        for i in range(25):
            engine.on_trade(2000.0 + i * 0.1, 0.01, "buy")
        assert engine._vol.ready

    def test_quotes_generated_after_warmup(self, engine):
        engine.start()
        # Warm up vol estimator
        for i in range(25):
            engine.on_trade(2000.0 + i * 0.1, 0.01, "buy")

        snap = _make_snapshot(2000.0, 2001.0)
        engine.on_depth(snap)

        # Check that order manager has quotes
        bid = engine._orders.live_bid
        ask = engine._orders.live_ask
        assert bid is not None or ask is not None

    def test_fill_processing(self, engine):
        engine.start()
        event = {
            "e": "ORDER_TRADE_UPDATE",
            "o": {
                "c": "mm_test123",
                "X": "FILLED",
                "S": "BUY",
                "l": "0.01",
                "L": "2000.0",
            },
        }
        engine.on_user_event(event)
        assert engine._inventory.net_qty == 0.01
        assert engine._inventory.total_fills == 1

    def test_kill_switch_cancels_and_stops(self, engine):
        engine.start()
        # Force daily loss
        engine._inventory.daily_pnl = -15.0
        engine._inventory._consecutive_losses = 0

        # Warm up and trigger quote cycle
        for i in range(25):
            engine.on_trade(2000.0 + i * 0.1, 0.01, "buy")
        engine.on_depth(_make_snapshot())

        assert not engine.is_running

    def test_set_funding_rate(self, engine):
        engine.set_funding_rate(0.0001)
        assert engine._funding_rate == 0.0001

    def test_invalid_depth_ignored(self, engine):
        engine.start()
        snap = _make_snapshot(0.0, 2001.0)
        engine.on_depth(snap)
        assert engine._mid == 0.0  # not updated

    def test_microstructure_integration(self):
        cfg = MarketMakerConfig(dry_run=True, quote_update_interval_s=0.0)
        micro = MagicMock()
        micro.on_trade.return_value = {"vpin": 0.5}
        micro.on_depth.return_value = {"vpin": 0.6}

        engine = MarketMakerEngine(cfg=cfg, microstructure=micro)
        engine.start()
        engine.on_trade(2000.0, 0.01, "buy")
        assert engine._vpin == 0.5

        snap = _make_snapshot()
        engine.on_depth(snap)
        assert engine._vpin == 0.6

    def test_microstructure_trade_failure_logs_and_preserves_previous_vpin(self, caplog, monkeypatch):
        import execution.market_maker.engine as engine_mod

        cfg = MarketMakerConfig(dry_run=True, quote_update_interval_s=0.0)
        micro = MagicMock()
        micro.on_trade.side_effect = RuntimeError("trade micro failed")
        engine = MarketMakerEngine(cfg=cfg, microstructure=micro)
        engine._vpin = 0.42

        monkeypatch.setattr(engine_mod.time, "monotonic", lambda: 100.0)
        with caplog.at_level("WARNING", logger=engine_mod.log.name):
            engine.on_trade(2000.0, 0.01, "buy")
            engine.on_trade(2000.0, 0.01, "buy")

        assert engine._vpin == 0.42
        assert caplog.text.count("Microstructure trade update failed; keeping previous state") == 1

    def test_microstructure_depth_failure_logs_and_preserves_previous_vpin(self, caplog, monkeypatch):
        import execution.market_maker.engine as engine_mod

        cfg = MarketMakerConfig(dry_run=True, quote_update_interval_s=999.0)
        micro = MagicMock()
        micro.on_depth.side_effect = RuntimeError("depth micro failed")
        engine = MarketMakerEngine(cfg=cfg, microstructure=micro)
        engine._vpin = 0.77
        engine.start()

        monkeypatch.setattr(engine_mod.time, "monotonic", lambda: 100.0)
        with caplog.at_level("WARNING", logger=engine_mod.log.name):
            engine.on_depth(_make_snapshot())
            engine.on_depth(_make_snapshot())

        assert engine._vpin == 0.77
        assert caplog.text.count("Microstructure depth update failed; keeping previous state") == 1
