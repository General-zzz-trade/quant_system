# tests/unit/bybit/test_ws_health.py
"""Tests for WebSocket health monitoring (stale detection, last_message_time)."""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

from execution.adapters.bybit.ws_client import BybitWsClient


class TestLastMessageTime:
    def test_initial_zero(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        assert ws.last_message_time == 0.0

    def test_seconds_since_inf_initially(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        assert ws.seconds_since_last_message == float("inf")

    def test_updated_on_message(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        before = time.time()
        # Simulate a WS message
        ws._on_message(None, json.dumps({"topic": "ping"}))
        after = time.time()
        assert before <= ws.last_message_time <= after

    def test_seconds_since_after_message(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        ws._on_message(None, json.dumps({"topic": "ping"}))
        elapsed = ws.seconds_since_last_message
        assert 0 <= elapsed < 1.0

    def test_updated_on_kline_message(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        kline_msg = json.dumps({
            "topic": "kline.60.ETHUSDT",
            "data": [{
                "confirm": True, "start": "1700000000000",
                "open": "2000", "high": "2010", "low": "1990",
                "close": "2005", "volume": "100", "turnover": "200000",
            }],
        })
        ws._on_bar = MagicMock()
        ws._on_message(None, kline_msg)
        assert ws.last_message_time > 0

    def test_updated_on_ticker_message(self):
        ws = BybitWsClient(symbols=["ETHUSDT"], on_tick=MagicMock())
        ticker_msg = json.dumps({
            "topic": "tickers.ETHUSDT",
            "data": {"symbol": "ETHUSDT", "lastPrice": "2000"},
        })
        ws._on_message(None, ticker_msg)
        assert ws.last_message_time > 0

    def test_updated_on_invalid_json(self):
        """Even invalid JSON updates timestamp (message was received)."""
        ws = BybitWsClient(symbols=["ETHUSDT"])
        ws._on_message(None, "not json")
        # _on_message sets time before parsing, so it should be updated
        assert ws.last_message_time > 0


class TestReconnectCount:
    def test_initial_zero(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        assert ws._reconnect_count == 0

    def test_increments(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        ws._reconnect_count += 1
        assert ws._reconnect_count == 1

    def test_reset_to_zero(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        ws._reconnect_count = 5
        ws._reconnect_count = 0
        assert ws._reconnect_count == 0


class TestStaleDetectionLogic:
    """Test the stale detection logic used in run_bybit_alpha.py."""

    def test_stale_when_no_messages(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        # Never received a message -> inf seconds
        assert ws.seconds_since_last_message > 120

    def test_not_stale_after_recent_message(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        ws._on_message(None, json.dumps({"topic": "ping"}))
        assert ws.seconds_since_last_message < 120

    def test_stale_after_timeout(self):
        ws = BybitWsClient(symbols=["ETHUSDT"])
        # Simulate a message from 200 seconds ago
        ws._last_message_time = time.time() - 200
        assert ws.seconds_since_last_message > 120


class TestAlphaRunnerDataStale:
    """Test AlphaRunner.seconds_since_last_bar property."""

    def test_seconds_since_last_bar_inf_initially(self):
        """Before any bar is processed, should return inf."""
        import sys
        from unittest.mock import MagicMock as MM
        if "_quant_hotpath" not in sys.modules:
            sys.modules["_quant_hotpath"] = MM()
        from scripts.ops.alpha_runner import AlphaRunner

        adapter = MM()
        adapter.get_ticker.return_value = {}
        model_info = {
            "model": MM(predict=MM(return_value=[0.0])),
            "features": ["close"],
            "horizon_models": [],
            "config": {"version": "v11"},
            "deadzone": 0.3, "min_hold": 18, "max_hold": 60,
            "zscore_window": 720, "zscore_warmup": 180,
        }
        runner = AlphaRunner(adapter=adapter, model_info=model_info,
                             symbol="ETHUSDT", dry_run=True)
        assert runner.seconds_since_last_bar == float("inf")

    def test_seconds_since_last_bar_after_process(self):
        """After process_bar, should return small value."""
        import sys
        from unittest.mock import MagicMock as MM
        if "_quant_hotpath" not in sys.modules:
            sys.modules["_quant_hotpath"] = MM()
        from scripts.ops.alpha_runner import AlphaRunner

        adapter = MM()
        adapter.get_ticker.return_value = {"fundingRate": "0.0001"}
        adapter.get_positions.return_value = []
        model_info = {
            "model": MM(predict=MM(return_value=[0.0])),
            "features": ["close"],
            "horizon_models": [],
            "config": {"version": "v11"},
            "deadzone": 0.3, "min_hold": 18, "max_hold": 60,
            "zscore_window": 720, "zscore_warmup": 180,
        }
        runner = AlphaRunner(adapter=adapter, model_info=model_info,
                             symbol="ETHUSDT", dry_run=True)
        # Mock engine to return empty features (no_features action)
        runner._engine = MM()
        runner._engine.get_features.return_value = []
        # Patch the OI cache (replaces former _fetch_binance_oi_data inline call)
        runner._oi_cache = MM()
        runner._oi_cache.get.return_value = {
            "open_interest": 0, "ls_ratio": float("nan"),
            "top_trader_ls_ratio": float("nan"),
            "taker_buy_vol": float("nan"), "taker_sell_vol": float("nan"),
        }
        runner.process_bar({"open": 100, "high": 101, "low": 99,
                            "close": 100, "volume": 1000})
        assert runner.seconds_since_last_bar < 1.0
