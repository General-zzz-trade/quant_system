"""Tests for polymarket.runner.PolymarketMakerRunner."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from polymarket.config import PolymarketConfig
from polymarket.runner import PolymarketMakerRunner, _current_window_ts, _WINDOW_5M
from polymarket.strategies.maker_5m import QuotePair


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return PolymarketConfig(
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://clob.polymarket.com",
    )


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.create_order.return_value = {"orderID": "order-123"}
    client.cancel_order.return_value = {"status": "cancelled"}
    client.get_orderbook.return_value = {
        "bids": [{"price": "0.48", "size": "100"}],
        "asks": [{"price": "0.52", "size": "100"}],
    }
    return client


@pytest.fixture
def mock_feed():
    feed = MagicMock()
    # 30 close prices that oscillate -> neutral RSI (~50)
    feed.refresh.return_value = [100.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(30)]
    feed.get_latest_price.return_value = 103.0
    return feed


@pytest.fixture
def runner(config, mock_client, mock_feed):
    r = PolymarketMakerRunner(
        config,
        client=mock_client,
        binance_feed=mock_feed,
        gamma=0.1,
        kappa=1.5,
        order_size=10.0,
        max_inventory=100.0,
        refresh_interval=5.0,
    )
    return r


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------

class TestMarketDiscovery:

    def test_discover_market_returns_dict(self, runner):
        """Successful market discovery returns dict with required keys."""
        fake_response = [{
            "title": "BTC Up or Down 5m",
            "closed": False,
            "markets": [{
                "tokens": [
                    {"outcome": "Up", "price": 0.52, "token_id": "up-tok-1"},
                    {"outcome": "Down", "price": 0.48, "token_id": "down-tok-1"},
                ]
            }],
        }]
        import json

        class FakeResp:
            def read(self):
                return json.dumps(fake_response).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with patch("polymarket.runner.urlopen", return_value=FakeResp()):
            market = runner.discover_market()

        assert market is not None
        assert market["up_token_id"] == "up-tok-1"
        assert market["down_token_id"] == "down-tok-1"
        assert "slug" in market
        assert market["closed"] is False

    def test_discover_market_returns_none_on_empty(self, runner):
        """Empty API response returns None."""

        class FakeResp:
            def read(self):
                return b"[]"
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with patch("polymarket.runner.urlopen", return_value=FakeResp()):
            market = runner.discover_market()

        assert market is None

    def test_discover_market_returns_none_on_network_error(self, runner):
        """Network error returns None gracefully."""
        with patch("polymarket.runner.urlopen", side_effect=OSError("timeout")):
            market = runner.discover_market()
        assert market is None

    def test_discover_market_missing_token_ids(self, runner):
        """Market without token IDs returns None."""
        import json

        class FakeResp:
            def read(self):
                return json.dumps([{
                    "title": "BTC Up or Down",
                    "closed": False,
                    "markets": [{"tokens": []}],
                }]).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with patch("polymarket.runner.urlopen", return_value=FakeResp()):
            assert runner.discover_market() is None


# ---------------------------------------------------------------------------
# Quote computation
# ---------------------------------------------------------------------------

class TestQuoteComputation:

    def test_quotes_symmetric_zero_inventory(self, runner):
        """With zero inventory, quotes should be roughly symmetric around mid."""
        quotes = runner._maker.compute_quotes(
            mid_price=0.50, inventory=0, volatility=0.05, time_remaining=0.5,
        )
        assert quotes.bid < 0.50
        assert quotes.ask > 0.50
        assert abs(quotes.mid - 0.50) < 0.02

    def test_quotes_shift_with_inventory(self, runner):
        """Long inventory shifts reservation price down."""
        q_zero = runner._maker.compute_quotes(
            mid_price=0.50, inventory=0, volatility=0.05, time_remaining=0.5,
        )
        q_long = runner._maker.compute_quotes(
            mid_price=0.50, inventory=50, volatility=0.05, time_remaining=0.5,
        )
        assert q_long.bid < q_zero.bid
        assert q_long.ask < q_zero.ask

    def test_rsi_bias_shifts_quotes(self, runner):
        """RSI UP signal shifts bid and ask higher."""
        base = runner._maker.compute_quotes(
            mid_price=0.50, inventory=0, volatility=0.05, time_remaining=0.5,
        )
        biased = runner._maker.apply_signal_bias(base, rsi_signal=1, bias_bps=0.01)
        assert biased.bid >= base.bid
        assert biased.ask >= base.ask


# ---------------------------------------------------------------------------
# Inventory management
# ---------------------------------------------------------------------------

class TestInventoryLimits:

    def test_both_sides_quoted_at_low_utilization(self, runner):
        """With low inventory, both sides should be quoted."""
        runner._inventory.update(yes_qty=10, no_qty=10)
        assert runner._inventory.should_quote_side("yes") is True
        assert runner._inventory.should_quote_side("no") is True

    def test_high_utilization_blocks_increasing_side(self, runner):
        """High YES inventory blocks further YES buying."""
        runner._inventory.update(yes_qty=90, no_qty=0)
        # util = 90/100 = 0.9 > warn_pct(0.8)
        assert runner._inventory.should_quote_side("yes") is False  # net>0, buying YES increases
        assert runner._inventory.should_quote_side("no") is True    # buying NO reduces

    def test_expiry_cancel_all(self, runner):
        """Very close to expiry -> cancel_all action."""
        action = runner._inventory.time_to_expiry_action(10)
        assert action == "cancel_all"

    def test_expiry_reduce_only(self, runner):
        """Between 60-120s from expiry -> reduce_only."""
        action = runner._inventory.time_to_expiry_action(90)
        assert action == "reduce_only"


# ---------------------------------------------------------------------------
# Order management
# ---------------------------------------------------------------------------

class TestOrderManagement:

    def test_place_quotes_creates_orders(self, runner, mock_client):
        """Place quotes should call create_order for bid and ask."""
        quotes = QuotePair(bid=0.48, ask=0.52, bid_size=10, ask_size=10)
        ids = runner.place_quotes("tok-1", quotes, quote_bid=True, quote_ask=True)
        assert len(ids) == 2
        assert mock_client.create_order.call_count == 2

    def test_place_quotes_bid_only(self, runner, mock_client):
        """Place only bid when ask is blocked."""
        quotes = QuotePair(bid=0.48, ask=0.52, bid_size=10, ask_size=10)
        ids = runner.place_quotes("tok-1", quotes, quote_bid=True, quote_ask=False)
        assert len(ids) == 1
        assert mock_client.create_order.call_count == 1
        call_args = mock_client.create_order.call_args
        assert call_args[1]["side"] == "BUY"

    def test_place_quotes_ask_only(self, runner, mock_client):
        """Place only ask when bid is blocked."""
        quotes = QuotePair(bid=0.48, ask=0.52, bid_size=10, ask_size=10)
        ids = runner.place_quotes("tok-1", quotes, quote_bid=False, quote_ask=True)
        assert len(ids) == 1
        call_args = mock_client.create_order.call_args
        assert call_args[1]["side"] == "SELL"

    def test_cancel_active_orders(self, runner, mock_client):
        """Cancel should call cancel_order for each active order."""
        runner._active_order_ids = ["ord-1", "ord-2", "ord-3"]
        cancelled = runner.cancel_active_orders()
        assert cancelled == 3
        assert mock_client.cancel_order.call_count == 3
        assert runner._active_order_ids == []

    def test_cancel_empty_is_noop(self, runner, mock_client):
        """Cancel with no active orders is a no-op."""
        cancelled = runner.cancel_active_orders()
        assert cancelled == 0
        assert mock_client.cancel_order.call_count == 0

    def test_cancel_partial_failure(self, runner, mock_client):
        """Partial cancel failure still clears the list."""
        runner._active_order_ids = ["ord-1", "ord-2"]
        mock_client.cancel_order.side_effect = [
            {"status": "cancelled"},
            Exception("timeout"),
        ]
        cancelled = runner.cancel_active_orders()
        assert cancelled == 1
        assert runner._active_order_ids == []


# ---------------------------------------------------------------------------
# Full cycle (run_once)
# ---------------------------------------------------------------------------

class TestRunOnce:

    def _make_market_response(self):
        import json
        data = [{
            "title": "BTC Up or Down",
            "closed": False,
            "markets": [{
                "tokens": [
                    {"outcome": "Up", "price": 0.52, "token_id": "up-tok"},
                    {"outcome": "Down", "price": 0.48, "token_id": "down-tok"},
                ]
            }],
        }]

        class FakeResp:
            def read(self):
                return json.dumps(data).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
        return FakeResp()

    def test_run_once_full_cycle(self, runner, mock_client):
        """Full cycle: discover -> quote -> place orders."""
        # Mock time.time to be 10s into the current window (plenty of time left)
        window_ts = _current_window_ts()
        fake_time = float(window_ts + 10)
        with patch("polymarket.runner.urlopen", return_value=self._make_market_response()), \
             patch("polymarket.runner.time") as mock_time:
            mock_time.time.return_value = fake_time
            mock_time.sleep = time.sleep
            summary = runner.run_once()

        assert summary["status"] == "quoted"
        assert summary["orders_placed"] >= 1
        assert "bid" in summary
        assert "ask" in summary

    def test_run_once_no_market(self, runner, mock_client):
        """No market found -> status no_market."""

        class FakeResp:
            def read(self):
                return b"[]"
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with patch("polymarket.runner.urlopen", return_value=FakeResp()):
            summary = runner.run_once()

        assert summary["status"] == "no_market"
        assert summary["orders_placed"] == 0

    def test_run_once_increments_cycle(self, runner, mock_client):
        """Each run_once increments cycle counter."""
        assert runner.cycle_count == 0

        class FakeResp:
            def read(self):
                return b"[]"
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with patch("polymarket.runner.urlopen", return_value=FakeResp()):
            runner.run_once()
            runner.run_once()
        assert runner.cycle_count == 2


# ---------------------------------------------------------------------------
# RSI signal computation
# ---------------------------------------------------------------------------

class TestRSISignal:

    def test_neutral_rsi_returns_zero(self, runner, mock_feed):
        """Neutral RSI (close to 50) returns 0."""
        # Default mock data gives gently rising closes -> neutral RSI
        signal = runner.compute_rsi_signal()
        assert signal == 0

    def test_rsi_resets_on_new_window(self, runner, mock_feed):
        """RSI strategy resets when window changes."""
        runner._last_window_ts = 12345  # force different window
        runner.compute_rsi_signal()
        # After reset + feed, strategy should have processed data
        assert runner._rsi_strategy.bar_count > 0


# ---------------------------------------------------------------------------
# Orderbook snapshot
# ---------------------------------------------------------------------------

class TestOrderbookSnapshot:

    def test_orderbook_mid_price(self, runner, mock_client):
        """Orderbook snapshot extracts correct mid price."""
        snap = runner.get_orderbook_snapshot("tok-1")
        assert abs(snap["mid_price"] - 0.50) < 0.01
        assert snap["best_bid"] == 0.48
        assert snap["best_ask"] == 0.52

    def test_orderbook_fallback_on_error(self, runner, mock_client):
        """Orderbook failure returns safe defaults."""
        mock_client.get_orderbook.side_effect = Exception("timeout")
        snap = runner.get_orderbook_snapshot("tok-1")
        assert snap["mid_price"] == 0.5
        assert snap["volatility_est"] > 0


# ---------------------------------------------------------------------------
# Window timestamp utility
# ---------------------------------------------------------------------------

class TestWindowTimestamp:

    def test_current_window_ts_aligned(self):
        """Window timestamp is aligned to 5-minute boundary."""
        ts = _current_window_ts()
        assert ts % _WINDOW_5M == 0

    def test_current_window_ts_not_future(self):
        """Window timestamp is not in the future."""
        ts = _current_window_ts()
        assert ts <= time.time()
