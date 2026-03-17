"""Tests for HedgeRunner — BTC Long + ALT Short hedge strategy."""
from __future__ import annotations



class MockAdapter:
    def __init__(self, equity=1000.0):
        self.orders = []
        self._equity = equity

    def send_market_order(self, symbol, side, qty, reduce_only=False):
        self.orders.append({"symbol": symbol, "side": side, "qty": qty, "reduce_only": reduce_only})
        return {"orderId": f"mock_{len(self.orders)}", "status": "Filled", "retCode": 0}

    def get_balances(self):
        return {"USDT": type("B", (), {"total": self._equity, "available": self._equity})()}

    def get_ticker(self, symbol):
        # Return 0 lastPrice so _fetch_basket_prices skips appending
        return {"fundingRate": "0.0001", "lastPrice": 0}

    def close_position(self, symbol):
        self.orders.append({"symbol": symbol, "action": "close"})
        return {"status": "ok"}

    def get_positions(self, symbol=None):
        return []


def _make_hedge(adapter=None, dry_run=False, ma_window=5):
    if adapter is None:
        adapter = MockAdapter()
    from scripts.ops.hedge_runner import HedgeRunner
    hr = HedgeRunner(adapter, dry_run=dry_run, ma_window=ma_window)
    return hr, adapter


def _populate_prices(hr, n_btc, btc_price=50000.0, n_alt_symbols=4, alt_price=10.0):
    """Populate BTC and alt prices for warmup without CSV."""
    for _ in range(n_btc):
        hr._btc_prices.append(btc_price)
    # Populate enough alt symbols (need >= 3)
    alt_syms = list(hr._alt_prices.keys())[:n_alt_symbols]
    for sym in alt_syms:
        for _ in range(n_btc):
            hr._alt_prices[sym].append(alt_price)


class TestHedgeRunner:

    def test_warmup_phase(self):
        hr, _ = _make_hedge(ma_window=10)
        # Only 5 BTC bars, need 11
        _populate_prices(hr, n_btc=5)
        result = hr.on_bar("ETHUSDT", 2000.0)
        assert result is not None
        assert result["action"] == "warmup"

    def test_non_eth_returns_none(self):
        hr, _ = _make_hedge()
        result = hr.on_bar("BTCUSDT", 50000.0)
        assert result is None

    def test_btc_price_tracked(self):
        hr, _ = _make_hedge()
        hr.on_bar("BTCUSDT", 50000.0)
        assert 50000.0 in hr._btc_prices

    def test_alt_price_tracked(self):
        hr, _ = _make_hedge()
        hr.on_bar("ADAUSDT", 0.5)
        assert 0.5 in hr._alt_prices["ADAUSDT"]

    def test_insufficient_alt_data(self):
        hr, _ = _make_hedge(ma_window=5)
        # Enough BTC bars but only 2 alt symbols have data (need >= 3)
        _populate_prices(hr, n_btc=10, n_alt_symbols=0)
        # Only give 2 symbols data
        alt_syms = list(hr._alt_prices.keys())
        hr._alt_prices[alt_syms[0]] = [10.0] * 10
        hr._alt_prices[alt_syms[1]] = [10.0] * 10
        result = hr.on_bar("ETHUSDT", 2000.0)
        assert result is not None
        assert result["action"] == "insufficient_data"

    def test_short_signal_when_btc_outperforms(self):
        """When ALT/BTC ratio < MA, BTC outperforming -> should_short=True."""
        hr, _ = _make_hedge(dry_run=True, ma_window=5)
        # Historical: ratio was higher (alt_price=20 / btc=50000)
        for i in range(6):
            hr._btc_prices.append(50000.0)
        alt_syms = list(hr._alt_prices.keys())[:4]
        for sym in alt_syms:
            for i in range(6):
                hr._alt_prices[sym].append(20.0)  # historical ratio = 20/50000

        # Now: alts dropped significantly, BTC same -> current ratio below MA
        for sym in alt_syms:
            hr._alt_prices[sym].append(5.0)  # current ratio = 5/50000 << MA of ~20/50000
        hr._btc_prices.append(50000.0)

        result = hr.on_bar("ETHUSDT", 2000.0)
        assert result is not None
        assert result["action"] == "signal"
        assert bool(result["should_short"]) is True

    def test_no_signal_when_alts_outperform(self):
        """When ALT/BTC ratio > MA, ALTs outperforming -> should_short=False."""
        hr, _ = _make_hedge(dry_run=True, ma_window=5)
        # Historical: ratio was lower
        for i in range(6):
            hr._btc_prices.append(50000.0)
        alt_syms = list(hr._alt_prices.keys())[:4]
        for sym in alt_syms:
            for i in range(6):
                hr._alt_prices[sym].append(10.0)  # historical ratio = 10/50000

        # Now: alts rallied significantly, ratio above MA
        for sym in alt_syms:
            hr._alt_prices[sym].append(50.0)  # current ratio = 50/50000 >> MA of ~10/50000
        hr._btc_prices.append(50000.0)

        result = hr.on_bar("ETHUSDT", 2000.0)
        assert result is not None
        assert result["action"] == "signal"
        assert bool(result["should_short"]) is False

    def test_dry_run_no_orders(self):
        hr, adapter = _make_hedge(dry_run=True, ma_window=5)
        # Set up prices so signal fires (BTC outperforming - alts crash)
        for i in range(6):
            hr._btc_prices.append(50000.0)
        alt_syms = list(hr._alt_prices.keys())[:4]
        for sym in alt_syms:
            for i in range(6):
                hr._alt_prices[sym].append(20.0)
            hr._alt_prices[sym].append(2.0)  # significant drop
        hr._btc_prices.append(50000.0)

        result = hr.on_bar("ETHUSDT", 2000.0)
        assert bool(result["should_short"]) is True
        assert result.get("trade") == "OPEN_HEDGE"
        # dry_run -> no orders sent
        assert len(adapter.orders) == 0

    def test_get_status(self):
        hr, _ = _make_hedge()
        status = hr.get_status()
        assert "active" in status
        assert "shorts" in status
        assert "trades" in status
        assert "bars" in status
        assert status["active"] is False

    def test_bars_processed_counter(self):
        hr, _ = _make_hedge(ma_window=5)
        _populate_prices(hr, n_btc=3)
        hr.on_bar("ETHUSDT", 2000.0)
        hr.on_bar("ETHUSDT", 2001.0)
        assert hr._bars_processed == 2

    def test_fetch_basket_prices_triggered_by_ethusdt(self):
        """on_bar for ETHUSDT/SUIUSDT/AXSUSDT triggers _fetch_basket_prices."""
        adapter = MockAdapter()
        # Override get_ticker to track calls
        ticker_calls = []
        orig = adapter.get_ticker

        def tracking_ticker(sym):
            ticker_calls.append(sym)
            return orig(sym)

        adapter.get_ticker = tracking_ticker
        hr, _ = _make_hedge(adapter=adapter, ma_window=5)
        _populate_prices(hr, n_btc=3)
        hr.on_bar("ETHUSDT", 2000.0)
        # Should have fetched BTC + all ALT basket tickers
        assert "BTCUSDT" in ticker_calls
