"""Tests for polymarket.strategies.maker_5m.AvellanedaStoikovMaker."""
from __future__ import annotations

import pytest

from polymarket.strategies.maker_5m import AvellanedaStoikovMaker, QuotePair


class TestQuotePair:

    def test_spread(self):
        q = QuotePair(bid=0.45, ask=0.55, bid_size=10, ask_size=10)
        assert abs(q.spread - 0.10) < 1e-9

    def test_mid(self):
        q = QuotePair(bid=0.40, ask=0.60, bid_size=10, ask_size=10)
        assert abs(q.mid - 0.50) < 1e-9


class TestAvellanedaStoikovMaker:

    def _maker(self, **kw):
        defaults = dict(gamma=0.1, kappa=1.5, max_inventory=100,
                        min_spread=0.02, max_spread=0.10, order_size=10)
        defaults.update(kw)
        return AvellanedaStoikovMaker(**defaults)

    def test_basic_quotes_symmetric(self):
        m = self._maker()
        q = m.compute_quotes(mid_price=0.50, inventory=0, volatility=0.05,
                             time_remaining=0.5)
        # Zero inventory -> reservation = mid -> quotes symmetric around 0.50
        assert q.bid < 0.50
        assert q.ask > 0.50
        assert abs(q.mid - 0.50) < 0.01

    def test_inventory_shifts_reservation(self):
        m = self._maker()
        q_long = m.compute_quotes(mid_price=0.50, inventory=50, volatility=0.05,
                                  time_remaining=0.5)
        q_short = m.compute_quotes(mid_price=0.50, inventory=-50, volatility=0.05,
                                   time_remaining=0.5)
        # Long inventory -> lower reservation -> lower bid/ask
        assert q_long.bid < q_short.bid
        assert q_long.ask < q_short.ask

    def test_prices_clamped_to_binary_range(self):
        m = self._maker()
        q = m.compute_quotes(mid_price=0.02, inventory=100, volatility=0.5,
                             time_remaining=1.0)
        assert q.bid >= 0.01
        assert q.ask <= 0.99
        assert q.bid < q.ask

    def test_prices_clamped_high(self):
        m = self._maker()
        q = m.compute_quotes(mid_price=0.98, inventory=-100, volatility=0.5,
                             time_remaining=1.0)
        assert q.ask <= 0.99
        assert q.bid >= 0.01

    def test_min_spread_enforced(self):
        m = self._maker(min_spread=0.05)
        q = m.compute_quotes(mid_price=0.50, inventory=0, volatility=0.001,
                             time_remaining=0.01)
        assert q.spread >= 0.049  # allow rounding

    def test_max_spread_enforced(self):
        m = self._maker(max_spread=0.08)
        q = m.compute_quotes(mid_price=0.50, inventory=0, volatility=1.0,
                             time_remaining=1.0)
        assert q.spread <= 0.08 + 0.001

    def test_signal_bias_up(self):
        m = self._maker()
        base = m.compute_quotes(mid_price=0.50, inventory=0, volatility=0.05,
                                time_remaining=0.5)
        biased = m.apply_signal_bias(base, rsi_signal=1, bias_bps=0.01)
        # UP signal -> bid should increase
        assert biased.bid > base.bid or abs(biased.bid - 0.99) < 0.001

    def test_signal_bias_down(self):
        m = self._maker()
        base = m.compute_quotes(mid_price=0.50, inventory=0, volatility=0.05,
                                time_remaining=0.5)
        biased = m.apply_signal_bias(base, rsi_signal=-1, bias_bps=0.01)
        # DOWN signal -> ask should decrease
        assert biased.ask < base.ask or abs(biased.ask - 0.01) < 0.001

    def test_signal_bias_zero_is_noop(self):
        m = self._maker()
        base = m.compute_quotes(mid_price=0.50, inventory=0, volatility=0.05,
                                time_remaining=0.5)
        biased = m.apply_signal_bias(base, rsi_signal=0)
        assert biased.bid == base.bid
        assert biased.ask == base.ask

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            AvellanedaStoikovMaker(gamma=0)

    def test_invalid_spread_raises(self):
        with pytest.raises(ValueError, match="max_spread"):
            AvellanedaStoikovMaker(min_spread=0.10, max_spread=0.05)

    def test_order_size_propagated(self):
        m = self._maker(order_size=25)
        q = m.compute_quotes(mid_price=0.50, inventory=0, volatility=0.05,
                             time_remaining=0.5)
        assert q.bid_size == 25
        assert q.ask_size == 25
