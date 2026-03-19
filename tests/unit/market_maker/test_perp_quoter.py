"""Tests for PerpQuoter (Avellaneda-Stoikov adapted for perps)."""

import pytest
from execution.market_maker.config import MarketMakerConfig
from execution.market_maker.perp_quoter import PerpQuoter


@pytest.fixture
def cfg():
    return MarketMakerConfig()


@pytest.fixture
def quoter(cfg):
    return PerpQuoter(cfg)


class TestPerpQuoter:
    def test_basic_quotes(self, quoter):
        q = quoter.compute_quotes(mid=2000.0, inventory=0.0, vol=0.001, time_remaining=0.5)
        assert q is not None
        assert q.bid < 2000.0
        assert q.ask > 2000.0
        assert q.bid < q.ask

    def test_symmetric_when_flat(self, quoter):
        q = quoter.compute_quotes(mid=2000.0, inventory=0.0, vol=0.001, time_remaining=0.5)
        assert q is not None
        # With zero inventory, quotes should be roughly symmetric around mid
        assert abs((q.bid + q.ask) / 2 - 2000.0) < 1.0

    def test_inventory_shifts_reservation(self, quoter):
        q_flat = quoter.compute_quotes(mid=2000.0, inventory=0.0, vol=0.001, time_remaining=0.5)
        q_long = quoter.compute_quotes(mid=2000.0, inventory=0.01, vol=0.001, time_remaining=0.5)
        assert q_flat is not None and q_long is not None
        # Long inventory → reservation price lower → bid/ask shift down
        assert q_long.reservation < q_flat.reservation

    def test_short_inventory_shifts_up(self, quoter):
        q_flat = quoter.compute_quotes(mid=2000.0, inventory=0.0, vol=0.001, time_remaining=0.5)
        q_short = quoter.compute_quotes(mid=2000.0, inventory=-0.01, vol=0.001, time_remaining=0.5)
        assert q_flat is not None and q_short is not None
        assert q_short.reservation > q_flat.reservation

    def test_vpin_widens_spread(self):
        cfg = MarketMakerConfig(vpin_threshold=0.5, vpin_spread_mult=2.0)
        q = PerpQuoter(cfg)
        q_normal = q.compute_quotes(mid=2000.0, inventory=0.0, vol=0.001, time_remaining=0.5, vpin=0.3)
        q_toxic = q.compute_quotes(mid=2000.0, inventory=0.0, vol=0.001, time_remaining=0.5, vpin=0.8)
        assert q_normal is not None and q_toxic is not None
        assert q_toxic.spread > q_normal.spread

    def test_funding_bias(self, quoter):
        q_zero = quoter.compute_quotes(mid=2000.0, inventory=0.0, vol=0.001, time_remaining=0.5, funding_rate=0.0)
        q_pos = quoter.compute_quotes(mid=2000.0, inventory=0.0, vol=0.001, time_remaining=0.5, funding_rate=0.001)
        assert q_zero is not None and q_pos is not None
        # Positive funding → reservation price lower (cost of holding long)
        assert q_pos.reservation < q_zero.reservation

    def test_invalid_inputs_return_none(self, quoter):
        assert quoter.compute_quotes(mid=0.0, inventory=0.0, vol=0.001, time_remaining=0.5) is None
        assert quoter.compute_quotes(mid=2000.0, inventory=0.0, vol=0.0, time_remaining=0.5) is None
        assert quoter.compute_quotes(mid=2000.0, inventory=0.0, vol=0.001, time_remaining=0.0) is None

    def test_spread_clamped_to_bounds(self):
        cfg = MarketMakerConfig(min_spread_bps=5.0, max_spread_bps=10.0)
        q = PerpQuoter(cfg)
        quote = q.compute_quotes(mid=2000.0, inventory=0.0, vol=0.001, time_remaining=0.5)
        assert quote is not None
        spread_bps = quote.spread / 2000.0 * 10000
        assert spread_bps >= 4.9  # allow small rounding
        assert spread_bps <= 10.5

    def test_tick_alignment(self, quoter):
        q = quoter.compute_quotes(mid=2000.005, inventory=0.0, vol=0.001, time_remaining=0.5)
        assert q is not None
        # Bid rounded down, ask rounded up to tick_size=0.01
        assert q.bid == round(q.bid, 2)
        assert q.ask == round(q.ask, 2)

    def test_higher_vol_wider_spread(self, quoter):
        q_low = quoter.compute_quotes(mid=2000.0, inventory=0.0, vol=0.0005, time_remaining=0.5)
        q_high = quoter.compute_quotes(mid=2000.0, inventory=0.0, vol=0.002, time_remaining=0.5)
        assert q_low is not None and q_high is not None
        assert q_high.spread >= q_low.spread


class TestConfig:
    def test_validate_ok(self):
        MarketMakerConfig().validate()

    def test_validate_bad_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            MarketMakerConfig(gamma=0).validate()

    def test_validate_bad_spread(self):
        with pytest.raises(ValueError, match="spread"):
            MarketMakerConfig(min_spread_bps=30, max_spread_bps=10).validate()

    def test_validate_bad_market_data_stale_window(self):
        with pytest.raises(ValueError, match="market_data_stale_s"):
            MarketMakerConfig(market_data_stale_s=0).validate()

    def test_max_inventory_qty(self):
        cfg = MarketMakerConfig(max_inventory_notional=50.0)
        assert abs(cfg.max_inventory_qty(2000.0) - 0.025) < 1e-6
        assert cfg.max_inventory_qty(0) == 0.0
