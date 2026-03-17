"""Parity tests: RustRiskAggregator vs Python RiskAggregator."""
import pytest

try:
    from _quant_hotpath import RustRiskAggregator
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not available")


class TestRiskAggregatorParity:
    def test_empty_aggregator_allows(self):
        agg = RustRiskAggregator()
        result = agg.evaluate({
            "symbol": "ETHUSDT", "side": "buy", "qty": 1.0,
            "price": 3000.0, "account_equity": 10000.0,
            "gross_exposure": 0.0, "net_exposure": 0.0,
        })
        verdict, reasons = result
        assert verdict == "allow"

    def test_add_max_position_rule(self):
        agg = RustRiskAggregator()
        agg.add_rule("max_pos", "max_position", {"max_qty": 10.0, "max_notional": 50000.0})
        assert agg.rule_count() == 1

        result = agg.evaluate({
            "symbol": "ETHUSDT", "side": "buy", "qty": 1.0,
            "price": 3000.0, "account_equity": 10000.0,
            "current_position_qty": 0.0,
        })
        verdict, reasons = result
        assert verdict == "allow"

    def test_max_position_reject(self):
        agg = RustRiskAggregator()
        agg.add_rule("max_pos", "max_position", {"max_qty": 0.5, "max_notional": 0.0})
        result = agg.evaluate({
            "symbol": "ETHUSDT", "side": "buy", "qty": 1.0,
            "price": 3000.0, "account_equity": 10000.0,
            "current_position_qty": 0.0,
        })
        verdict, reasons = result
        assert verdict == "reduce"  # headroom exists, so reduce not reject

    def test_nan_price_rejects(self):
        agg = RustRiskAggregator()
        agg.add_rule("max_pos", "max_position", {"max_qty": 10.0, "max_notional": 0.0})
        result = agg.evaluate({
            "symbol": "ETHUSDT", "side": "buy", "qty": 1.0,
            "price": float('nan'), "account_equity": 10000.0,
        })
        verdict, reasons = result
        assert verdict == "reject"

    def test_leverage_cap_reduce(self):
        agg = RustRiskAggregator()
        agg.add_rule("lev", "leverage_cap", {"max_gross_leverage": 1.0, "max_net_leverage": 0.0})
        result = agg.evaluate({
            "symbol": "ETHUSDT", "side": "buy", "qty": 5.0,
            "price": 3000.0, "account_equity": 10000.0,
            "gross_exposure": 5000.0,
        })
        verdict, reasons = result
        # 5000 + 5*3000 = 20000 / 10000 = 2.0x > 1.0x -> reduce
        assert verdict == "reduce"

    def test_drawdown_kills(self):
        agg = RustRiskAggregator()
        agg.add_rule("dd", "max_drawdown", {"warning_pct": 0.1, "kill_pct": 0.2})
        result = agg.evaluate({
            "symbol": "ETHUSDT", "side": "buy", "qty": 1.0,
            "price": 3000.0, "account_equity": 10000.0,
            "drawdown_pct": 0.25,
        })
        verdict, reasons = result
        assert verdict == "reject"

    def test_enable_disable(self):
        agg = RustRiskAggregator()
        agg.add_rule("r1", "max_position", {"max_qty": 0.1, "max_notional": 0.0})
        agg.disable("r1")
        result = agg.evaluate({
            "symbol": "ETHUSDT", "side": "buy", "qty": 1.0,
            "price": 3000.0, "account_equity": 10000.0,
        })
        verdict, _ = result
        assert verdict == "allow"  # disabled rule doesn't fire

    def test_snapshot(self):
        agg = RustRiskAggregator()
        agg.add_rule("r1", "max_position", {"max_qty": 10.0, "max_notional": 0.0})
        agg.evaluate({
            "symbol": "ETHUSDT", "side": "buy", "qty": 1.0,
            "price": 3000.0, "account_equity": 10000.0,
        })
        snap = agg.snapshot()
        assert isinstance(snap, dict)
        assert "r1" in snap

    def test_reduce_only_passes(self):
        agg = RustRiskAggregator()
        agg.add_rule("dd", "max_drawdown", {"warning_pct": 0.1, "kill_pct": 0.2})
        result = agg.evaluate({
            "symbol": "ETHUSDT", "side": "sell", "qty": 1.0,
            "price": 3000.0, "account_equity": 10000.0,
            "drawdown_pct": 0.25,
            "is_reduce_only": True,
        })
        verdict, _ = result
        assert verdict == "allow"
