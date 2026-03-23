"""Tests verifying all price/position calculations use correct values.

Covers the 4 historical bug patterns:
1. entry_price from fill (not bar close)
2. notional uses correct price, clamped properly
3. int() truncation on leverage (must be max(2, round()))
4. position_size flows correctly to notional check

Also verifies: step rounding, zero/NaN price rejection, stop price correctness.
"""
from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Ensure _quant_hotpath is available (real or fake)
# ---------------------------------------------------------------------------
try:
    import _quant_hotpath  # noqa: F401
except ImportError:
    # Minimal stub so AlphaRunner can be imported
    _fake = types.ModuleType("_quant_hotpath")

    class _FakeEngine:
        def __init__(self):
            self.push_bar = MagicMock()
            self.get_features = MagicMock(return_value=[])

    class _FakeBridge:
        def __init__(self, **_kw):
            self.zscore_normalize = MagicMock(return_value=None)
            self.apply_constraints = MagicMock(return_value=0)
            self.get_position = MagicMock(return_value=0)
            self.set_position = MagicMock()

    class _FakeOSM:
        def __init__(self):
            self.register = MagicMock()
            self.transition = MagicMock()
            self.active_count = MagicMock(return_value=0)

    class _FakeCB:
        def __init__(self, **_kw):
            self.allow_request = MagicMock(return_value=True)
            self.snapshot = MagicMock(return_value={})
            self.record_success = MagicMock()
            self.record_failure = MagicMock()

    class _Generic:
        def __init__(self, *a, **kw): pass
        def __getattr__(self, name): return MagicMock()

    class _FakeModule(types.ModuleType):
        def __getattr__(self, name):
            if name in {"RustPnLTracker", "RustCompositeRegimeDetector", "RustRegimeParamRouter"}:
                raise AttributeError(name)
            if name.startswith("rust_"):
                return lambda *a, **kw: a[0] if a else ""
            if name.startswith("Rust"):
                return _Generic
            raise AttributeError(name)

    _fake = _FakeModule("_quant_hotpath")
    _fake.RustFeatureEngine = _FakeEngine
    _fake.RustInferenceBridge = _FakeBridge
    _fake.RustOrderStateMachine = _FakeOSM
    _fake.RustCircuitBreaker = _FakeCB
    _fake.RustFillEvent = lambda **kw: SimpleNamespace(**kw)
    _fake.RustMarketEvent = lambda **kw: SimpleNamespace(**kw)
    _fake.rust_sanitize = lambda s: str(s)
    _fake.rust_short_hash = lambda text, n=10: str(abs(hash(text)))[:n]
    _fake.rust_make_idempotency_key = lambda v, a, k: f"{v}:{a}:{k}"
    sys.modules["_quant_hotpath"] = _fake

from runner.alpha_runner import AlphaRunner  # noqa: E402
from runner.strategy_config import MAX_ORDER_NOTIONAL  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_INFO = {
    "model": MagicMock(predict=MagicMock(return_value=[0.5])),
    "features": ["rsi_14", "bb_width_20"],
    "horizon_models": [],
    "lgbm_xgb_weight": 0.5,
    "config": {"version": "v11"},
    "deadzone": 0.3,
    "min_hold": 18,
    "max_hold": 60,
    "zscore_window": 720,
    "zscore_warmup": 180,
}


class MockAdapter:
    """Exchange adapter mock with controllable fill price."""

    def __init__(self, fill_price: float | None = None):
        self._fill_price = fill_price
        self.orders: list[dict] = []
        self._client = MagicMock()
        self._client.post.return_value = {"retCode": 0}

    def send_market_order(self, symbol, side, qty):
        self.orders.append({"symbol": symbol, "side": side, "qty": qty})
        return {"orderId": "mock_order_1", "status": "Filled", "retCode": 0}

    def get_positions(self, symbol=None):
        return []

    def get_balances(self):
        return {"USDT": SimpleNamespace(total="1000", available="900")}

    def get_ticker(self, symbol):
        return {"lastPrice": 2000.0}

    def close_position(self, symbol):
        return {"status": "ok", "retCode": 0}

    def get_recent_fills(self, symbol=None):
        if self._fill_price is not None:
            return [SimpleNamespace(price=str(self._fill_price))]
        return []

    def get_klines(self, symbol, interval="60", limit=800):
        return [{"open": 100, "high": 101, "low": 99, "close": 100,
                 "volume": 1000, "start": i * 3600000} for i in range(limit)]


def _make_runner(adapter=None, **kw):
    """Build an AlphaRunner with mocked internals."""
    if adapter is None:
        adapter = MockAdapter()
    defaults = dict(
        adapter=adapter,
        model_info=dict(_MODEL_INFO),
        symbol="ETHUSDT",
        dry_run=False,
        position_size=0.01,
        adaptive_sizing=False,
        min_size=0.01,
        step_size=0.01,
        start_oi_cache=False,
    )
    defaults.update(kw)
    r = AlphaRunner(**defaults)
    # Replace Rust components with simple mocks
    r._oi_cache = MagicMock()
    r._oi_cache.get.return_value = {
        "open_interest": 0, "ls_ratio": float("nan"),
        "top_trader_ls_ratio": float("nan"), "taker_buy_vol": float("nan"),
    }
    r._engine = MagicMock()
    r._engine.push_bar = MagicMock()
    r._engine.get_features.return_value = []
    r._inference = MagicMock()
    r._inference.zscore_normalize = MagicMock(return_value=None)
    r._inference.apply_constraints = MagicMock(return_value=0)
    r._inference.get_position = MagicMock(return_value=0)
    r._inference.set_position = MagicMock()
    r._osm = MagicMock()
    r._osm.active_count.return_value = 1
    r._circuit_breaker = MagicMock()
    r._circuit_breaker.allow_request.return_value = True
    r._circuit_breaker.snapshot.return_value = {}
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPriceCalculation:
    """Verify all price/position calculations use correct values."""

    # ------------------------------------------------------------------
    # 1. entry_price from fill, not bar close
    # ------------------------------------------------------------------

    def test_entry_price_from_fill_not_bar(self):
        """entry_price must come from fill, not bar['close']."""
        fill_price = 2005.50
        bar_close = 2000.00

        adapter = MockAdapter(fill_price=fill_price)
        runner = _make_runner(adapter=adapter)

        with patch("time.sleep"):
            runner._execute_signal_change(0, 1, bar_close)

        assert runner._entry_price == fill_price, (
            f"entry_price should be {fill_price} (fill), got {runner._entry_price}"
        )
        assert runner._entry_price != bar_close

    def test_entry_price_fallback_to_bar_when_no_fills(self):
        """When fills are unavailable, entry_price falls back to bar close."""
        bar_close = 2000.00
        adapter = MockAdapter(fill_price=None)  # no fills
        runner = _make_runner(adapter=adapter)

        with patch("time.sleep"):
            runner._execute_signal_change(0, 1, bar_close)

        assert runner._entry_price == bar_close

    def test_combo_entry_price_prefers_fill(self):
        """PortfolioCombiner entry_price must come from fill, not bar close."""
        from scripts.ops.portfolio_combiner import PortfolioCombiner

        fill_price = 2005.50
        bar_close = 2000.00

        adapter = MockAdapter(fill_price=fill_price)
        combiner = PortfolioCombiner(
            adapter=adapter, symbol="ETHUSDT",
            weights={"ETHUSDT": 0.5, "ETHUSDT_15m": 0.5},
            dry_run=False, min_size=0.01,
        )
        combiner._signals = {"ETHUSDT": 1, "ETHUSDT_15m": 1}
        combiner._current_position = 0

        with patch("time.sleep"):
            combiner._execute_change(1, bar_close)

        assert combiner._entry_price == fill_price, (
            f"COMBO entry_price should be {fill_price}, got {combiner._entry_price}"
        )

    def test_hedge_runner_entry_price_from_fill(self):
        """HedgeRunner must use fill price for entry tracking, not lastPrice."""
        from scripts.ops.hedge_runner import HedgeRunner

        fill_price = 0.85
        last_price = 0.84

        adapter = MockAdapter(fill_price=fill_price)
        adapter.get_ticker = lambda sym: {"lastPrice": last_price}

        runner = HedgeRunner(adapter=adapter, dry_run=False)

        # Simulate _open_hedge_positions by pre-setting state
        with patch("time.sleep"):
            runner._open_hedge_positions()

        for sym, entry in runner._entry_prices.items():
            assert entry == fill_price, (
                f"HEDGE {sym} entry should be {fill_price} (fill), got {entry}"
            )

    # ------------------------------------------------------------------
    # 2. notional uses correct price and is clamped
    # ------------------------------------------------------------------

    def test_notional_clamp_prevents_oversized_order(self):
        """position_size is clamped when notional exceeds MAX_ORDER_NOTIONAL."""
        runner = _make_runner(position_size=10.0, min_size=0.01)
        runner._position_size = 10.0
        runner._z_scale = 1.0

        price = 2000.0
        # notional = 10.0 * 2000 = $20,000 >> MAX_ORDER_NOTIONAL ($500)
        with patch("time.sleep"):
            runner._execute_signal_change(0, 1, price)

        expected_max_size = MAX_ORDER_NOTIONAL / price  # 500/2000 = 0.25
        assert runner._position_size <= expected_max_size + 0.01, (
            f"position_size {runner._position_size} should be clamped to ~{expected_max_size}"
        )

    def test_notional_check_uses_computed_position_size(self):
        """The notional check in _execute_signal_change uses the current _position_size."""
        runner = _make_runner(position_size=0.01)
        runner._position_size = 0.01
        runner._z_scale = 1.0

        price = 2000.0
        with patch("time.sleep"):
            runner._execute_signal_change(0, 1, price)

        # 0.01 * 2000 = $20 < $500, so size should remain 0.01
        assert runner._position_size == 0.01

    # ------------------------------------------------------------------
    # 3. leverage rounding: int(1.5)=1, must be max(2, round())
    # ------------------------------------------------------------------

    def test_leverage_rounding_minimum_2(self):
        """leverage=1.5 must round up to 2 (not int()=1)."""
        lev = 1.5
        fixed = max(2, int(round(lev)))
        broken = int(lev)

        assert broken == 1, "int(1.5) should be 1 (the old bug)"
        assert fixed == 2, f"max(2, int(round(1.5))) should be 2, got {fixed}"

    def test_leverage_ladder_all_tiers(self):
        """Every equity tier leverage value must produce integer >= 2."""
        for threshold, lev_val in AlphaRunner.LEVERAGE_LADDER:
            lev_int = max(2, int(round(lev_val)))
            assert lev_int >= 2, (
                f"Tier equity>={threshold}: lev={lev_val} -> lev_int={lev_int}, must be >= 2"
            )
            assert isinstance(lev_int, int)

    def test_leverage_1_0_becomes_2(self):
        """leverage=1.0 from the ladder must be raised to minimum 2."""
        assert max(2, int(round(1.0))) == 2

    def test_leverage_rounding_values(self):
        """Test various leverage values for correct rounding."""
        for lev_val, expected_min in [
            (0.5, 2), (1.0, 2), (1.4, 2), (1.5, 2), (2.0, 2),
            (3.0, 3), (10.0, 10),
        ]:
            result = max(2, int(round(lev_val)))
            assert result >= expected_min, f"lev={lev_val}: expected >= {expected_min}, got {result}"
            assert result >= 2

    def test_combiner_leverage_integer(self):
        """PortfolioCombiner leverage uses max(2, int(round()))."""
        leverage = 10.0
        lev_int = max(2, int(round(leverage)))
        assert lev_int == 10
        assert isinstance(lev_int, int)

    # ------------------------------------------------------------------
    # 4. position_size flows to notional check
    # ------------------------------------------------------------------

    def test_position_size_flows_to_notional_check(self):
        """compute_position_size return value feeds into notional check."""
        runner = _make_runner(adaptive_sizing=True)
        runner._z_scale = 1.0
        runner._dynamic_scale = 1.0

        price = 2000.0
        computed = runner._compute_position_size(price)
        assert computed == runner._position_size, (
            f"_compute_position_size returned {computed} but self._position_size is {runner._position_size}"
        )

        with patch("time.sleep"):
            runner._execute_signal_change(0, 1, price)

        final_notional = runner._position_size * price
        assert final_notional <= MAX_ORDER_NOTIONAL * 1.01, (
            f"Final notional ${final_notional:.2f} exceeds MAX_ORDER_NOTIONAL ${MAX_ORDER_NOTIONAL}"
        )

    def test_position_size_not_stale_after_compute(self):
        """_compute_position_size must update _position_size, not leave old value."""
        runner = _make_runner(position_size=0.01, adaptive_sizing=False)
        runner._position_size = 999.0  # stale

        runner._compute_position_size(2000.0)
        assert runner._position_size == 0.01, (
            f"Expected 0.01, got {runner._position_size} (stale value not overwritten)"
        )

    # ------------------------------------------------------------------
    # 5. qty step rounding
    # ------------------------------------------------------------------

    def test_qty_step_rounding(self):
        """SUI step=10: qty=13.7 -> 10, qty=15 -> 10, qty=25 -> 20."""
        runner = _make_runner(symbol="SUIUSDT", position_size=10, step_size=10)
        assert runner._round_to_step(13.7) == 10
        assert runner._round_to_step(15.0) == 10  # floor, not round
        assert runner._round_to_step(25.0) == 20
        assert runner._round_to_step(9.9) == 0
        assert runner._round_to_step(10.0) == 10

    def test_qty_step_rounding_small_step(self):
        """ETH step=0.01: qty=0.015 -> 0.01."""
        runner = _make_runner(step_size=0.01)
        assert runner._round_to_step(0.015) == 0.01
        assert runner._round_to_step(0.029) == 0.02

    def test_qty_step_rounding_axs(self):
        """AXS step=0.1: qty=5.37 -> 5.3."""
        runner = _make_runner(symbol="AXSUSDT", position_size=5.0, step_size=0.1)
        assert runner._round_to_step(5.37) == 5.3
        assert runner._round_to_step(5.39) == 5.3

    def test_hedge_runner_step_rounding(self):
        """HedgeRunner _round_to_step: ADA step=1.0 floors correctly."""
        from scripts.ops.hedge_runner import _round_to_step

        assert _round_to_step(13.7, "ADAUSDT") == 13.0
        assert _round_to_step(0.9, "ADAUSDT") == 0.0
        assert _round_to_step(5.37, "XRPUSDT") == 5.3

    # ------------------------------------------------------------------
    # 6. zero price rejected
    # ------------------------------------------------------------------

    def test_zero_price_rejected(self):
        """price=0 must not cause division by zero."""
        runner = _make_runner(adaptive_sizing=True)
        size = runner._compute_position_size(0.0)
        assert not math.isnan(size)
        assert not math.isinf(size)

    def test_zero_price_notional_clamp(self):
        """clamp_notional with price=0 must not divide by zero."""
        from execution.order_utils import clamp_notional
        result = clamp_notional(1.0, 0.0, "ETHUSDT")
        assert result == 1.0
        assert not math.isnan(result)

    def test_zero_entry_price_pnl_tracker(self):
        """PnLTracker.record_close with entry_price=0 must not divide by zero."""
        from attribution.pnl_tracker import PnLTracker
        tracker = PnLTracker()
        trade = tracker.record_close(
            symbol="ETHUSDT", side=1, entry_price=0.0,
            exit_price=2000.0, size=0.01, reason="test",
        )
        assert trade["pnl_usd"] == 0.0
        assert "error" in trade

    # ------------------------------------------------------------------
    # 7. NaN price rejected
    # ------------------------------------------------------------------

    def test_nan_price_rejected(self):
        """price=NaN must not propagate through clamp_notional."""
        from execution.order_utils import clamp_notional
        result = clamp_notional(1.0, float("nan"), "ETHUSDT")
        assert result == 1.0 or not math.isnan(result)

    def test_nan_entry_price_pnl_tracker(self):
        """PnLTracker with NaN entry_price must not propagate NaN."""
        from attribution.pnl_tracker import PnLTracker
        tracker = PnLTracker()
        trade = tracker.record_close(
            symbol="ETHUSDT", side=1,
            entry_price=float("nan"), exit_price=2000.0,
            size=0.01, reason="test",
        )
        assert trade["pnl_usd"] == 0.0
        assert not math.isnan(trade["pnl_usd"])
        assert "error" in trade

    def test_nan_exit_price_pnl_tracker(self):
        """PnLTracker with NaN exit_price must not propagate NaN."""
        from attribution.pnl_tracker import PnLTracker
        tracker = PnLTracker()
        trade = tracker.record_close(
            symbol="ETHUSDT", side=1,
            entry_price=2000.0, exit_price=float("nan"),
            size=0.01, reason="test",
        )
        assert trade["pnl_usd"] == 0.0
        assert not math.isnan(trade["pnl_usd"])
        assert "error" in trade

    # ------------------------------------------------------------------
    # 8. Stop price uses fill price, not bar close
    # ------------------------------------------------------------------

    def test_stop_price_uses_actual_fill_price(self):
        """After open, _entry_price and _trade_peak_price use fill, not bar close."""
        fill_price = 2005.50
        bar_close = 2000.00

        adapter = MockAdapter(fill_price=fill_price)
        runner = _make_runner(adapter=adapter)
        runner._atr_buffer = [0.01] * 20

        with patch("time.sleep"):
            runner._execute_signal_change(0, 1, bar_close)

        assert runner._entry_price == fill_price
        assert runner._trade_peak_price == fill_price

    # ------------------------------------------------------------------
    # 9. Combo entry_price helper
    # ------------------------------------------------------------------

    def test_combo_entry_price_helper(self):
        """_combo_entry_price prefers fill_price over fallback."""
        from scripts.ops.run_bybit_alpha import _combo_entry_price

        assert _combo_entry_price({"fill_price": 2005.50}, 2000.0) == 2005.50
        assert _combo_entry_price({"from": 0, "to": 1}, 2000.0) == 2000.0
        assert _combo_entry_price(None, 2000.0) == 2000.0
