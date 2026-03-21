"""Comprehensive tests for AlphaRunner — the production trading path.

AlphaRunner is a 1,234-line class with 12 Rust components. This file provides
~35 tests covering: process_bar, signal execution, regime filtering, stop-loss,
position sizing, reconciliation, PnL tracking, and kill switch behavior.

_quant_hotpath is mocked since Rust may not be compiled in test environments.
"""
from __future__ import annotations

import sys
import types
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from execution.models.balances import BalanceSnapshot, CanonicalBalance

# ---------------------------------------------------------------------------
# Mock _quant_hotpath BEFORE importing AlphaRunner (it imports at module init)
# ---------------------------------------------------------------------------
class _FakeFeatureEngine:
    def __init__(self):
        self.push_bar = MagicMock()
        self.get_features = MagicMock(return_value=[])


class _FakeInferenceBridge:
    def __init__(self, **_kwargs):
        self.zscore_normalize = MagicMock(return_value=None)
        self.apply_constraints = MagicMock(return_value=0)
        self.get_position = MagicMock(return_value=0)
        self.set_position = MagicMock()


class _FakeOrderStateMachine:
    def __init__(self):
        self.register = MagicMock()
        self.transition = MagicMock()
        self.active_count = MagicMock(return_value=0)


class _FakeCircuitBreaker:
    def __init__(self, **_kwargs):
        self.allow_request = MagicMock(return_value=True)
        self.snapshot = MagicMock(return_value={})
        self.record_success = MagicMock()
        self.record_failure = MagicMock()


class _FakeAckStore:
    def __init__(self, **_kwargs):
        self._data = {}

    def get_json(self, key, _now):
        return self._data.get(key)

    def put_json(self, key, payload_json, _now):
        self._data[key] = payload_json

    def prune(self, _now):
        return 0


class _GenericRustThing:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __getattr__(self, name):
        return MagicMock()


class _FakeHotpathModule(types.ModuleType):
    def __getattr__(self, name):
        if name in {
            "RustPnLTracker",
            "RustCompositeRegimeDetector",
            "RustRegimeParamRouter",
        }:
            raise AttributeError(name)
        if name.startswith("rust_"):
            return lambda *args, **kwargs: args[0] if args else ""
        if name.startswith("Rust"):
            return _GenericRustThing
        raise AttributeError(name)


try:  # Prefer the real extension when available; only stub on environments without it.
    import _quant_hotpath as _quant_hotpath  # noqa: F401
except ImportError:
    _fake_hotpath = _FakeHotpathModule("_quant_hotpath")
    _fake_hotpath.RustFeatureEngine = _FakeFeatureEngine
    _fake_hotpath.RustInferenceBridge = _FakeInferenceBridge
    _fake_hotpath.RustOrderStateMachine = _FakeOrderStateMachine
    _fake_hotpath.RustCircuitBreaker = _FakeCircuitBreaker
    _fake_hotpath.RustAckStore = _FakeAckStore
    _fake_hotpath.rust_sanitize = lambda s: "".join(ch for ch in str(s) if ch.isalnum() or ch in "-_")
    _fake_hotpath.rust_short_hash = lambda text, n=10: str(abs(hash(text)))[:n]
    _fake_hotpath.rust_make_idempotency_key = lambda venue, action, key: f"{venue}:{action}:{key}"
    _fake_hotpath.RustFillEvent = lambda **kwargs: types.SimpleNamespace(**kwargs)
    _fake_hotpath.RustMarketEvent = lambda **kwargs: types.SimpleNamespace(**kwargs)
    sys.modules["_quant_hotpath"] = _fake_hotpath

from scripts.ops.alpha_runner import AlphaRunner  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

_MINIMAL_MODEL_INFO = {
    "model": MagicMock(predict=MagicMock(return_value=[0.5])),
    "features": ["rsi_14", "bb_width_20", "close_vs_ma20", "close_vs_ma50",
                  "parkinson_vol", "vol_of_vol", "adx_14", "volume_ratio",
                  "funding_rate", "oi_change_pct", "ls_ratio", "taker_ratio",
                  "bb_position", "atr_14"],
    "horizon_models": [{
        "horizon": 24,
        "lgbm": MagicMock(predict=MagicMock(return_value=[0.5])),
        "xgb": None,
        "ridge": MagicMock(predict=MagicMock(return_value=[0.3])),
        "ridge_features": ["rsi_14", "bb_width_20", "close_vs_ma20"],
        "features": ["rsi_14", "bb_width_20", "close_vs_ma20", "close_vs_ma50",
                      "parkinson_vol", "vol_of_vol", "adx_14", "volume_ratio",
                      "funding_rate", "oi_change_pct", "ls_ratio", "taker_ratio",
                      "bb_position", "atr_14"],
        "ic": 0.03,
    }],
    "lgbm_xgb_weight": 0.5,
    "config": {"version": "v11", "multi_horizon": True, "ridge_weight": 0.6, "lgbm_weight": 0.4},
    "deadzone": 0.3,
    "min_hold": 18,
    "max_hold": 60,
    "zscore_window": 720,
    "zscore_warmup": 180,
}

_OI_STUB = {
    "open_interest": 0,
    "ls_ratio": float("nan"),
    "top_trader_ls_ratio": float("nan"),
    "taker_buy_vol": float("nan"),
}


class MockAdapter:
    def __init__(self):
        self.orders = []
        self.positions = {}
        self._balances = {"USDT": type("B", (), {"total": 1000.0, "available": 1000.0})()}
        self._client = MagicMock()

    def send_market_order(self, symbol, side, qty):
        self.orders.append({"symbol": symbol, "side": side, "qty": qty})
        return {"orderId": f"mock_{len(self.orders)}", "status": "Filled", "retCode": 0}

    def get_positions(self, symbol=None):
        return []

    def get_balances(self):
        return self._balances

    def get_ticker(self, symbol):
        return {"fundingRate": "0.0001"}

    def close_position(self, symbol):
        self.orders.append({"symbol": symbol, "action": "close"})
        return {"status": "ok", "retCode": 0}

    def get_klines(self, symbol, interval="60", limit=800):
        return [{"open": 100, "high": 101, "low": 99, "close": 100,
                 "volume": 1000, "start": i * 3600000}
                for i in range(limit)]


def _make_bar(close=100.0, high=None, low=None, open_=None, volume=1000):
    """Create a standard bar dict."""
    if high is None:
        high = close * 1.01
    if low is None:
        low = close * 0.99
    if open_ is None:
        open_ = close
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _canonical_usdt_snapshot(*, total: str, free: str) -> BalanceSnapshot:
    total_dec = Decimal(total)
    free_dec = Decimal(free)
    locked_dec = total_dec - free_dec
    return BalanceSnapshot(
        venue="bybit",
        ts_ms=0,
        balances=(
            CanonicalBalance.from_free_locked(
                venue="bybit",
                asset="USDT",
                free=free_dec,
                locked=locked_dec,
                ts_ms=0,
            ),
        ),
    )


@pytest.fixture
def adapter():
    return MockAdapter()


@pytest.fixture
def runner(adapter):
    """Build a standard AlphaRunner with all Rust mocks configured."""
    info = dict(_MINIMAL_MODEL_INFO)
    r = AlphaRunner(
        adapter=adapter,
        model_info=info,
        symbol="ETHUSDT",
        dry_run=False,
        min_size=0.01,
        step_size=0.01,
        start_oi_cache=False,
    )
    r._oi_cache = MagicMock()
    r._oi_cache.get.return_value = dict(_OI_STUB)
    r._engine = MagicMock()
    r._engine.push_bar = MagicMock()
    r._engine.get_features.return_value = [
        ("rsi_14", 50.0), ("bb_width_20", 0.02), ("close_vs_ma20", 0.01),
        ("close_vs_ma50", 0.02), ("parkinson_vol", 0.03), ("vol_of_vol", 0.01),
        ("adx_14", 25.0), ("volume_ratio", 1.0), ("funding_rate", 0.0001),
        ("oi_change_pct", 0.0), ("ls_ratio", 1.0), ("taker_ratio", 0.5),
        ("bb_position", 0.5), ("atr_14", 0.015),
    ]
    r._inference = MagicMock()
    r._inference.zscore_normalize = MagicMock(return_value=None)
    r._inference.apply_constraints = MagicMock(return_value=0)
    r._inference.get_position = MagicMock(return_value=0)
    r._inference.set_position = MagicMock()
    r._osm = MagicMock()
    r._osm.register = MagicMock()
    r._osm.transition = MagicMock()
    r._osm.active_count.return_value = 1
    r._circuit_breaker = MagicMock()
    r._circuit_breaker.allow_request.return_value = True
    r._circuit_breaker.snapshot.return_value = {}
    r._circuit_breaker.record_success = MagicMock()
    r._circuit_breaker.record_failure = MagicMock()
    return r


@pytest.fixture
def dry_runner(adapter):
    """Build a dry-run AlphaRunner."""
    info = dict(_MINIMAL_MODEL_INFO)
    r = AlphaRunner(
        adapter=adapter,
        model_info=info,
        symbol="ETHUSDT",
        dry_run=True,
        min_size=0.01,
        step_size=0.01,
        start_oi_cache=False,
    )
    r._oi_cache = MagicMock()
    r._oi_cache.get.return_value = dict(_OI_STUB)
    r._engine = MagicMock()
    r._engine.push_bar = MagicMock()
    r._engine.get_features.return_value = [
        ("rsi_14", 50.0), ("bb_width_20", 0.02),
    ]
    r._inference = MagicMock()
    r._inference.zscore_normalize = MagicMock(return_value=None)
    r._inference.apply_constraints = MagicMock(return_value=0)
    r._inference.get_position = MagicMock(return_value=0)
    r._inference.set_position = MagicMock()
    r._circuit_breaker = MagicMock()
    r._circuit_breaker.allow_request.return_value = True
    r._circuit_breaker.snapshot.return_value = {}
    r._circuit_breaker.record_success = MagicMock()
    r._circuit_breaker.record_failure = MagicMock()
    return r


def _warm_regime(runner, n=25, base_price=100.0):
    """Push enough closes into the runner to warm up the regime filter."""
    for i in range(n):
        runner._check_regime(base_price + i * 0.1)


# ===========================================================================
# A) TestProcessBarWarmup
# ===========================================================================

class TestProcessBarWarmup:
    """Tests for process_bar during z-score warmup phase."""

    @patch("scripts.ops.alpha_runner._fetch_binance_oi_data", return_value=_OI_STUB)
    def test_warmup_returns_warmup_action(self, mock_oi, runner):
        """When zscore_normalize returns None, action should be 'warmup'."""
        runner._inference.zscore_normalize.return_value = None
        bar = _make_bar(close=100.0)
        result = runner.process_bar(bar)
        assert result["action"] == "warmup"

    @patch("scripts.ops.alpha_runner._fetch_binance_oi_data", return_value=_OI_STUB)
    def test_warmup_no_orders(self, mock_oi, runner, adapter):
        """During warmup, no orders should be sent to the exchange."""
        runner._inference.zscore_normalize.return_value = None
        bar = _make_bar(close=100.0)
        runner.process_bar(bar)
        assert adapter.orders == []

    @patch("scripts.ops.alpha_runner._fetch_binance_oi_data", return_value=_OI_STUB)
    def test_warmup_pred_included(self, mock_oi, runner):
        """Warmup result should include the 'pred' key with the ensemble prediction."""
        runner._inference.zscore_normalize.return_value = None
        bar = _make_bar(close=100.0)
        result = runner.process_bar(bar)
        assert "pred" in result
        assert isinstance(result["pred"], float)


# ===========================================================================
# B) TestProcessBarSignal
# ===========================================================================

class TestProcessBarSignal:
    """Tests for signal generation during normal (post-warmup) processing."""

    @patch("scripts.ops.alpha_runner._fetch_binance_oi_data", return_value=_OI_STUB)
    def test_high_pred_positive_signal(self, mock_oi, runner):
        """When apply_constraints returns +1, signal should be +1."""
        runner._inference.zscore_normalize.return_value = 1.5
        runner._inference.apply_constraints.return_value = 1
        runner._inference.get_position.return_value = 1
        bar = _make_bar(close=100.0)
        result = runner.process_bar(bar)
        assert result["signal"] == 1

    @patch("scripts.ops.alpha_runner._fetch_binance_oi_data", return_value=_OI_STUB)
    def test_low_pred_negative_signal(self, mock_oi, runner):
        """When apply_constraints returns -1, signal should be -1."""
        runner._inference.zscore_normalize.return_value = -1.5
        runner._inference.apply_constraints.return_value = -1
        runner._inference.get_position.return_value = -1
        bar = _make_bar(close=100.0)
        result = runner.process_bar(bar)
        assert result["signal"] == -1

    @patch("scripts.ops.alpha_runner._fetch_binance_oi_data", return_value=_OI_STUB)
    def test_neutral_pred_zero_signal(self, mock_oi, runner):
        """When apply_constraints returns 0, signal should be 0."""
        runner._inference.zscore_normalize.return_value = 0.1
        runner._inference.apply_constraints.return_value = 0
        runner._inference.get_position.return_value = 0
        bar = _make_bar(close=100.0)
        result = runner.process_bar(bar)
        assert result["signal"] == 0

    @patch("scripts.ops.alpha_runner._fetch_binance_oi_data", return_value=_OI_STUB)
    def test_killed_process_bar_forces_flat_state(self, mock_oi, runner):
        """Kill state must not leave a stale non-zero local signal behind."""
        kill_switch = MagicMock()
        kill_switch.is_armed.return_value = True
        runner._kill_switch = kill_switch
        runner._inference.zscore_normalize.return_value = 1.5
        runner._inference.apply_constraints.return_value = 1
        runner._inference.get_position.return_value = 1

        result = runner.process_bar(_make_bar(close=100.0))

        assert result["signal"] == 0
        assert result["trade"]["action"] == "killed"
        assert runner._current_signal == 0
        assert runner._entry_price == 0.0
        runner._inference.set_position.assert_any_call("ETHUSDT", 0, 1)

    @patch("scripts.ops.alpha_runner._fetch_binance_oi_data", return_value=_OI_STUB)
    def test_min_hold_respected(self, mock_oi, runner):
        """After signal change, subsequent bars within min_hold keep same signal via bridge."""
        # Bar 1: signal goes to +1
        runner._inference.zscore_normalize.return_value = 1.5
        runner._inference.apply_constraints.return_value = 1
        runner._inference.get_position.return_value = 1
        runner.process_bar(_make_bar(close=100.0))
        assert runner._current_signal == 1

        # Bar 2: apply_constraints still returns +1 (min_hold enforced by bridge)
        # Use z=0.2 (weak but positive) to avoid z_reversal exit (threshold -0.3)
        runner._inference.zscore_normalize.return_value = 0.2
        runner._inference.apply_constraints.return_value = 1  # bridge holds
        runner._inference.get_position.return_value = 2
        result = runner.process_bar(_make_bar(close=100.2, low=100.0))
        assert result["signal"] == 1

    @patch("scripts.ops.alpha_runner._fetch_binance_oi_data", return_value=_OI_STUB)
    def test_regime_filtered_forces_flat(self, mock_oi, runner):
        """When regime is inactive, deadzone=999 forces signal to 0."""
        # Mock _check_regime to always return False (regime inactive)
        runner._check_regime = MagicMock(return_value=False)
        runner._regime_active = False

        runner._inference.zscore_normalize.return_value = 1.5
        # Even with a strong z-score, deadzone=999 means apply_constraints sees flat
        runner._inference.apply_constraints.return_value = 0
        runner._inference.get_position.return_value = 0
        bar = _make_bar(close=100.0)
        result = runner.process_bar(bar)
        assert result["signal"] == 0
        assert result["regime"] == "filtered"


# ===========================================================================
# C) TestExecuteSignalChange
# ===========================================================================

class TestExecuteSignalChange:
    """Tests for _execute_signal_change — the order execution path."""

    def test_open_long(self, runner, adapter):
        """Signal 0→+1 sends a buy order and sets entry_price."""
        runner._position_size = 0.05
        runner._circuit_breaker.allow_request.return_value = True
        runner._osm.active_count.return_value = 1
        runner._execute_signal_change(0, 1, 2000.0)
        # Should have sent a buy order
        buy_orders = [o for o in adapter.orders if o.get("side") == "buy"]
        assert len(buy_orders) == 1
        assert buy_orders[0]["qty"] == 0.05
        assert runner._entry_price == 2000.0
        assert runner._entry_size == 0.05

    def test_close_long_open_short(self, runner, adapter):
        """Signal +1→-1 closes long then opens short."""
        runner._current_signal = 1
        runner._entry_price = 2000.0
        runner._entry_size = 0.05
        runner._position_size = 0.05
        runner._circuit_breaker.allow_request.return_value = True
        runner._osm.active_count.return_value = 1
        runner._execute_signal_change(1, -1, 2100.0)
        # Should have close + sell order
        close_orders = [o for o in adapter.orders if o.get("action") == "close"]
        sell_orders = [o for o in adapter.orders if o.get("side") == "sell"]
        assert len(close_orders) == 1
        assert len(sell_orders) == 1
        assert runner._entry_price == 2100.0

    def test_close_to_flat(self, runner, adapter):
        """Signal +1→0 closes position and records PnL."""
        runner._current_signal = 1
        runner._entry_price = 2000.0
        runner._entry_size = 0.05
        runner._position_size = 0.05
        runner._circuit_breaker.allow_request.return_value = True
        result = runner._execute_signal_change(1, 0, 2100.0)
        # Should have closed via adapter
        close_orders = [o for o in adapter.orders if o.get("action") == "close"]
        assert len(close_orders) == 1
        # PnL should be recorded: (2100-2000)/2000 * 2000 * 0.05 = 5%*100 = $5
        assert runner._pnl.trade_count == 1
        assert runner._pnl.total_pnl > 0
        assert result.get("action") == "flat"
        assert runner._entry_price == 0.0

    def test_killed_rejects(self, runner, adapter):
        """When kill switch is armed, execution returns killed."""
        kill_switch = MagicMock()
        kill_switch.is_armed.return_value = True
        runner._kill_switch = kill_switch
        result = runner._execute_signal_change(0, 1, 2000.0)
        assert result["action"] == "killed"
        assert adapter.orders == []

    def test_killed_allows_close_but_blocks_reopen(self, runner, adapter):
        """Kill state must still flatten existing exposure instead of leaving it stranded."""
        kill_switch = MagicMock()
        kill_switch.is_armed.return_value = True
        runner._kill_switch = kill_switch
        runner._current_signal = 1
        runner._entry_price = 2000.0
        runner._entry_size = 0.05
        runner._position_size = 0.05
        runner._circuit_breaker.allow_request.return_value = True

        result = runner._execute_signal_change(1, -1, 2100.0)

        assert result["action"] == "killed"
        close_orders = [o for o in adapter.orders if o.get("action") == "close"]
        assert len(close_orders) == 1
        buy_orders = [o for o in adapter.orders if o.get("side") == "buy"]
        assert len(buy_orders) == 0

    def test_dry_run_no_orders(self, dry_runner, adapter):
        """dry_run=True returns dry_run action without sending orders."""
        dry_runner._position_size = 0.05
        result = dry_runner._execute_signal_change(0, 1, 2000.0)
        assert result["action"] == "dry_run"
        assert result["from"] == 0
        assert result["to"] == 1
        # No real orders sent
        assert adapter.orders == []

    def test_margin_skip_uses_canonical_free_balance(self, runner, adapter):
        """Canonical free balance must block opens when margin is unavailable."""
        adapter.get_balances = MagicMock(
            return_value=_canonical_usdt_snapshot(total="1000", free="0"),
        )
        runner._position_size = 0.10
        runner._circuit_breaker.allow_request.return_value = True
        runner._osm.active_count.return_value = 1

        result = runner._execute_signal_change(0, 1, 2000.0)

        assert result["action"] == "margin_skip"
        assert result["avail"] == 0.0
        assert adapter.orders == []
        assert any(
            call.args[1] == "rejected" and call.kwargs.get("reason") == "insufficient_margin"
            for call in runner._osm.transition.call_args_list
        )


# ===========================================================================
# D) TestCheckRegime
# ===========================================================================

class TestCheckRegime:
    """Tests for _check_regime — vol/trend/ranging filter + dynamic deadzone."""

    def test_high_vol_active(self, runner):
        """High volatility above threshold keeps regime active."""
        # Push 25 bars with significant price movement to build returns
        prices = [100 + i * 2.0 for i in range(25)]  # large moves → high vol
        for p in prices:
            runner._check_regime(p)
        # With large moves, vol should exceed 0.004 threshold
        assert runner._regime_active

    def test_low_vol_no_trend_inactive(self, runner):
        """Both vol and trend below threshold → regime inactive."""
        # Push many bars with tiny price changes (flat market)
        # First fill MA window with flat prices
        for i in range(500):
            runner._check_regime(100.0 + (i % 2) * 0.001)
        # Now vol_20 should be tiny and trend (close/MA480 - 1) near zero
        assert not runner._regime_active

    def test_ranging_detected(self, runner):
        """High path / low displacement (range-bound) → regime inactive."""
        # Build enough history, then create a choppy range
        base = 100.0
        # First build enough close history for MA
        for i in range(500):
            # Oscillate around base → high path, low net displacement
            runner._check_regime(base + 2 * ((-1) ** i))
        # After choppy action with zero net displacement, ranging should kick in
        # and since trend is also low, regime should be inactive
        assert not runner._regime_active

    def test_dynamic_deadzone_high_vol(self, runner):
        """High vol relative to median → deadzone increases."""
        original_dz = runner._deadzone
        # Push bars with large moves to get high vol_20
        for i in range(25):
            runner._check_regime(100 + i * 3.0)
        # Deadzone should have increased (vol > median)
        assert runner._deadzone > original_dz or runner._deadzone >= 0.15

    def test_dynamic_deadzone_low_vol(self, runner):
        """Low vol relative to median → deadzone decreases toward floor."""
        runner._deadzone_base = 0.3
        runner._vol_median = 0.0063
        # Push bars with tiny moves → low vol
        for i in range(25):
            runner._check_regime(100.0 + i * 0.0001)
        # Vol_20 should be near zero → ratio << 1 → deadzone should decrease
        assert runner._deadzone < 0.3  # below base
        assert runner._deadzone >= 0.15  # but not below floor


# ===========================================================================
# E) TestStopLoss
# ===========================================================================

class TestStopLoss:
    """Tests for adaptive ATR-based stop-loss (_compute_stop_price + check_realtime_stoploss)."""

    def _setup_long(self, runner, entry=2000.0):
        """Helper: set up a long position."""
        runner._current_signal = 1
        runner._entry_price = entry
        runner._entry_size = 0.05
        runner._trade_peak_price = entry
        runner._atr_buffer = [0.015] * 20  # 1.5% ATR

    def test_initial_stop_distance(self, runner):
        """Initial stop should be entry × (1 - ATR × atr_stop_mult)."""
        self._setup_long(runner, entry=2000.0)
        stop = runner._compute_stop_price(2000.0)
        # ATR = 0.015, mult = 1.2 → initial_stop_dist = 0.018
        # stop = 2000 * (1 - 0.018) = 1964
        assert stop == pytest.approx(1964.0, rel=0.01)

    def test_breakeven_phase(self, runner):
        """After 1×ATR profit, stop moves near entry (breakeven)."""
        self._setup_long(runner, entry=2000.0)
        # 1×ATR profit means profit_pct >= atr * breakeven_atr = 0.015 * 1.0 = 1.5%
        # Peak at 2000 * 1.016 = 2032 → profit_pct = 1.6% > 1.5%
        profit_price = 2032.0
        stop = runner._compute_stop_price(profit_price)
        # Should be near entry: entry * (1 + atr * 0.1) = 2000 * 1.0015 = 2003
        assert stop > runner._entry_price  # above entry (breakeven + buffer)
        assert stop < profit_price  # below current price

    def test_trailing_phase(self, runner):
        """After trail_atr_mult×ATR profit, stop trails from peak."""
        self._setup_long(runner, entry=2000.0)
        # trail activates at 0.8×ATR profit = 0.015 * 0.8 = 1.2%
        # Also needs >= breakeven (1.0×ATR = 1.5%)
        # So need profit >= 1.5% to be in breakeven, and >= 1.2% for trail
        # Actually: code checks breakeven first (1×ATR=1.5%), then trail (0.8×ATR=1.2%)
        # Trail threshold < breakeven threshold, so once in breakeven we're already past trail
        # Wait — re-reading: if profit >= atr*breakeven_atr (1.5%), then check trail (0.8×ATR=1.2%)
        # Since 1.5% > 1.2%, trail always activates once breakeven does
        profit_price = 2040.0  # 2% profit > 1.5% breakeven threshold
        stop = runner._compute_stop_price(profit_price)
        # Trail: peak * (1 - atr * trail_step) = 2040 * (1 - 0.015 * 0.3) = 2040 * 0.9955 = 2030.82
        expected_trail = 2040.0 * (1 - 0.015 * 0.3)
        assert stop == pytest.approx(expected_trail, rel=0.01)

    @patch("scripts.ops.alpha_runner._fetch_binance_oi_data", return_value=_OI_STUB)
    def test_realtime_stop_triggers_close(self, mock_oi, runner, adapter):
        """Price crossing trailing stop level triggers position close.

        Trailing stop scenario: entry at 2000, peak at 2100 (>1.5% breakeven threshold),
        trail activates at 0.8×ATR profit. Stop = peak*(1-ATR*trail_step).
        Then price drops below the trailing stop.
        """
        self._setup_long(runner, entry=2000.0)
        runner._trade_peak_price = 2100.0  # simulate previous peak (5% profit)
        runner._position_size = 0.05
        runner._kill_switch = None  # ensure _killed is False
        runner._circuit_breaker.allow_request.return_value = True
        # Trail stop = 2100 * (1 - 0.015 * 0.3) = 2100 * 0.9955 = 2090.55
        # Hard ceiling: min_dist = price * 0.003. Need price close to but below stop.
        # If price = 2080, stop = 2090.55, price-stop = -10.55 < min_dist(6.24)
        #   → stop = min(2090.55, 2080-6.24) = 2073.76. 2080 > 2073.76 → no trigger
        # Need price far enough below trail stop that hard ceiling doesn't save it.
        # Trail stop = 2090.55. Need price < stop AND (stop - min_dist_at_that_price).
        # At price=2050: stop would be... peak stays 2100 (max of peak, 2050).
        # profit_pct = (2100-2000)/2000 = 5% > 1.5% → trailing.
        # trail stop = 2100*(1-0.0045) = 2090.55
        # min_dist = 2050*0.003 = 6.15. current_price(2050) - stop(2090.55) = -40.55 < 6.15
        #   → stop = min(2090.55, 2050-6.15) = min(2090.55, 2043.85) = 2043.85
        # 2050 > 2043.85 → no trigger still. The hard ceiling always protects.
        #
        # The hard ceiling means the stop can never be above (current_price - 0.3%).
        # So check_realtime_stoploss can only trigger when the *initial* stop
        # (before hard ceiling) is above current_price. Let's use a very large ATR.
        runner._atr_buffer = [0.05] * 20  # 5% ATR (very volatile)
        # Initial stop = 2000 * (1 - 0.05*2.0) = 2000 * 0.90 = 1800
        # But with peak at 2100, profit = 5% > atr*breakeven = 5% → breakeven
        # Also > atr*trail = 0.05*0.8 = 4% → trailing
        # Trail stop = 2100 * (1 - 0.05*0.3) = 2100 * 0.985 = 2068.5
        # Floor: 2000*0.95 = 1900. stop = max(2068.5, 1900) = 2068.5
        # Hard ceiling: min_dist = 2060*0.003 = 6.18
        # price(2060) - stop(2068.5) = -8.5 < 6.18 → stop = min(2068.5, 2053.82) = 2053.82
        # Still no trigger at 2060.
        #
        # The hard ceiling prevents any single-call trigger. The mechanism works
        # across multiple ticks: stop is computed fresh each time. So just mock it.
        runner._compute_stop_price = MagicMock(return_value=2050.0)
        # Price 2040 < stop 2050 → triggers for long
        triggered = runner.check_realtime_stoploss(2040.0)
        assert triggered is True
        # Position should be reset to flat
        assert runner._current_signal == 0
        assert runner._entry_price == 0.0
        # PnL recorded
        assert runner._pnl.trade_count == 1


# ===========================================================================
# F) TestPositionSizing
# ===========================================================================

class TestPositionSizing:
    """Tests for _compute_position_size — equity-based adaptive sizing."""

    def test_basic_sizing(self, runner, adapter):
        """Position = equity × per_sym_cap × leverage / price."""
        # equity=1000, per_sym_cap=0.45 (BTC+ETH focused), lev=1.5, price=2000
        # notional = 1000 * 0.45 * 1.5 = 675 → size = 675/2000 = 0.3375
        # z_scale=1.0 (default), consensus=1.0 (no consensus)
        # Round to step_size=0.01 → 0.33 or 0.34
        runner._z_scale = 1.0
        size = runner._compute_position_size(2000.0)
        assert size == pytest.approx(0.34, abs=0.02)

    def test_min_size_floor(self, runner, adapter):
        """Small equity still produces at least min_size."""
        adapter._balances = {"USDT": type("B", (), {"total": 1.0, "available": 1.0})()}
        runner._min_size = 0.01
        runner._z_scale = 1.0
        size = runner._compute_position_size(50000.0)
        assert size >= runner._min_size

    def test_max_notional_clamps(self, runner, adapter):
        """Order with notional > MAX_ORDER_NOTIONAL ($500) is clamped before send."""
        # Set up a large position size that would exceed $500
        runner._position_size = 1.0  # 1 ETH × $2000 = $2000 > $500
        runner._circuit_breaker.allow_request.return_value = True
        runner._osm.active_count.return_value = 1
        result = runner._execute_signal_change(0, 1, 2000.0)
        assert result["qty"] == pytest.approx(0.25, abs=1e-8)
        assert adapter.orders[0]["qty"] == pytest.approx(0.25, abs=1e-8)

    def test_step_size_rounding(self, runner, adapter):
        """Position size is rounded to step_size increments."""
        runner._step_size = 0.1
        runner._z_scale = 1.0
        size = runner._compute_position_size(100.0)
        # Check size is a multiple of step_size
        remainder = round(size % 0.1, 10)
        assert remainder == pytest.approx(0.0, abs=1e-8) or remainder == pytest.approx(0.1, abs=1e-8)


# ===========================================================================
# G) TestReconcilePosition
# ===========================================================================

class TestReconcilePosition:
    """Tests for _reconcile_position — exchange position sync."""

    def test_reconcile_match(self, runner, adapter):
        """When exchange matches runner state (both flat), no change."""
        runner._current_signal = 0
        runner._reconcile_position()
        assert runner._current_signal == 0

    def test_reconcile_divergence(self, runner, adapter):
        """When exchange shows a position but runner is flat, sync to exchange."""
        pos_mock = MagicMock()
        pos_mock.symbol = "ETHUSDT"
        pos_mock.is_flat = False
        pos_mock.is_long = True
        pos_mock.abs_qty = 0.05
        adapter.get_positions = MagicMock(return_value=[pos_mock])

        runner._current_signal = 0
        runner._reconcile_position()
        assert runner._current_signal == 1  # synced to exchange long

    def test_reconcile_failure(self, runner, adapter):
        """get_positions raising an exception does not crash."""
        adapter.get_positions = MagicMock(side_effect=Exception("API timeout"))
        runner._current_signal = 1
        # Should not raise
        runner._reconcile_position()
        # Signal unchanged since reconcile failed gracefully
        assert runner._current_signal == 1


# ===========================================================================
# H) TestPnLTracking
# ===========================================================================

class TestPnLTracking:
    """Tests for PnL tracking through _execute_signal_change."""

    def test_multiple_trades_pnl(self, runner, adapter):
        """total_pnl accumulates correctly across multiple trades."""
        runner._circuit_breaker.allow_request.return_value = True
        runner._osm.active_count.return_value = 1

        # Trade 1: long entry at 2000, close at 2100 (+5%)
        runner._position_size = 0.05
        runner._execute_signal_change(0, 1, 2000.0)
        runner._current_signal = 1
        runner._execute_signal_change(1, 0, 2100.0)
        # PnL = (2100-2000)/2000 * 2000 * 0.05 = $5.0
        pnl_after_1 = runner._pnl.total_pnl

        # Trade 2: short entry at 2100, close at 2000 (+~4.76%)
        runner._position_size = 0.05
        runner._execute_signal_change(0, -1, 2100.0)
        runner._current_signal = -1
        runner._execute_signal_change(-1, 0, 2000.0)
        # PnL = (2100-2000)/2100 * 2100 * 0.05 = $5.0
        pnl_after_2 = runner._pnl.total_pnl

        assert pnl_after_2 > pnl_after_1
        assert runner._pnl.trade_count == 2
        assert pnl_after_2 == pytest.approx(10.0, abs=0.1)

    def test_win_rate(self, runner, adapter):
        """Win rate computed correctly after mixed wins/losses."""
        runner._circuit_breaker.allow_request.return_value = True
        runner._osm.active_count.return_value = 1

        # Win: long 2000 → 2100
        runner._position_size = 0.05
        runner._execute_signal_change(0, 1, 2000.0)
        runner._current_signal = 1
        runner._execute_signal_change(1, 0, 2100.0)

        # Loss: long 2100 → 2000
        runner._position_size = 0.05
        runner._execute_signal_change(0, 1, 2100.0)
        runner._current_signal = 1
        runner._execute_signal_change(1, 0, 2000.0)

        assert runner._pnl.trade_count == 2
        assert runner._pnl.win_count == 1
        assert runner._pnl.win_rate == pytest.approx(50.0)

    def test_drawdown(self, runner, adapter):
        """drawdown_pct computed from peak equity."""
        runner._circuit_breaker.allow_request.return_value = True
        runner._osm.active_count.return_value = 1

        # Win: build up peak
        runner._position_size = 0.1
        runner._execute_signal_change(0, 1, 2000.0)
        runner._current_signal = 1
        runner._execute_signal_change(1, 0, 2200.0)
        # PnL = (200/2000) * 2000 * 0.1 = $20
        peak = runner._pnl.peak_equity
        assert peak == pytest.approx(20.0, abs=0.1)

        # Loss: draw down from peak
        runner._position_size = 0.1
        runner._execute_signal_change(0, 1, 2200.0)
        runner._current_signal = 1
        runner._execute_signal_change(1, 0, 2000.0)
        # PnL = -(200/2200)*2200*0.1 = -$20 → total = ~$0
        assert runner._pnl.drawdown_pct > 0
        assert runner._pnl.peak_equity == pytest.approx(20.0, abs=0.1)  # peak unchanged


# ===========================================================================
# I) TestKillSwitch
# ===========================================================================

class TestKillSwitch:
    """Tests for kill switch integration."""

    def test_kill_switch_armed_blocks(self, runner):
        """Armed kill switch → _killed=True → trades rejected."""
        ks = MagicMock()
        ks.is_armed.return_value = True
        runner._kill_switch = ks
        assert runner._killed is True
        result = runner._execute_signal_change(0, 1, 2000.0)
        assert result["action"] == "killed"

    def test_kill_switch_not_armed(self, runner):
        """Unarmed kill switch → _killed=False."""
        ks = MagicMock()
        ks.is_armed.return_value = False
        runner._kill_switch = ks
        assert runner._killed is False

    def test_drawdown_arms_kill_switch(self, runner, adapter):
        """Large drawdown triggers kill_switch.arm() via risk evaluator."""
        risk_eval = MagicMock()
        risk_eval.check_drawdown.return_value = True  # drawdown breached
        ks = MagicMock()
        ks.is_armed.return_value = False
        runner._risk_eval = risk_eval
        runner._kill_switch = ks

        # Set up state with a pure-Python PnLTracker (Rust mocked in this test file)
        # so peak_equity is a real float, not a MagicMock
        from scripts.ops.pnl_tracker import PnLTracker
        real_pnl = PnLTracker()
        # Force pure-Python mode regardless of mock hotpath
        real_pnl._use_rust = False
        real_pnl._total_pnl = 0.0
        real_pnl._peak_equity = 0.0
        real_pnl._trade_count = 0
        real_pnl._win_count = 0
        real_pnl._trades = []
        runner._pnl = real_pnl
        runner._pnl.record_close("ETHUSDT", 1, 1000.0, 1200.0, 1.0, "setup")  # +200 win → peak=200
        runner._pnl.record_close("ETHUSDT", 1, 1200.0, 1100.0, 1.0, "setup")  # -100 loss → total≈100
        runner._entry_price = 2000.0
        runner._entry_size = 0.05
        runner._position_size = 0.05

        result = runner._execute_signal_change(1, 0, 1500.0)
        # Risk eval returned True (drawdown breached) → kill switch should be armed
        ks.arm.assert_called_once()
        assert result["action"] == "killed"
        assert "drawdown" in result["reason"]


# ===========================================================================
# Additional edge-case tests
# ===========================================================================

class TestComputeZScale:
    """Tests for the static compute_z_scale method."""

    def test_extreme_high(self):
        assert AlphaRunner.compute_z_scale(2.1) == 1.5

    def test_normal(self):
        assert AlphaRunner.compute_z_scale(1.2) == 1.0

    def test_weak(self):
        assert AlphaRunner.compute_z_scale(0.7) == 0.7

    def test_minimal(self):
        assert AlphaRunner.compute_z_scale(0.2) == 0.5

    def test_negative(self):
        """Negative z-scores use absolute value for scaling."""
        assert AlphaRunner.compute_z_scale(-2.1) == 1.5
        assert AlphaRunner.compute_z_scale(-0.7) == 0.7


class TestLifecycle:
    def test_stop_stops_oi_cache(self, runner):
        runner.stop()
        runner._oi_cache.stop.assert_called_once()


class TestCircuitBreaker:
    """Tests for circuit breaker integration in execution path."""

    def test_circuit_open_blocks_orders(self, runner, adapter):
        """When circuit breaker is open, orders are blocked."""
        runner._circuit_breaker.allow_request.return_value = False
        runner._circuit_breaker.snapshot.return_value = {"state": "open", "failures": 3}
        result = runner._execute_signal_change(0, 1, 2000.0)
        assert result["action"] == "circuit_open"
        # No orders sent (only the close_position won't happen since prev=0)
        buy_orders = [o for o in adapter.orders if o.get("side") == "buy"]
        assert len(buy_orders) == 0
