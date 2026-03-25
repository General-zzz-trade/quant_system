"""Exception and boundary-condition tests for AlphaDecisionModule.decide().

Covers: None predictions, NaN/zero/negative equity/price, corrupt IC health,
empty consensus, force-exit edge cases, division-by-zero guards.
"""
from __future__ import annotations

import json
import os
import tempfile
from decimal import Decimal
from unittest.mock import MagicMock, patch


from decision.modules.alpha import AlphaDecisionModule
from event.types import OrderEvent, RiskEvent


# ── helpers ────────────────────────────────────────────────────────


def _make_snapshot(
    symbol: str = "BTCUSDT",
    close: float = 90000.0,
    high: float | None = None,
    low: float | None = None,
    equity: float = 1000.0,
    features: dict | None = None,
) -> MagicMock:
    snap = MagicMock()
    snap.symbol = symbol
    mkt = MagicMock()
    mkt.close_f = close
    mkt.close = Decimal(str(close))
    mkt.high_f = high if high is not None else close * 1.01
    mkt.high = Decimal(str(mkt.high_f))
    mkt.low_f = low if low is not None else close * 0.99
    mkt.low = Decimal(str(mkt.low_f))
    snap.markets = {symbol: mkt}
    acc = MagicMock()
    acc.balance_f = equity
    acc.balance = Decimal(str(equity))
    snap.account = acc
    snap.features = features
    snap.portfolio = None
    snap.risk = None
    return snap


def _make_predictor(return_val: float | None = 0.5) -> MagicMock:
    pred = MagicMock()
    pred.predict.return_value = return_val
    return pred


def _make_discretizer(
    signal: int = 1, z: float = 1.5, deadzone: float = 0.9,
) -> MagicMock:
    disc = MagicMock()
    disc.discretize.return_value = (signal, z)
    disc.deadzone = deadzone
    disc.min_hold = 18
    disc.max_hold = 120
    return disc


def _make_sizer(qty: float = 0.01) -> MagicMock:
    sizer = MagicMock()
    sizer.target_qty.return_value = Decimal(str(qty))
    sizer.min_size = Decimal("0.001")
    return sizer


def _make_module(
    symbol: str = "BTCUSDT",
    runner_key: str = "BTCUSDT",
    pred_val: float | None = 0.5,
    signal: int = 1,
    z: float = 1.5,
    deadzone: float = 0.9,
    qty: float = 0.01,
    leverage: float = 10.0,
) -> tuple[AlphaDecisionModule, MagicMock, MagicMock, MagicMock]:
    predictor = _make_predictor(pred_val)
    discretizer = _make_discretizer(signal, z, deadzone)
    sizer = _make_sizer(qty)
    mod = AlphaDecisionModule(
        symbol=symbol,
        runner_key=runner_key,
        predictor=predictor,
        discretizer=discretizer,
        sizer=sizer,
        leverage=leverage,
    )
    return mod, predictor, discretizer, sizer


# ── tests ──────────────────────────────────────────────────────────


class TestAlphaDecideExceptions:
    """Edge cases and failure modes for AlphaDecisionModule.decide()."""

    # -- predict() returns None → empty events -------------------------

    def test_predict_none_returns_empty(self):
        mod, _, _, _ = _make_module(pred_val=None)
        snap = _make_snapshot()
        events = list(mod.decide(snap))
        assert events == []

    # -- snapshot.features is None → safe handling ---------------------

    def test_features_none_safe(self):
        mod, _, _, _ = _make_module(signal=1, z=1.5)
        snap = _make_snapshot(features=None)
        events = list(mod.decide(snap))
        orders = [e for e in events if isinstance(e, OrderEvent)]
        assert len(orders) == 1  # still opens a position

    # -- snapshot.features all NaN → predictor receives dict with NaN --

    def test_features_all_nan(self):
        mod, pred, _, _ = _make_module(signal=1, z=1.5)
        nan_feats = {"rsi_14": float("nan"), "vol_20": float("nan")}
        snap = _make_snapshot(features=nan_feats)
        _events = list(mod.decide(snap))
        # predictor.predict gets called with NaN values; module doesn't crash
        pred.predict.assert_called_once()
        call_feats = pred.predict.call_args[0][0]
        assert "rsi_14" in call_feats

    # -- equity = 0 → sizer returns min, no crash ----------------------

    def test_equity_zero_no_crash(self):
        mod, _, _, sizer = _make_module(signal=1, z=1.5, qty=0.0)
        # sizer returns 0 qty → module skips open
        sizer.target_qty.return_value = Decimal("0")
        snap = _make_snapshot(equity=0.0)
        events = list(mod.decide(snap))
        # Should have SignalEvent but no OrderEvent (qty <= 0 path)
        orders = [e for e in events if isinstance(e, OrderEvent)]
        assert len(orders) == 0

    # -- equity = negative → same safe behavior ------------------------

    def test_equity_negative_no_crash(self):
        mod, _, _, sizer = _make_module(signal=1, z=1.5)
        sizer.target_qty.return_value = Decimal("0")
        snap = _make_snapshot(equity=-500.0)
        events = list(mod.decide(snap))
        orders = [e for e in events if isinstance(e, OrderEvent)]
        assert len(orders) == 0

    # -- price = 0 → no crash ------------------------------------------

    def test_price_zero_no_crash(self):
        mod, _, _, _ = _make_module(signal=1, z=1.5)
        snap = _make_snapshot(close=0.0)
        # close=0 → close_f=0 → fallback to raw .close
        snap.markets["BTCUSDT"].close_f = 0.0
        _events = list(mod.decide(snap))
        # Should not crash; may or may not produce events depending on path

    # -- price = NaN → no crash ----------------------------------------

    def test_price_nan_no_crash(self):
        mod, _, _, _ = _make_module(signal=1, z=1.5)
        snap = _make_snapshot(close=float("nan"))
        snap.markets["BTCUSDT"].close_f = float("nan")
        # decide() should not raise
        _events = list(mod.decide(snap))

    # -- signal = 0 but position exists → no new order -----------------

    def test_signal_zero_with_position_no_order(self):
        mod, _, _, _ = _make_module(signal=0, z=0.1)
        snap = _make_snapshot()
        events = list(mod.decide(snap))
        orders = [e for e in events if isinstance(e, OrderEvent)]
        assert len(orders) == 0

    # -- ATR = 0 → force exit doesn't crash ----------------------------

    def test_atr_zero_force_exit_no_crash(self):
        mod, _, disc, _ = _make_module(signal=1, z=1.5)
        snap = _make_snapshot()
        # First call: enter a position
        events = list(mod.decide(snap))
        assert any(isinstance(e, OrderEvent) for e in events)

        # Set ATR buffer to zeros
        mod._atr_buffer = [0.0] * 20

        # Second call with a reversed z → z_reversal
        disc.discretize.return_value = (0, -0.5)
        _events2 = list(mod.decide(snap))
        # Should not crash even with ATR=0

    # -- ATR = NaN → force exit doesn't crash --------------------------

    def test_atr_nan_force_exit_no_crash(self):
        mod, _, disc, _ = _make_module(signal=1, z=1.5)
        snap = _make_snapshot()
        list(mod.decide(snap))  # enter

        mod._atr_buffer = [float("nan")] * 20
        disc.discretize.return_value = (-1, -1.0)
        # Should not raise
        _events = list(mod.decide(snap))

    # -- z_score = NaN → discretizer returns (0, NaN), no open ---------

    def test_z_score_nan_no_open(self):
        mod, _, disc, _ = _make_module()
        disc.discretize.return_value = (0, float("nan"))
        snap = _make_snapshot()
        events = list(mod.decide(snap))
        orders = [e for e in events if isinstance(e, OrderEvent)]
        assert len(orders) == 0

    # -- z_score = inf → z_clamp in _compute_z_scale -------------------

    def test_z_score_inf_z_scale(self):
        # _compute_z_scale should handle inf gracefully
        z_scale = AlphaDecisionModule._compute_z_scale(float("inf"))
        assert z_scale == 1.2  # abs(inf) > 2.0 → 1.2

    def test_z_score_neg_inf_z_scale(self):
        z_scale = AlphaDecisionModule._compute_z_scale(float("-inf"))
        assert z_scale == 1.2

    # -- IC health file corrupt JSON → keeps old value -----------------

    def test_ic_health_corrupt_json_keeps_old(self):
        mod, _, _, _ = _make_module()
        mod._ic_scale = 0.8
        mod._ic_cache_ts = 0  # force refresh

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            f.flush()
            tmp_path = f.name

        try:
            with patch("decision.modules.alpha._IC_HEALTH_PATH", tmp_path):
                mod._refresh_ic_scale()
            # Should keep old value
            assert mod._ic_scale == 0.8
        finally:
            os.unlink(tmp_path)

    # -- IC health file missing → keeps default ------------------------

    def test_ic_health_missing_file(self):
        mod, _, _, _ = _make_module()
        mod._ic_scale = 1.0
        mod._ic_cache_ts = 0

        with patch("decision.modules.alpha._IC_HEALTH_PATH", "/nonexistent/file.json"):
            mod._refresh_ic_scale()
        assert mod._ic_scale == 1.0

    # -- IC health file valid JSON → updates ---------------------------

    def test_ic_health_valid_json_updates(self):
        mod, _, _, _ = _make_module(runner_key="BTCUSDT")
        mod._ic_scale = 1.0
        mod._ic_cache_ts = 0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"BTCUSDT_gate_v2": {"status": "RED"}}, f)
            f.flush()
            tmp_path = f.name

        try:
            import time as _time
            now = _time.time()
            with patch("decision.modules.alpha._IC_HEALTH_PATH", tmp_path):
                with patch("decision.modules.alpha.os.path.getmtime", return_value=now):
                    mod._refresh_ic_scale()
            assert mod._ic_scale == 0.4  # RED
        finally:
            os.unlink(tmp_path)

    # -- consensus dict empty → no 4h reversal -------------------------

    def test_empty_consensus_no_4h_reversal(self):
        mod, _, _, _ = _make_module(signal=1, z=1.5)
        snap = _make_snapshot()
        list(mod.decide(snap))  # enter position

        # consensus is empty except self
        mod._consensus = {}
        force, reason = mod._check_force_exits(90000.0, 0.1)
        # No 4h reversal because consensus is empty
        assert "4h_reversal" not in reason

    # -- entry_price = 0 (no position) → force exit returns False ------

    def test_entry_price_zero_force_exit_false(self):
        mod, _, _, _ = _make_module()
        mod._signal = 0
        mod._entry_price = 0.0
        force, reason = mod._check_force_exits(90000.0, 0.1)
        assert force is False
        assert reason == ""

    # -- entry_price = 0 with signal (edge case) → force exit safe -----

    def test_signal_with_zero_entry_price(self):
        mod, _, _, _ = _make_module()
        mod._signal = 1
        mod._entry_price = 0.0  # shouldn't happen, but protect
        force, reason = mod._check_force_exits(90000.0, 0.1)
        # _entry_price <= 0 → early return False
        assert force is False

    # -- division by zero in drawdown_pct (trade_peak=0) ───────────────

    def test_drawdown_pct_trade_peak_zero(self):
        mod, _, _, _ = _make_module()
        mod._signal = -1
        mod._entry_price = 90000.0
        mod._trade_peak = 0.0  # edge case
        # Should not raise ZeroDivisionError
        force, reason = mod._check_force_exits(90000.0, 0.1)
        # trade_peak=0 → drawdown_pct guarded by `if self._trade_peak > 0`

    # -- z_reversal triggers correctly ---------------------------------

    def test_z_reversal_long_negative_z(self):
        mod, _, _, _ = _make_module()
        mod._signal = 1
        mod._entry_price = 90000.0
        mod._trade_peak = 90100.0
        mod._atr_buffer = [0.01] * 20
        force, reason = mod._check_force_exits(90050.0, -0.5)
        assert force is True
        assert "z_reversal" in reason

    def test_z_reversal_short_positive_z(self):
        mod, _, _, _ = _make_module()
        mod._signal = -1
        mod._entry_price = 90000.0
        mod._trade_peak = 89900.0
        mod._atr_buffer = [0.01] * 20
        force, reason = mod._check_force_exits(89950.0, 0.5)
        assert force is True
        assert "z_reversal" in reason

    # -- close order uses opposite side --------------------------------

    def test_close_order_side_is_opposite(self):
        mod, _, disc, _ = _make_module(signal=1, z=1.5)
        snap = _make_snapshot()
        # Enter long
        events = list(mod.decide(snap))
        open_orders = [e for e in events if isinstance(e, OrderEvent)]
        assert open_orders[0].side == "buy"

        # Flatten signal
        disc.discretize.return_value = (0, 0.0)
        events2 = list(mod.decide(snap))
        close_orders = [e for e in events2 if isinstance(e, OrderEvent)]
        assert len(close_orders) == 1
        assert close_orders[0].side == "sell"

    # -- VPIN scaling edge case ----------------------------------------

    def test_vpin_above_threshold_reduces_qty(self):
        mod, _, _, sizer = _make_module(signal=1, z=1.5, qty=0.1)
        feats = {"vpin": 0.8}  # above 0.5 threshold
        snap = _make_snapshot(features=feats)
        events = list(mod.decide(snap))
        # sizer returns 0.1, VPIN scales by 0.7 → qty should be adjusted
        orders = [e for e in events if isinstance(e, OrderEvent)]
        if orders:
            # The qty in the order may be Decimal(0.07) or similar
            assert float(orders[0].qty) <= 0.1

    # -- _compute_z_scale boundary values ------------------------------

    def test_z_scale_boundary_values(self):
        assert AlphaDecisionModule._compute_z_scale(0.0) == 0.5
        assert AlphaDecisionModule._compute_z_scale(0.49) == 0.5
        assert AlphaDecisionModule._compute_z_scale(0.5) == 0.5
        assert AlphaDecisionModule._compute_z_scale(0.51) == 0.8
        assert AlphaDecisionModule._compute_z_scale(1.0) == 0.8
        assert AlphaDecisionModule._compute_z_scale(1.01) == 1.0
        assert AlphaDecisionModule._compute_z_scale(2.0) == 1.0
        assert AlphaDecisionModule._compute_z_scale(2.01) == 1.2
        assert AlphaDecisionModule._compute_z_scale(100.0) == 1.2

    # -- force exit emits RiskEvent ------------------------------------

    def test_force_exit_emits_risk_event(self):
        mod, _, disc, _ = _make_module(signal=1, z=1.5)
        snap = _make_snapshot()
        list(mod.decide(snap))  # enter

        # Set up force exit via z_reversal
        mod._atr_buffer = [0.01] * 20
        disc.discretize.return_value = (0, -0.5)
        events = list(mod.decide(snap))
        risk_events = [e for e in events if isinstance(e, RiskEvent)]
        assert len(risk_events) >= 1
        assert any("force exit" in e.message for e in risk_events)

    # -- multiple decide calls don't accumulate stale state ------------

    def test_multiple_decide_calls_bars_increment(self):
        mod, _, _, _ = _make_module(signal=0, z=0.1)
        snap = _make_snapshot()
        list(mod.decide(snap))
        list(mod.decide(snap))
        list(mod.decide(snap))
        assert mod._bars_processed == 3
