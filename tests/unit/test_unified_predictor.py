"""Tests for RustUnifiedPredictor — zero-copy feature→predict→signal pipeline."""
import os
import pytest

_quant_hotpath = pytest.importorskip("_quant_hotpath")
from _quant_hotpath import RustUnifiedPredictor

# Skip if no model JSON available
MODEL_PATH = "models_v8/BTCUSDT_gate_v2/lgbm_v8.json"
BEAR_PATH = "models_v8/BTCUSDT_bear_c/lgbm_bear.json"
SHORT_PATH = "models_v8/BTCUSDT_short/lgbm_short.json"

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model JSON not available",
)


def _make_predictor(**kw):
    return RustUnifiedPredictor.create([MODEL_PATH], **kw)


def _push_bars(pred, symbol, n, start_close=50000.0):
    result = None
    for i in range(n):
        close = start_close + i * 10
        result = pred.push_bar_and_predict(
            symbol, close=close, volume=1000.0, high=close + 50, low=close - 50,
            open=close - 5, hour_key=i,
            hour=i % 24, dow=i % 7,
            funding_rate=0.0001,
        )
    return result


class TestBasicPrediction:
    def test_create_and_predict(self):
        pred = _make_predictor()
        pred.configure_symbol("BTCUSDT")
        result = _push_bars(pred, "BTCUSDT", 50)
        assert "ml_score" in result
        assert "raw_score" in result
        assert "ml_short_score" in result

    def test_raw_score_is_float(self):
        pred = _make_predictor()
        pred.configure_symbol("BTCUSDT")
        result = _push_bars(pred, "BTCUSDT", 50)
        assert isinstance(result["raw_score"], float)

    def test_short_score_zero_without_short_model(self):
        pred = _make_predictor()
        pred.configure_symbol("BTCUSDT")
        result = _push_bars(pred, "BTCUSDT", 50)
        assert result["ml_short_score"] == 0.0

    def test_multiple_symbols(self):
        pred = _make_predictor()
        pred.configure_symbol("BTCUSDT")
        pred.configure_symbol("ETHUSDT")
        _push_bars(pred, "BTCUSDT", 50, start_close=50000)
        _push_bars(pred, "ETHUSDT", 50, start_close=3000)
        # Both should have independent state
        pos_btc = pred.get_position("BTCUSDT")
        pos_eth = pred.get_position("ETHUSDT")
        assert isinstance(pos_btc, float)
        assert isinstance(pos_eth, float)


class TestConstraints:
    def test_no_constraints_returns_raw(self):
        pred = _make_predictor()
        pred.configure_symbol("BTCUSDT", min_hold=0)
        result = _push_bars(pred, "BTCUSDT", 50)
        # With min_hold=0, ml_score == raw_score
        assert result["ml_score"] == result["raw_score"]

    def test_min_hold_holds_position(self):
        pred = _make_predictor()
        pred.configure_symbol("BTCUSDT", min_hold=48, deadzone=0.5)
        # Push many bars — position should be held for at least min_hold bars
        results = []
        for i in range(200):
            close = 50000.0 + i * 10
            r = pred.push_bar_and_predict(
                "BTCUSDT", close=close, volume=1000.0, high=close + 50,
                low=close - 50, open=close - 5, hour_key=i,
                hour=i % 24, dow=i % 7, funding_rate=0.0001,
            )
            results.append(r["ml_score"])
        # Score should be discretized to {-1, 0, 1}
        unique = set(results)
        assert unique.issubset({-1.0, 0.0, 1.0})


class TestMonthlyGate:
    def test_gate_warmup_allows_trading(self):
        pred = _make_predictor()
        pred.configure_symbol("BTCUSDT", min_hold=1, deadzone=0.01,
                              monthly_gate=True, monthly_gate_window=480)
        # During warmup (< 480 bars), gate should allow trading
        result = _push_bars(pred, "BTCUSDT", 200)
        # Should have a valid score (not necessarily 0)
        assert isinstance(result["ml_score"], float)


@pytest.mark.skipif(not os.path.exists(SHORT_PATH), reason="Short model not available")
class TestShortModel:
    def test_short_model_returns_signal(self):
        pred = RustUnifiedPredictor.create([MODEL_PATH], short_model_path=SHORT_PATH)
        pred.configure_symbol("BTCUSDT", min_hold=48, deadzone=0.5)
        result = _push_bars(pred, "BTCUSDT", 200)
        # Short score should be <= 0 (short-only)
        assert result["ml_short_score"] <= 0.0


@pytest.mark.skipif(not os.path.exists(BEAR_PATH), reason="Bear model not available")
class TestBearModel:
    def test_bear_model_loaded(self):
        pred = RustUnifiedPredictor.create(
            [MODEL_PATH], bear_model_path=BEAR_PATH)
        pred.configure_symbol("BTCUSDT", min_hold=48, deadzone=0.5,
                              monthly_gate=True, monthly_gate_window=480,
                              bear_thresholds=[(0.85, -1.0), (0.75, 0.5)])
        result = _push_bars(pred, "BTCUSDT", 200)
        assert isinstance(result["ml_score"], float)


class TestCheckpoint:
    def test_checkpoint_restore(self):
        pred = _make_predictor()
        pred.configure_symbol("BTCUSDT", min_hold=48, deadzone=0.5)
        _push_bars(pred, "BTCUSDT", 200)

        ckpt = dict(pred.checkpoint())
        assert "BTCUSDT" in ckpt

        pos_before = pred.get_position("BTCUSDT")

        # Create new predictor and restore
        pred2 = _make_predictor()
        pred2.configure_symbol("BTCUSDT", min_hold=48, deadzone=0.5)
        pred2.restore(ckpt)
        assert pred2.get_position("BTCUSDT") == pos_before

    def test_set_position(self):
        pred = _make_predictor()
        pred.configure_symbol("BTCUSDT")
        _push_bars(pred, "BTCUSDT", 10)
        pred.set_position("BTCUSDT", 1.0, 5)
        assert pred.get_position("BTCUSDT") == 1.0


class TestGetFeatures:
    def test_get_features_dict(self):
        pred = _make_predictor()
        pred.configure_symbol("BTCUSDT")
        _push_bars(pred, "BTCUSDT", 50)
        feats = dict(pred.get_features("BTCUSDT"))
        assert "ret_1" in feats
        assert "rsi_14" in feats
        # get_features skips NaN values, so count may be < 105
        assert len(feats) >= 40  # at minimum after 50 bars
        assert len(feats) <= 105
