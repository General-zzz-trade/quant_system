# tests/performance/test_hotpath_regression.py
"""Hot-path performance regression tests.

Ensures critical trading path components do not regress in latency.
Each test uses warmup (not timed) + measurement (timed), with relaxed
thresholds to tolerate CI jitter while catching real regressions.

Run:  pytest tests/performance/test_hotpath_regression.py -v
Skip: pytest -m "not performance"
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

_SCALE = 100_000_000


def _measure(fn, *, warmup: int = 10, iterations: int = 100) -> dict:
    """Run *fn* with warmup, return timing stats in microseconds."""
    for _ in range(warmup):
        fn()

    times: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_us = [t * 1_000_000 for t in times]
    times_us.sort()
    return {
        "avg_us": sum(times_us) / len(times_us),
        "p50_us": times_us[len(times_us) // 2],
        "p99_us": times_us[int(0.99 * len(times_us))],
        "min_us": times_us[0],
        "max_us": times_us[-1],
    }


# ---------------------------------------------------------------------------
# a) RustFeatureEngine: push_bar + get_features
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.benchmark
def test_rust_feature_engine_latency():
    """push_bar + get_features combined must average < 500us."""
    from _quant_hotpath import RustFeatureEngine

    engine = RustFeatureEngine()

    # Pre-fill history so rolling windows are warm
    for i in range(200):
        c = 40000.0 + i * 10.0
        engine.push_bar(c, 500.0 + i, c + 50.0, c - 50.0, c - 10.0,
                        hour=i % 24, dow=i % 7)

    bar_idx = [0]

    def step():
        idx = bar_idx[0]
        c = 42000.0 + (idx % 100) * 5.0
        engine.push_bar(c, 800.0 + idx % 200, c + 30.0, c - 30.0, c - 5.0,
                        hour=idx % 24, dow=idx % 7,
                        funding_rate=0.0001,
                        trades=1200.0,
                        taker_buy_volume=400.0,
                        quote_volume=50000.0,
                        taker_buy_quote_volume=25000.0)
        engine.get_features()
        bar_idx[0] += 1

    stats = _measure(step, warmup=10, iterations=100)
    print(f"\nFeatureEngine push+get: avg={stats['avg_us']:.1f}us "
          f"p99={stats['p99_us']:.1f}us")
    assert stats["avg_us"] < 500, (
        f"FeatureEngine avg {stats['avg_us']:.1f}us > 500us threshold"
    )


# ---------------------------------------------------------------------------
# b) RustStateStore: process_event (market)
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.benchmark
def test_rust_state_store_reduce_latency():
    """RustStateStore.process_event(RustMarketEvent) must average < 100us."""
    from _quant_hotpath import RustStateStore, RustMarketEvent

    store = RustStateStore(["BTCUSDT"], "USDT", int(10000.0 * _SCALE))
    idx = [0]

    def step():
        i = idx[0]
        c = 40000.0 + (i % 200) * 10.0
        ev = RustMarketEvent("BTCUSDT", c - 5.0, c + 20.0, c - 20.0, c, 500.0)
        store.process_event(ev, "BTCUSDT")
        idx[0] += 1

    stats = _measure(step, warmup=10, iterations=100)
    print(f"\nStateStore reduce_market: avg={stats['avg_us']:.1f}us "
          f"p99={stats['p99_us']:.1f}us")
    assert stats["avg_us"] < 100, (
        f"StateStore avg {stats['avg_us']:.1f}us > 100us threshold"
    )


# ---------------------------------------------------------------------------
# c) RustGateChain: process (all gates)
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.benchmark
def test_rust_gate_chain_latency():
    """RustGateChain.process with multiple gates must average < 200us."""
    from _quant_hotpath import RustGateChain

    chain = RustGateChain(True)
    chain.add_gate("equity_leverage", {})
    chain.add_gate("drawdown", {"max_drawdown_pct": 0.20})
    chain.add_gate("correlation", {"max_avg_correlation": 0.70})
    chain.add_gate("alpha_health", {})
    chain.add_gate("regime_sizer", {})
    chain.add_gate("staged_risk", {})

    idx = [0]

    def step():
        i = idx[0]
        ctx = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "signal": 1,
            "qty": 0.05 + (i % 10) * 0.01,
            "price": 40000.0 + (i % 100) * 10.0,
            "equity": 10000.0,
            "peak_equity": 10500.0,
            "drawdown_pct": 0.03 + (i % 5) * 0.005,
            "z_score": 1.5 + (i % 20) * 0.1,
            "avg_correlation": 0.3,
            "alpha_health_scale": 1.0,
            "staged_risk_scale": 0.9,
            "regime_scale": 1.0,
            "consensus": {"BTCUSDT": 1, "ETHUSDT": 1},
        }
        chain.process(ctx)
        idx[0] += 1

    stats = _measure(step, warmup=10, iterations=100)
    print(f"\nGateChain process: avg={stats['avg_us']:.1f}us "
          f"p99={stats['p99_us']:.1f}us")
    assert stats["avg_us"] < 200, (
        f"GateChain avg {stats['avg_us']:.1f}us > 200us threshold"
    )


# ---------------------------------------------------------------------------
# d) EnsemblePredictor: predict
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.benchmark
def test_ensemble_predict_latency():
    """EnsemblePredictor.predict with mock models must average < 1000us (1ms)."""
    from decision.signals.alpha_signal import EnsemblePredictor

    # Build a lightweight mock that mimics Ridge + LGBM predict interface
    class _MockModel:
        """Mimics sklearn/lgbm .predict([x]) -> [float]."""
        def __init__(self, coef: float = 0.001):
            self._coef = coef

        def predict(self, X):
            return [sum(x * self._coef for x in X[0])]

    feature_names = [f"feat_{i}" for i in range(30)]
    horizon_models = [
        {
            "features": feature_names,
            "ridge_features": feature_names[:20],
            "ridge": _MockModel(0.001),
            "lgbm": _MockModel(0.002),
            "ic": 0.05,
            "weight": 1.0,
        },
    ]
    config = {"version": "v8_1h", "ridge_weight": 0.6, "lgbm_weight": 0.4}
    predictor = EnsemblePredictor(horizon_models, config)

    # Build a realistic feature dict
    rng = np.random.RandomState(42)
    base_feats = {f"feat_{i}": float(rng.randn()) for i in range(30)}
    base_feats.update({"rsi_14": 52.0, "ls_ratio": 1.02, "bb_pctb_20": 0.55})

    idx = [0]

    def step():
        # Slightly vary features each call
        i = idx[0]
        feats = dict(base_feats)
        feats["feat_0"] = float(i % 50) * 0.01
        predictor.predict(feats)
        idx[0] += 1

    stats = _measure(step, warmup=10, iterations=100)
    print(f"\nEnsemblePredictor predict: avg={stats['avg_us']:.1f}us "
          f"p99={stats['p99_us']:.1f}us")
    assert stats["avg_us"] < 1000, (
        f"EnsemblePredictor avg {stats['avg_us']:.1f}us > 1000us threshold"
    )


# ---------------------------------------------------------------------------
# e) SignalDiscretizer: discretize
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.benchmark
def test_signal_discretize_latency():
    """SignalDiscretizer.discretize must average < 50us."""
    from _quant_hotpath import RustInferenceBridge
    from decision.signals.alpha_signal import SignalDiscretizer

    bridge = RustInferenceBridge(zscore_window=720, zscore_warmup=180)

    # Warm up the z-score buffer
    for i in range(300):
        bridge.zscore_normalize("BTCUSDT", 0.01 * (i % 50 - 25), i)

    disc = SignalDiscretizer(
        bridge=bridge,
        symbol="BTCUSDT",
        deadzone=1.0,
        min_hold=18,
        max_hold=120,
        long_only=False,
    )

    idx = [0]

    def step():
        i = idx[0]
        pred = 0.01 * ((i % 80) - 40)
        hour_key = 300 + i
        disc.discretize(pred, hour_key, regime_ok=True, current_signal=0)
        idx[0] += 1

    stats = _measure(step, warmup=10, iterations=1000)
    print(f"\nSignalDiscretizer discretize: avg={stats['avg_us']:.1f}us "
          f"p99={stats['p99_us']:.1f}us")
    assert stats["avg_us"] < 50, (
        f"SignalDiscretizer avg {stats['avg_us']:.1f}us > 50us threshold"
    )


# ---------------------------------------------------------------------------
# f) AlphaDecisionModule: full decide() path
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.benchmark
def test_full_decide_latency():
    """AlphaDecisionModule.decide() end-to-end must average < 5000us (5ms)."""
    from _quant_hotpath import RustInferenceBridge
    from decision.signals.alpha_signal import EnsemblePredictor, SignalDiscretizer
    from decision.sizing.adaptive import AdaptivePositionSizer
    from decision.modules.alpha import AlphaDecisionModule

    # --- mock model ---
    class _MockModel:
        def __init__(self, coef: float = 0.001):
            self._coef = coef
        def predict(self, X):
            return [sum(x * self._coef for x in X[0])]

    feature_names = [f"feat_{i}" for i in range(30)]
    horizon_models = [
        {
            "features": feature_names,
            "ridge_features": feature_names[:20],
            "ridge": _MockModel(0.001),
            "lgbm": _MockModel(0.002),
            "ic": 0.05,
            "weight": 1.0,
        },
    ]
    config = {"version": "v8_1h", "ridge_weight": 0.6, "lgbm_weight": 0.4}
    predictor = EnsemblePredictor(horizon_models, config)

    # --- inference bridge + discretizer ---
    bridge = RustInferenceBridge(zscore_window=720, zscore_warmup=180)
    for i in range(300):
        bridge.zscore_normalize("BTCUSDT", 0.01 * (i % 50 - 25), i)

    discretizer = SignalDiscretizer(
        bridge=bridge,
        symbol="BTCUSDT",
        deadzone=1.0,
        min_hold=18,
        max_hold=120,
    )

    # --- sizer ---
    sizer = AdaptivePositionSizer(runner_key="BTCUSDT", step_size=0.001)

    # --- decision module ---
    module = AlphaDecisionModule(
        symbol="BTCUSDT",
        runner_key="BTCUSDT",
        predictor=predictor,
        discretizer=discretizer,
        sizer=sizer,
        leverage=10.0,
    )

    # --- build a minimal snapshot duck-type ---
    rng = np.random.RandomState(42)
    base_feats = {f"feat_{i}": float(rng.randn()) for i in range(30)}
    base_feats.update({"rsi_14": 52.0, "ls_ratio": 1.02, "bb_pctb_20": 0.55})

    class _MockMarket:
        def __init__(self, close: float):
            self.close_f = close
            self.close = close
            self.high_f = close + 50.0
            self.low_f = close - 50.0
            self.high = self.high_f
            self.low = self.low_f

    class _MockSnapshot:
        def __init__(self, close: float, feats: dict):
            self.markets = {"BTCUSDT": _MockMarket(close)}
            self.features = feats
            self.portfolio = None
            self.risk = None

    idx = [0]

    def step():
        i = idx[0]
        close = 40000.0 + (i % 100) * 10.0
        feats = dict(base_feats)
        feats["feat_0"] = float(i % 50) * 0.01
        snap = _MockSnapshot(close, feats)
        list(module.decide(snap))  # consume the generator
        idx[0] += 1

    stats = _measure(step, warmup=10, iterations=100)
    print(f"\nAlphaDecisionModule decide: avg={stats['avg_us']:.1f}us "
          f"p99={stats['p99_us']:.1f}us")
    assert stats["avg_us"] < 5000, (
        f"AlphaDecisionModule avg {stats['avg_us']:.1f}us > 5000us threshold"
    )
