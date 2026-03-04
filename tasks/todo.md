# Research Phase: MaxDD Control + Fair WF + Multi-Asset

## Phase 1: MaxDD Control — Position Management Layer (DONE)

### 1a. Graded Bear Position Sizing
- [x] `_prob_to_score()` — converts bear probability to graded score via threshold list
- [x] `_apply_regime_switch()` — accepts `bear_thresholds` param (None = binary legacy)
- [x] `bridge.py` — `bear_thresholds` param in constructor, graded scoring in `enrich()`
- [x] CLI: `--bear-thresholds '[[0.7,-1.0],[0.6,-0.5],[0.5,0.0]]'`

### 1b. Vol-Adaptive Position Sizing
- [x] `_apply_regime_switch()` — `vol_target` / `vol_feature` params, scales all positions
- [x] `bridge.py` — `vol_target` / `vol_feature` params, applies in `enrich()` after scoring
- [x] CLI: `--vol-target 0.02 --vol-feature atr_norm_14`

### 1c. Drawdown Circuit Breaker
- [x] `_apply_dd_breaker()` — standalone function, tracks equity, forces flat + cooldown
- [x] Applied inside `_apply_regime_switch()` (bear model path) and standalone (non-bear path)
- [x] CLI: `--dd-limit -0.15 --dd-cooldown 48`

## Phase 2: Fair Walk-Forward Validation (DONE)

### 2a. Per-fold V8 Retraining
- [x] `run_fold_strategy_f()` — trains LGBM+XGB ensemble per fold (not fixed production model)
- [x] Feature selection within each fold (fixed + greedy IC)
- [x] HPO optional per fold

### 2b. Strategy F Complete WF
- [x] Bull=per-fold V8 long-only + Bear=per-fold C short, regime-switch per fold
- [x] All Phase 1 features available: `--bear-thresholds`, `--vol-target`, `--dd-limit`
- [x] CLI: `--strategy-f` flag dispatches to `run_fold_strategy_f()`
- [x] Results saved as `wf_{SYMBOL}_strategy_f.json`

## Phase 2.5: Backtest Realism Fixes (DONE)

### Z-Score Lookahead Bias
- [x] Fixed: `_pred_to_signal()` used global `np.mean/std(y_pred)` — future info
- [x] Replaced with rolling-window z-score (720 bars = 30 days, 168 warmup)
- [x] Causal: each bar's z-score uses only past predictions

### Funding Rate Costs
- [x] Backtest: `funding_cost = signal * current_rate / 8.0` deducted from PnL
- [x] WF: same funding deduction added to both `run_fold` and `run_fold_strategy_f`
- [x] Historical average: ~12.6%/year annualized funding cost (BTC perps)

## Phase 3: SOL Multi-Asset Extension (READY)

- [x] SOL data available: `data_files/SOLUSDT_1h.csv` (47k bars, ~5.4 years)
- [x] No code changes needed — WF framework handles `--symbol SOLUSDT` natively
- [x] **SOL Bull-only WF**: 12/17 positive Sharpe → PASS (borderline), Avg Sharpe=-0.68, Return=+121.8%
- [x] **SOL Strategy F WF**: 11/17 positive Sharpe → FAIL, Avg Sharpe=0.97, Return=+105.8%

## Corrected Results (rolling z-score + funding)

### OOS 18-Month (2024-09 → 2026-02)
| Metric | Bull-only | Strategy F |
|--------|-----------|------------|
| Sharpe | 2.67 | 3.56 |
| Return | +19.7% | +54.6% |
| MaxDD | -11.4% | -9.6% |
| Pos Months | 8/18 | 14/18 |

### Walk-Forward (21 folds, per-fold retrained)
| Metric | Bull-only | Strategy F |
|--------|-----------|------------|
| Positive Folds | 14/21 | 15/21 |
| Avg Sharpe | 0.83 | 1.82 |
| Total Return | +33.5% | +63.8% |
| PASS | Yes | Yes |

### Strategy F Value-Add
- Sharpe: +0.99 (0.83 → 1.82)
- Return: +30.3% (33.5% → 63.8%)
- Bear market folds dramatically improved (fold 6: -13.7 → +1.0)

## Phase 4: ABCDE Implementation (DONE)

### A. SOL Walk-Forward (completed)
- [x] SOL Bull-only: 12/17 PASS (borderline), Avg Sharpe=-0.68, Return=+121.8%
- [x] SOL Strategy F: 11/17 FAIL, Avg Sharpe=0.97, Return=+105.8%

### B. Model Hot Reload SIGHUP (completed)
- [x] InferenceEngine.set_models()
- [x] LiveInferenceBridge.update_models() — clears position/hold/close state
- [x] LiveRunner: inference_bridge field + SIGHUP handler + main loop reload check
- [x] 5 unit tests in test_model_hot_reload.py

### C. Production Deployment Hardening (completed)
- [x] Grafana auto-provisioning: datasource + dashboard provider YAMLs
- [x] docker-compose.yml: provisioning volume mount
- [x] Systemd service unit: ExecReload=/bin/kill -HUP, Restart=on-failure
- [x] Logrotate config: daily, rotate 14, maxsize 100M
- [x] Telegram alert sink: 3 unit tests

### D. Test Coverage Config + Gap Analysis (completed)
- [x] pyproject.toml: coverage.run + coverage.report sections
- [x] pytest.ini: fixed testpaths (added tests_unit)
- [x] Baseline coverage: 59% (2152 tests)
- [x] fail_under = 57 (baseline - 2%)
- [x] Targeted tests: risk_gate portfolio (6), health_server (4), feature_hook+bridge (3)

### E. Alpha Research — New Features (completed)
- [x] 3 cross-factor features: liquidation_cascade_score, funding_term_slope, cross_tf_regime_sync
- [x] 3 Deribit IV features: implied_vol_zscore_24, iv_rv_spread, put_call_ratio
- [x] Deribit IV download script + DeribitIVPoller (5min polling)
- [x] ENRICHED_FEATURE_NAMES: 79 → 85

### E.4 IC Analysis + Candidate Pool Update (completed)
- [x] ic_analysis_v9.py: Spearman IC vs 5-bar forward return
- [x] liquidation_cascade_score: IC=0.031, PASS → added to BTC candidate_pool
- [x] funding_term_slope: IC=0.009, FAIL (below 0.02 threshold)
- [x] cross_tf_regime_sync: SKIP (external feature, no multi-TF data in 1h CSV)
- [x] implied_vol_zscore_24, iv_rv_spread, put_call_ratio: SKIP (no Deribit IV data yet)
- [x] BTC Strategy F WF regression: 15/21 PASS, Avg Sharpe=1.82, Return=+63.83% (no degradation)
- [x] Removed 4 duplicate test files (tests_unit/ collisions with tests/unit/)

## Phase 5: ABC 三路线并行实施 (DONE)

### Route A: BTC Testnet Dry-Run 准备
- [x] A.1 docker-compose.yml — testnet-trader service, models_v8 volume, 去掉 monitoring profiles
- [x] A.2 Prometheus scrape target — quant:9090 → live-trader:9090
- [x] A.3 Grafana Dashboard — 8 panels (PnL, P99 Latency, ML Score, Inference Latency, WS State, Rejections, Position, Data Age)
- [x] A.4 Production checklist — tasks/production_checklist.md (5 phases: paper→shadow→live→longrun→compare)
- [x] A.5 Testnet config verified — models_v8/BTCUSDT_gate_v2/config.json has all Strategy F params

### Route B: Deribit IV + Live Pipeline
- [x] B.1 Downloaded Deribit IV — 384 rows → data_files/BTCUSDT_deribit_iv.csv
- [x] B.2 Fixed ic_analysis_v9.py — ISO→epoch ms, on_bar传入implied_vol/put_call_ratio
- [x] B.3 IC Results: implied_vol_zscore_24 IC=-0.1078 PASS, iv_rv_spread IC=0.0801 PASS, put_call_ratio FAIL
- [x] B.4 Wired IV into live pipeline — FeatureComputeHook + LiveRunner + LivePaperRunner + testnet_validation (all 4 phases)
- [x] B.5 Added implied_vol_zscore_24 + iv_rv_spread to candidate_pool (strategy_config.py + config.json)

### Route C: Engineering Quality
- [x] C.1 Fixed embargo.py vars() bug — is_dataclass() + fields() for frozen+slots dataclass
- [x] C.2 3 embargo stamp tests (SimpleNamespace, frozen+slots, regular dataclass)
- [x] C.3 4 DeribitIVPoller tests (before_fetch, parse_hv, put_call_ratio, lifecycle)
- [x] Fixed risk_gate tests (fail-open → fail-closed to match code behavior)

## Verification
- 3215 tests passing, 0 failures
- Coverage baseline: 59%, fail_under=57
- All new functions unit-tested

## CLI Examples

```bash
# OOS bull-only baseline (no bear model)
python3 -m scripts.backtest_alpha_v8 --symbol BTCUSDT --monthly-gate --long-only --bear-model none

# OOS Strategy F (auto-detects position mgmt from config)
python3 -m scripts.backtest_alpha_v8 --symbol BTCUSDT --monthly-gate

# Bull-only WF with funding
python3 -m scripts.walkforward_validate --symbol BTCUSDT --no-hpo \
  --long-only --monthly-gate --ensemble \
  --fixed-features basis ret_24 fgi_normalized fgi_extreme parkinson_vol \
    atr_norm_14 rsi_14 tf4h_atr_norm_14 basis_zscore_24 cvd_20 \
  --candidate-pool funding_zscore_24 basis_momentum vol_ma_ratio_5_20 \
    mean_reversion_20 funding_sign_persist hour_sin \
  --out-dir results/walkforward_fixed

# Strategy F WF with funding + graded bear
python3 -m scripts.walkforward_validate --symbol BTCUSDT --no-hpo \
  --strategy-f \
  --fixed-features basis ret_24 fgi_normalized fgi_extreme parkinson_vol \
    atr_norm_14 rsi_14 tf4h_atr_norm_14 basis_zscore_24 cvd_20 \
  --candidate-pool funding_zscore_24 basis_momentum vol_ma_ratio_5_20 \
    mean_reversion_20 funding_sign_persist hour_sin \
  --bear-thresholds '[[0.7,-1.0],[0.6,-0.5],[0.5,0.0]]' \
  --out-dir results/walkforward_fixed
```

## Backtest Realism + C++ Acceleration (DONE)

### Phase 1: C++ Feature Engine
- [x] `ext/rolling/feature_engine.hpp` — batch feature engine (91 features, 23x speedup on 56k bars)
- [x] `ext/rolling/bindings.cpp` — registered cpp_compute_all_features + cpp_feature_names
- [x] `features/batch_feature_engine.py` — Python wrapper with C++/Python fallback
- [x] `tests/unit/features/test_feature_engine_parity.py` — 91 feature parity tests (atol=1e-10)
- [x] Integration: `backtest_alpha_v8.py`, `train_v7_alpha.py` use batch engine
- [x] Performance: 22.7s → 0.98s on 56k bars (23.2x speedup)

### Phase 2: Realistic Cost Model
- [x] `execution/sim/cost_model.py` — RealisticCostModel with 5 components
  - Trading fees (maker/taker weighted)
  - Market impact (Almgren-Chriss sqrt model)
  - Bid-ask spread (volatility-proportional)
  - Volume participation constraint
  - Funding costs (unchanged)
- [x] `tests/unit/execution/test_cost_model.py` — 10 unit tests
- [x] `--cost-model flat|realistic` flag in backtest_alpha_v8.py + walkforward_validate.py
- [x] 2284 tests passing (0 regressions)
