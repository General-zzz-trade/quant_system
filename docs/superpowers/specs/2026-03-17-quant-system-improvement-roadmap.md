# Quant System Improvement Roadmap

**Date**: 2026-03-17
**Status**: Approved
**Approach**: Conservative — architecture and safety first, 8-10 weeks total
**Methodology**: Incremental — every change keeps tests green, each PR independently verifiable and rollable back

## Context

The quant_system is a production-grade quantitative trading system (297K LOC: 189K Python, 109K Rust) currently running on Bybit Demo with 5 symbols. Deep analysis identified 15 safety/robustness issues, an architectural dual-track risk (AlphaRunner vs LiveRunner), and alpha enhancement opportunities.

### Key Decisions

- **Convergence target**: LiveRunner (framework path with gate_chain, builders, recovery)
- **Alpha strategy**: Deepen existing 5 symbols + explore new factors (no new symbols yet)
- **Live target**: $500 ETHUSDT-only validation (system correctness, not profitability)
- **Change style**: Incremental, test-green at every step, no big-bang rewrites

---

## Phase 1 — Production Hardening (2 weeks)

**Goal**: Fix all 15 identified safety and robustness issues without touching architecture. Each fix is one independent commit.

### Week 1: High Priority (5 fixes)

#### 1. Bybit orderLinkId Collision
- **Problem**: `f"qs_{symbol}_{side[0]}_{int(time.time())}"` — two orders within same second collide
- **Fix**: Change to `f"qs_{symbol}_{side[0]}_{ts_ms}_{random_4hex}"`
- **File**: `execution/adapters/bybit/adapter.py`
- **Test**: Unit test asserting two rapid-fire orders produce distinct keys

#### 2. Cross-Asset Ordering Assertion
- **Problem**: `CrossAssetComputer` requires BTC pushed before altcoins, but no runtime check
- **Fix**: Add warning log + return NaN features when benchmark not yet fed (not hard assert — production-safe degradation). The Rust implementation already returns NaN for unfed features, but Python callers need explicit notification.
- **File**: `features/cross_asset_computer.py`
- **Test**: Unit test verifying altcoin-before-BTC returns NaN features + emits warning

#### 3. Feature Hook Source Exception Isolation
- **Problem**: Exception in any source callable (e.g., `funding_rate_source`) breaks entire bar
- **Fix**: Wrap each source callable in try/except, return NaN on failure, emit alert
- **File**: `engine/feature_hook.py`
- **Test**: Unit test simulating source exception, verifying bar completes with NaN

#### 4. Saga Timeout
- **Problem**: Orders can stay in SUBMITTED state forever (no TTL)
- **Fix**: Add `ttl_seconds` field to `OrderSaga`, `SagaManager.tick()` auto-cancels expired sagas
- **File**: `engine/saga.py`
- **Test**: Unit test verifying SUBMITTED order auto-transitions to CANCELLED after 60s

#### 5. AdapterRegistry Locking
- **Problem**: No thread safety on `_adapters` dict — concurrent registration corrupts state
- **Fix**: Add `threading.RLock` protecting all mutations and reads
- **File**: `execution/adapters/registry.py`
- **Test**: Concurrent registration stress test

### Week 2: Medium Priority (10 fixes)

#### 6. ExecutionBridge Retry + Circuit Breaker
- **Fix**: Integrate existing `CircuitBreakerConfig` + `RetryPolicy` into `ExecutionBridge`
- **File**: `engine/execution_bridge.py`

#### 7. LiveRunnerConfig Schema Validation
- **Fix**: Add `__post_init__` validation for 71 fields (type checks, range assertions). Note: LiveRunnerConfig is `frozen=True` — `__post_init__` can only raise, never assign.
- **File**: `runner/config.py`

#### 8. RiskGate Price Source Consistency
- **Fix**: Require mark_price; reject order if mark_price unavailable (no fallback to order price)
- **File**: `execution/safety/risk_gate.py`

#### 9. EngineLoop Metrics
- **Fix**: Add drop/retry/error counters to MetricsEffect
- **File**: `engine/loop.py`

#### 10. SQLiteDedupStore Atomicity
- **Fix**: Replace INSERT+UPDATE with `INSERT ... ON CONFLICT ... UPDATE` single statement
- **File**: `execution/store/dedup_store.py`

#### 11. Observability Memory Leak
- **Fix**: Add 60s TTL cleanup to `_active` dict in TracingInterceptor
- **File**: `core/observability.py`

#### 12. PositionState Input Validation
- **Fix**: `with_update()` validates qty/price are not NaN/Inf
- **File**: `state/position.py`

#### 13. AccountState Margin Precondition
- **Fix**: Validate balance >= margin_used in `with_update()`
- **File**: `state/account.py`

#### 14. ReplayClock Monotonicity
- **Fix**: `feed(ts)` asserts ts >= last_ts
- **File**: `core/clock.py`

#### 15. Error Classification for Network Errors
- **Fix**: Add ConnectionError/OSError → IO domain mapping
- **File**: `engine/errors.py`

### Phase 1 Acceptance Criteria

- `make test` all green (pytest + cargo test + ruff lint)
- Each fix is one independent commit, independently revertable
- Commit prefix convention: `fix(P1-NN):` (e.g., `fix(P1-03): isolate feature_hook source exceptions`) for easy identification during rollback
- No new files created (except test files)
- Zero architecture changes

---

## Phase 2 — Dual-Track Convergence (3-4 weeks)

**Goal**: Migrate AlphaRunner capabilities into LiveRunner framework. AlphaRunner becomes deprecated fallback.

### Convergence Mapping

```
AlphaRunner Capability              → LiveRunner Target
────────────────────────────────────────────────────────
12 Rust component init              → NEW: rust_components_builder.py (9 stateful components;
                                       RustFillEvent/RustMarketEvent/rust_pipeline_apply are
                                       imports, not initialized)
Ridge+LightGBM ensemble inference   → features_builder.py extension
CompositeRegime + ParamRouter       → RegimeAwareDecisionModule (add BTC logic)
Adaptive stop-loss (ATR 3-phase)    → NEW: AdaptiveStopGate in gate_chain
PortfolioCombiner AGREE             → NEW: combo_builder.py
PnL tracking                        → engine_hook.py extension (reuse existing PnLTracker
                                       class from scripts/ops/pnl_tracker.py, wrap in hook)
Dominance features (V14)            → features_builder.py extension (Python path initially;
                                       replaced by Rust in Phase 3.1)
Kelly leverage ladder               → NEW: EquityLeverageGate in gate_chain (equity-bracket
                                       based: $0-5K@1.5x, $5-20K@1.2x, $20K+@1.0x;
                                       separate from StagedRiskGate which is drawdown-based)
Cross-symbol consensus scaling      → NEW: ConsensusScalingGate in gate_chain (contrarian
                                       +30% boost against consensus, 0.7x on low agreement;
                                       uses shared _consensus_signals dict)
Non-linear z-score position scaling → Absorbed into EquityLeverageGate (z_scale multiplier
                                       applied after leverage bracket lookup)
WS event loop + reconnect           → market_data_builder.py extension
```

### Migration Steps (each step = 1 PR, tests green)

#### Week 1

**Step 1**: New `runner/builders/rust_components_builder.py`
- Initialize all 12 Rust components (RustFeatureEngine, RustInferenceBridge, RustRiskEvaluator, RustKillSwitch, RustOrderStateMachine, RustCircuitBreaker, RustStateStore, RustFillEvent, RustMarketEvent, rust_pipeline_apply, RustUnifiedPredictor, RustTickProcessor)
- LiveRunner opt-in via config flag `enable_rust_components`
- Test: Builder output contains all 12 components

**Step 2**: Extend `features_builder.py` for V14 dominance + cross-asset features
- Align with AlphaRunner's `_compute_features()` logic
- V14 dominance features use existing Python path (temporary — replaced by Rust in Phase 3.1)
- Test: Same bar data → feature values diff < 1e-8 between paths

#### Week 2

**Step 3**: Extend `RegimeAwareDecisionModule` for BTC CompositeRegime
- Add ParamRouter enable flag per symbol (from SYMBOL_CONFIG)
- Test: Same price sequence → regime labels identical

**Step 4**: New `AdaptiveStopGate` in gate_chain
- ATR 3-phase: initial ATR×2.0 → breakeven at 1×ATR profit → trailing at peak-ATR×0.3
- Position in gate_chain: after StagedRiskGate, before PortfolioAllocatorGate
- Test: Phase transitions (initial→breakeven→trailing) with synthetic price data

#### Week 3

**Step 5**: Extend `market_data_builder.py` (must precede combo_builder)
- Multi-interval WS support (kline.60 + kline.15 separate streams)
- Disconnect detection (120s stale threshold) + auto-reconnect + KillSwitch arm
- Test: Simulated WS disconnect → reconnect → kill switch arm sequence

**Step 6**: New `runner/builders/combo_builder.py`
- Wire PortfolioCombiner (AGREE ONLY mode) into LiveRunner
- Depends on Step 5's multi-interval WS infrastructure
- Test: 1h+15m signal AGREE/DISAGREE scenarios

**Step 7**: New gates: `EquityLeverageGate` + `ConsensusScalingGate`
- `EquityLeverageGate`: Equity-bracket leverage ($0-5K@1.5x, $5-20K@1.2x, $20K+@1.0x) + z_scale multiplier. Position in gate_chain: after StagedRiskGate.
- `ConsensusScalingGate`: Cross-symbol consensus scaling (+30% contrarian boost, 0.7x low agreement). Position: after EquityLeverageGate, before PortfolioAllocatorGate.
- Test: Bracket transitions, contrarian vs aligned scenarios

#### Week 4

**Step 8**: End-to-end parity test
- LiveRunner vs AlphaRunner on 500 bars of historical data
- Verify: signal, position, **sizing** (including consensus scale + z_scale), PnL alignment
- Position diff = 0 AND size diff < 1% (accounting for float precision)

**Step 9**: Switch `run_bybit_alpha.py` to LiveRunner
- Add `--legacy` flag to fall back to AlphaRunner
- Mark AlphaRunner as `@deprecated` (retain 3 months)
- Test: systemd service start/stop with LiveRunner

### Key Design Decisions

1. **SYMBOL_CONFIG unchanged** — LiveRunner reads per-symbol params from it, no hardcoding
2. **Gate chain order**: ...existing 13 gates... + AdaptiveStopGate (position 7.5, after StagedRiskGate)
3. **AlphaRunner retained** — `@deprecated`, kept 3 months as fallback, then removed
4. **Rollback**: `--legacy` flag in run_bybit_alpha.py switches to AlphaRunner

### Phase 2 Acceptance Criteria

- LiveRunner on 500 bars history matches AlphaRunner signals exactly
- `make test` all green + ≥30 new convergence comparison tests
- systemd service on LiveRunner stable 24h (Demo)
- No performance regression: LiveRunner per-bar latency ≤ AlphaRunner × 1.2

---

## Phase 3 — Alpha Enhancement (2-3 weeks)

**Goal**: Deepen existing symbols + explore new factors, all on unified LiveRunner path.

### 3.1 V14 Dominance Rust Migration (Week 1)

**Current state**: V14 BTC/ETH dominance features (4 features, BTC model's #1 predictor) computed in Python `enriched_computer.py`, not in RustFeatureEngine.

- **Rust implementation**: `feature_engine.rs` new method `push_dominance(btc_close, eth_close)` computing:
  - `btc_dom_ratio_dev_20`
  - `btc_dom_ratio_mom_10`
  - `btc_dom_return_diff_6h`
  - `btc_dom_return_diff_24h`
- **Python side**: `feature_hook.py` passes `eth_close` to RustFeatureEngine
- **Parity test**: Python vs Rust output diff < 1e-10 over 500 bars

### 3.2 Existing Model Optimization (Weeks 1-2)

All validated via `walkforward_validate.py` (expanding window, ≥ 4 folds).

| Symbol | Current Sharpe | Optimization Direction | Target |
|--------|---------------|----------------------|--------|
| BTCUSDT | 2.03 (V14) / 5.52 (regime) | Crisis deadzone tuning, ranging sub-regime refinement | Baseline Sharpe ≥ 2.5 |
| ETHUSDT | 1.52 (1h) | Re-evaluate CompositeRegime with V14 factors | If Sharpe improves, enable |
| SUIUSDT | 1.63 | Add V13 OI features as data accumulates | Maintain ≥ 1.5 |
| AXSUSDT | 1.25 | Feature reselection + min_hold parameter search | Target ≥ 1.4 |

### 3.3 New Factor Exploration (Weeks 2-3)

Ordered by expected IC contribution and data availability:

| Priority | Factor | Data Source | Expected Value | Complexity |
|----------|--------|------------|----------------|------------|
| P0 | Large on-chain transfers | Blockchain.com / Whale Alert API | Exchange inflow → sell pressure | Low |
| P0 | Aggregated funding rate (multi-exchange) | Multi-exchange funding spread | Cross-exchange arb direction | Low |
| P1 | Options implied volatility | Deribit API (V7 placeholder exists) | IV-RV spread → vol direction | Medium |
| P1 | Liquidation heatmap | Coinglass API | Dense liquidation levels → S/R | Medium |
| P2 | Social sentiment enhancement | LunarCrush / Santiment | Social volume anomaly → reversal | High |

**Validation pipeline**:
1. Download ≥ 3 months historical data
2. Compute single-factor IC (Spearman rank correlation with forward returns)
3. Gate: IC ≥ 0.02 AND ICIR ≥ 0.3 → enter greedy IC selection pool
4. Full walk-forward validation; accept only if Sharpe ≥ baseline × 0.95

### Phase 3 Acceptance Criteria

- V14 Rust migration: parity test 500 bars, diff < 1e-10
- Model optimization: ≥ 2 symbols Sharpe improved ≥ 10% (walk-forward validated)
- **No-degradation gate**: No symbol Sharpe degrades more than 5% from baseline
- New factors: ≥ 1 P0 factor passes IC gate and enters production feature set
- `make test` all green, PRODUCTION_FEATURES frozenset updated

---

## Phase 4 — Live Preparation & Launch (1-2 weeks)

**Goal**: Complete burn-in, go live with $500 ETHUSDT-only for correctness validation.

### 4.1 Burn-in Process (Week 1)

| Phase | Duration | Content | Pass Criteria |
|-------|----------|---------|---------------|
| A — Paper | 3 days | `--dry-run` mode, signals only, no orders | Signal frequency normal (ETH ~2/day), no crash, no error logs |
| B — Shadow | 3 days | Demo API orders, compare signal vs execution | Signal-execution match ≥ 95%, slippage < 5bps, no WS disconnect > 60s |
| C — Testnet | 3 days | Bybit testnet real orders (covers weekday + weekend) | Full flow: order → fill → position → PnL → stop-loss trigger |

Shortened from original 17 days (7+7+3) to 9 days (3+3+3) because:
- Demo has been running for extended period with accumulated data
- LiveRunner already validated via 500-bar parity + 24h Demo stability in Phase 2
- Minimal capital ($500) limits risk; burn-in validates flow, not statistical significance
- Phase C extended to 3 days (vs original 1 day) to cover weekend/off-hours behavior
- **Extension rule**: If any anomaly occurs in Phase C, extend by 3 more days before proceeding

### 4.2 Live Configuration

```yaml
initial_equity: 500
leverage: 0.7                    # half-Kelly
max_order_notional: 100          # single order cap $100 (overrides global MAX_ORDER_NOTIONAL=$500)
symbols: [ETHUSDT]               # most validated symbol only
composite_regime: false           # ETH uses fixed params
staged_risk: survival             # forced survival stage (50% risk fraction, 25% max DD)
```

**Notional precedence**: LiveRunner config `max_order_notional` takes precedence over the global `MAX_ORDER_NOTIONAL` constant ($500) for the live session. Both are checked; the lower value wins.

**Why ETHUSDT only**:
- Sharpe 1.52, 14/20 positive folds, longest validation history
- BTC h=96 holding period too long (min_hold=48 bars = 2 days) for $500
- SUI/AXS have fewer validation folds than ETH

### 4.3 Monitoring & Alerts

| Monitor | Threshold | Action |
|---------|-----------|--------|
| Balance < $400 | 20% DD | KillSwitch HARD_KILL + Telegram alert |
| Single trade loss > $50 | 10% equity | Alert + manual review |
| WS disconnect > 120s | stale | Auto-reconnect; if fails → KillSwitch |
| No bar processed > 2h | stale data | Alert (ETH 1h bars; 2h allows margin for delays. Note: "no signal" ≠ "no bar" — flat signals are normal and can last 18h due to min_hold) |
| Model IC < 0 for 5 consecutive days | alpha decay | AlphaHealthMonitor reduces to 50% |

### 4.4 Live Switch

```bash
# 1. Update .env
BYBIT_BASE_URL=https://api.bybit.com  # demo → live

# 2. Update systemd command (ETHUSDT only)
sudo systemctl edit bybit-alpha.service

# 3. Launch
sudo systemctl restart bybit-alpha.service
tail -f logs/bybit_alpha.log
```

### 4.5 Rollback Plan

| Trigger | Action |
|---------|--------|
| 3 consecutive losses AND cumulative DD > 15% | Manual KillSwitch → switch back to Demo |
| Any unexpected exception (crash, duplicate order, position mismatch) | Stop service → analyze → rollback to AlphaRunner (`--legacy`) |
| 1 week PnL vs backtest deviation > 30% | Pause → run `compare_live_backtest` analysis |

### 4.6 Post-Validation Expansion Path

After 2-4 weeks of successful validation:
1. Add ETHUSDT 15m (dual alpha AGREE mode)
2. Add BTCUSDT (enable CompositeRegime)
3. Add SUI/AXS
4. Gradually increase capital

### Phase 4 Acceptance Criteria

- Burn-in A/B/C all passed
- Live running ≥ 72h with zero crashes, zero duplicate orders, zero position mismatches
- `pre_live_checklist.py` all PASS
- `security_scan.py` all PASS

---

## Timeline Summary

```
Week 1-2:   Phase 1 — Production Hardening (15 fixes)
Week 3-6:   Phase 2 — Dual-Track Convergence (9 steps)
Week 7-9:   Phase 3 — Alpha Enhancement (V14 Rust + optimization + new factors)
Week 10-11: Phase 4 — Burn-in 9 days (3+3+3) + Live Launch ($500 ETHUSDT)
Week 12-15: Post-launch validation + gradual expansion
```

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Phase 2 convergence introduces regression | 500-bar parity test + `--legacy` fallback flag |
| New factors degrade existing models | Walk-forward gate: Sharpe ≥ baseline × 0.95 |
| Live trading unexpected behavior | $500 cap + KillSwitch at 20% DD + 72h validation window |
| Phase 1 fix breaks something | Each fix = 1 commit, independently revertable |
| Timeline overrun | Phases are sequential but independent; can skip Phase 3 and go straight to Phase 4 |
