# Phase 3: Alpha Enhancement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Deepen existing symbols + explore new factors on unified LiveRunner path.

**Architecture:** V14 Rust migration, model re-optimization via walk-forward, new factor data pipelines.

**Spec:** `docs/superpowers/specs/2026-03-17-quant-system-improvement-roadmap.md` (Phase 3)

---

## Task 1: V14 Dominance Rust Migration

**Goal:** Move 4 BTC/ETH dominance features from Python (DominanceComputer) to RustFeatureEngine.

**Files:**
- Modify: `ext/rust/src/feature_engine.rs` — add `push_dominance(btc_close, eth_close)` method
- Modify: `ext/rust/src/lib.rs` — expose new method via PyO3
- Modify: `engine/feature_hook.py` — pass eth_close to RustFeatureEngine
- Create: `tests/integration/test_dominance_rust_parity.py`

**Implementation:**
1. Add to RustFeatureEngine: ratio buffer (deque 75), BTC/ETH return buffers (deque 25)
2. `push_dominance()` computes: `btc_dom_ratio_dev_20`, `btc_dom_ratio_mom_10`, `btc_dom_return_diff_6h`, `btc_dom_return_diff_24h`
3. Parity test: Python DominanceComputer vs Rust on 500 bars, diff < 1e-10

**Commit:** `feat(P3-01): migrate V14 dominance features to RustFeatureEngine`

---

## Task 2: Model Re-optimization (BTCUSDT)

**Goal:** Optimize BTC model's crisis deadzone and ranging sub-regime params.

**Files:**
- Create: `scripts/research/optimize_btc_regime.py`
- Modify: `regime/param_router.py` — add ranging sub-regimes if beneficial

**Process:**
1. Run walk-forward with current params as baseline
2. Grid search: crisis deadzone [1.5, 2.0, 2.5, 3.0], ranging_deadzone [0.8, 1.0, 1.2, 1.5]
3. Accept only if Sharpe >= baseline × 0.95 AND no symbol degrades > 5%

**Commit:** `feat(P3-02): optimize BTC regime params via walk-forward grid search`

---

## Task 3: Model Re-optimization (ETH CompositeRegime Evaluation)

**Goal:** Re-evaluate whether CompositeRegime helps ETH (previous conclusion: fixed > adaptive).

**Files:**
- Create: `scripts/research/evaluate_eth_regime.py`

**Process:**
1. Run ETH walk-forward with fixed params (baseline Sharpe 1.52)
2. Run ETH walk-forward with CompositeRegime + ParamRouter enabled
3. Compare: if adaptive Sharpe > fixed Sharpe × 1.05, enable for ETH

**Commit:** `research(P3-03): evaluate CompositeRegime for ETHUSDT`

---

## Task 4: Model Re-optimization (SUI/AXS)

**Goal:** Improve SUIUSDT and AXSUSDT via feature reselection + min_hold tuning.

**Files:**
- Create: `scripts/research/optimize_sui_axs.py`

**Process:**
1. SUI: add V13 OI features, re-run greedy IC selection, walk-forward validate
2. AXS: grid search min_hold [12, 18, 24], feature reselection, validate

**Commit:** `research(P3-04): optimize SUI/AXS features and params`

---

## Task 5: New Factor — Multi-Exchange Funding Rate

**Goal:** Add aggregated funding rate spread as a new factor (P0 priority).

**Files:**
- Create: `scripts/data/download_multi_exchange_funding.py`
- Create: `features/funding_spread.py`
- Create: `tests/unit/features/test_funding_spread.py`

**Implementation:**
1. Download funding rates from Binance + Bybit + OKX for target symbols
2. Compute: funding_spread = max(rates) - min(rates), funding_skew = mean - median
3. IC validation: require IC >= 0.02 AND ICIR >= 0.3
4. If passes, add to PRODUCTION_FEATURES

**Commit:** `feat(P3-05): add multi-exchange funding rate spread factor`

---

## Task 6: New Factor — On-Chain Large Transfers

**Goal:** Add BTC exchange inflow as a sell-pressure signal (P0 priority).

**Files:**
- Create: `scripts/data/download_onchain_flows.py`
- Create: `features/onchain_flow.py`
- Create: `tests/unit/features/test_onchain_flow.py`

**Implementation:**
1. Fetch large transfer data from public API (blockchain.com or similar)
2. Compute: exchange_inflow_zscore (rolling z-score of hourly inflow)
3. IC validation same as Task 5

**Commit:** `feat(P3-06): add on-chain exchange inflow factor`

---

## Task 7: Feature Catalog Update

**Goal:** Update PRODUCTION_FEATURES frozenset with any new validated factors.

**Files:**
- Modify: `features/feature_catalog.py`
- Modify: `features/enriched_computer.py` (if new features need incremental computation)

**Commit:** `feat(P3-07): update PRODUCTION_FEATURES with validated new factors`

---

## Acceptance Criteria

- V14 Rust parity: 500 bars, diff < 1e-10
- Model optimization: >= 2 symbols Sharpe improved >= 10%
- No-degradation gate: No symbol Sharpe degrades > 5%
- >= 1 P0 factor passes IC gate
- `make test` all green
