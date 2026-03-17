# Phase 2: Dual-Track Convergence Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate all AlphaRunner capabilities into LiveRunner's builder framework, making LiveRunner the single production path.

**Architecture:** AlphaRunner's 12 Rust components, regime detection, adaptive sizing, ATR stop-loss, and consensus scaling plug into LiveRunner's existing 12-phase builder architecture via new builders and gates. Each step is one PR with tests.

**Tech Stack:** Python 3.12, pytest, _quant_hotpath (Rust FFI), LiveRunner builder pattern, GateChain

**Spec:** `docs/superpowers/specs/2026-03-17-quant-system-improvement-roadmap.md` (Phase 2 section)

**Key reference files:**
- AlphaRunner source: `scripts/ops/alpha_runner.py` (1242 LOC)
- LiveRunner: `runner/live_runner.py` (1043 LOC)
- Builders: `runner/builders/` (12 files, 2141 LOC)
- Gate chain: `runner/gate_chain.py` (398 LOC)
- Config: `runner/config.py` (231 LOC)
- Run entry: `scripts/ops/run_bybit_alpha.py` (420 LOC)

**Test commands:**
- Single test: `pytest tests/unit/path/test_file.py::TestClass -xvs`
- Runner tests: `pytest tests/unit/runner/ -x -q`
- All: `make test`

---

## Dependency Graph

```
Step 1 (Rust builder) ──┐
Step 2 (Features)    ───┤
Step 3 (Regime)      ───┤──→ Step 8 (Parity) ──→ Step 9 (Switch)
Step 4 (Stop gate)   ───┤
Step 5 (WS multi-TF) ───┤
Step 6 (Combo)       ───┘ (depends on Step 5)
Step 7 (Sizing gates)──→ Step 8
```

Steps 1-5,7 are independent (can be parallel). Step 6 depends on Step 5. Step 8 depends on all. Step 9 depends on Step 8.

---

## Task 1: Rust Components Builder

**Goal:** Create a new builder that initializes all 9 stateful Rust components, making them available to downstream phases.

**Files:**
- Create: `runner/builders/rust_components_builder.py`
- Modify: `runner/live_runner.py` (add Phase 1.5 between core_infra and monitoring)
- Modify: `runner/config.py` (add `enable_rust_components: bool = False`)
- Create: `tests/unit/runner/test_rust_components_builder.py`

**What to build:**
- A `build_rust_components(config, kill_switch)` function that returns a named tuple with:
  - `feature_engine: RustFeatureEngine`
  - `inference_bridge: RustInferenceBridge` (per-symbol dict, keyed by symbol)
  - `risk_evaluator: RustRiskEvaluator`
  - `kill_switch_rust: RustKillSwitch`
  - `order_state_machine: RustOrderStateMachine`
  - `circuit_breaker: RustCircuitBreaker`
  - `state_store: RustStateStore`
- Initialize with params from config and SYMBOL_CONFIG
- Return None components if `enable_rust_components=False`

**Key AlphaRunner code to port (alpha_runner.py lines 37-154):**
- `RustFeatureEngine()` — no args
- `RustInferenceBridge(zscore_window=720, zscore_warmup=180)` — from config
- `RustCircuitBreaker(failure_threshold=3, window_s=120, recovery_timeout_s=60)`
- `RustRiskEvaluator()`, `RustKillSwitch()`, `RustOrderStateMachine()`, `RustStateStore()`

**Tests:**
- Builder returns all components when enabled
- Builder returns None when disabled
- Components are properly initialized (can call basic methods)

**Commit:** `feat(P2-01): add rust_components_builder for LiveRunner`

---

## Task 2: Features Builder Extension (V14 Dominance + Cross-Asset)

**Goal:** Extend features_builder.py to compute V14 dominance features and wire cross-asset features, matching AlphaRunner's `_compute_dominance_features()`.

**Files:**
- Modify: `runner/builders/features_builder.py`
- Modify: `runner/config.py` (add `enable_dominance_features: bool = False`, `dominance_benchmark_symbol: str = "ETHUSDT"`)
- Create: `tests/unit/runner/test_features_builder_v14.py`

**What to build:**
- In `build_features_and_inference()`, when `enable_dominance_features=True`:
  - Create a dominance feature computer that tracks BTC/ETH price ratio
  - After `feat_hook.on_event()`, inject 4 dominance features into the feature dict:
    - `btc_dom_ratio_dev_20`, `btc_dom_ratio_mom_10`
    - `btc_dom_return_diff_6h`, `btc_dom_return_diff_24h`
- This uses the **Python path** (temporary — Phase 3 migrates to Rust)
- Port logic from `alpha_runner.py` lines 234-264

**Key AlphaRunner code to port:**
- `_compute_dominance_features()`: maintains BTC/ETH ratio buffer (75 bars), computes MA deviations
- Called during `process_bar()` after feature engine, before prediction

**Tests:**
- Dominance features present in output when enabled
- Features are NaN when insufficient history (< 20 bars)
- Feature values match manual calculation for known price series

**Commit:** `feat(P2-02): add V14 dominance features to features_builder (Python path)`

---

## Task 3: RegimeAwareDecisionModule BTC CompositeRegime

**Goal:** Extend RegimeAwareDecisionModule to support BTC's CompositeRegimeDetector + ParamRouter, per-symbol enable via config.

**Files:**
- Modify: `decision/regime_bridge.py`
- Modify: `runner/config.py` (add `composite_regime_symbols: tuple[str, ...] = ()`)
- Modify: `runner/builders/decision_builder.py`
- Create: `tests/unit/decision/test_composite_regime_bridge.py`

**What to build:**
- `RegimeAwareDecisionModule` already supports CompositeRegimeDetector
- Add per-symbol composite regime enable: only symbols in `composite_regime_symbols` use CompositeRegime + ParamRouter
- Other symbols use fixed params (existing behavior)
- Port AlphaRunner's 4-layer regime filter logic (alpha_runner.py lines 791-893):
  - Layer 1: Vol + trend filter
  - Layer 2: Ranging detector (efficiency ratio)
  - Layer 3: Dynamic deadzone (vol-scaled)
  - Layer 4: CompositeRegime (opt-in per symbol)

**Key insight:** AlphaRunner's `_check_regime()` has 4 layers but `RegimeAwareDecisionModule` only has CompositeRegimeDetector. Need to add vol/trend/ranging as pre-filters in the decision module.

**Tests:**
- BTC in composite_regime_symbols → uses ParamRouter params
- ETH not in composite_regime_symbols → uses fixed params
- Same price sequence → same regime labels as AlphaRunner

**Commit:** `feat(P2-03): add per-symbol CompositeRegime enable to RegimeAwareDecisionModule`

---

## Task 4: AdaptiveStopGate

**Goal:** Create a new gate for ATR 3-phase adaptive stop-loss, integrated into gate_chain.

**Files:**
- Create: `runner/gates/adaptive_stop_gate.py`
- Modify: `runner/gate_chain.py` (add to gate list)
- Modify: `runner/config.py` (add `enable_adaptive_stop: bool = False`, ATR params)
- Create: `tests/unit/runner/test_adaptive_stop_gate.py`

**What to build:**
- `AdaptiveStopGate` implements the Gate protocol
- Maintains per-symbol state: entry_price, peak_price, atr_buffer, stop_phase
- Three phases (port from alpha_runner.py lines 463-548):
  - **Initial**: stop = entry ∓ atr × 2.0
  - **Breakeven**: triggered when profit > atr × 1.0, stop = entry ± atr × 0.1
  - **Trailing**: triggered when profit > atr × 0.8, stop = peak ∓ atr × 0.3
- Hard limits: max 5% loss floor, min 0.3% distance
- Gate evaluates on each ORDER event:
  - If current price breaches stop → reject order (force exit instead)
  - Scale = 1.0 normally, 0.0 if stop triggered

**Also need:** A `check_realtime_stoploss(symbol, price)` method callable from WS tick handler (outside gate chain).

**Tests:**
- Phase transitions: initial → breakeven → trailing with synthetic prices
- Stop triggered at correct price levels
- Hard limits enforced (5% max loss, 0.3% min distance)
- Gate returns allowed=False when stop triggered

**Commit:** `feat(P2-04): add AdaptiveStopGate with ATR 3-phase stop-loss`

---

## Task 5: Market Data Builder Multi-Interval WS

**Goal:** Extend market_data_builder to support multiple WS intervals (kline.60 + kline.15) on separate streams.

**Files:**
- Modify: `runner/builders/market_data_builder.py`
- Modify: `runner/config.py` (add `multi_interval_symbols: Optional[Dict[str, list]] = None`)
- Create: `tests/unit/runner/test_market_data_multi_interval.py`

**What to build:**
- When `multi_interval_symbols` is set (e.g., `{"ETHUSDT": ["60", "15"]}`):
  - Subscribe to multiple kline intervals per symbol
  - Each interval gets its own WS stream
  - Bar events tagged with interval for downstream routing
- Stale detection per stream (120s threshold)
- Auto-reconnect with exponential backoff
- KillSwitch arm on sustained disconnect (> 5 min)

**Port from run_bybit_alpha.py:**
- Multi-WS setup (lines for kline.60 + kline.15 separate clients)
- `_stale_check()` logic (120s threshold → log warning → kill switch)

**Tests:**
- Multi-interval config creates separate subscriptions
- Simulated disconnect → warning → reconnect
- Stale detection triggers at correct threshold

**Commit:** `feat(P2-05): add multi-interval WS support to market_data_builder`

---

## Task 6: Combo Builder (PortfolioCombiner AGREE)

**Goal:** Wire PortfolioCombiner (AGREE ONLY mode) into LiveRunner via a new builder.

**Files:**
- Create: `runner/builders/combo_builder.py`
- Modify: `runner/live_runner.py` (add combo phase after decision)
- Modify: `runner/config.py` (add `enable_combo: bool = False`, `combo_mode: str = "agree"`)
- Create: `tests/unit/runner/test_combo_builder.py`

**What to build:**
- `build_combo(config, runners_or_signals)` function
- Wraps PortfolioCombiner from `scripts/ops/portfolio_combiner.py`
- AGREE ONLY mode: both 1h and 15m must agree direction
- Conviction: both agree = 100%, one only = 50%
- Per-symbol cap: 30% equity × leverage
- **Depends on Step 5**: needs multi-interval bar routing to separate 1h/15m signals

**Port from portfolio_combiner.py (167 LOC):**
- `PortfolioCombiner.__init__()`: takes two runner references
- `combine_signals()`: AGREE logic
- Position management: single net position on exchange

**Tests:**
- Both agree long → combined signal = long, 100% conviction
- One long, one flat → signal = long, 50% conviction
- Disagree (long vs short) → signal = flat
- Per-symbol cap enforced

**Commit:** `feat(P2-06): add combo_builder for dual-alpha AGREE mode`

---

## Task 7: Equity Leverage Gate + Consensus Scaling Gate

**Goal:** Create two new gates for equity-bracket leverage and cross-symbol consensus scaling.

**Files:**
- Create: `runner/gates/equity_leverage_gate.py`
- Create: `runner/gates/consensus_scaling_gate.py`
- Modify: `runner/gate_chain.py` (add both gates)
- Modify: `runner/config.py` (add `enable_equity_leverage: bool = False`, `enable_consensus_scaling: bool = False`)
- Create: `tests/unit/runner/test_equity_leverage_gate.py`
- Create: `tests/unit/runner/test_consensus_scaling_gate.py`

**EquityLeverageGate:**
- Equity-bracket based leverage (port from alpha_runner.py lines 30-35):
  - $0-5K: 1.5x
  - $5K-20K: 1.5x
  - $20K-50K: 1.0x
  - $50K+: 1.0x
- Z-score position scaling (port from alpha_runner.py lines 683-702):
  - |z| > 2.0: 1.5x
  - |z| > 1.0: 1.0x
  - |z| > 0.5: 0.7x
  - else: 0.5x
- Gate scales order qty by leverage_bracket × z_scale

**ConsensusScalingGate:**
- Cross-symbol consensus (port from alpha_runner.py lines 628-680):
  - All others disagree: 1.3x (contrarian boost)
  - 3/4+ agree: 1.0x
  - 1-2 agree: 0.7x
  - Nobody agrees: 0.5x
- Uses shared `_consensus_signals` dict from config
- Gate scales order qty by consensus_scale

**Tests for EquityLeverageGate:**
- $500 equity → 1.5x leverage
- $25K equity → 1.0x leverage
- z=2.5 → 1.5x z_scale
- z=0.3 → 0.5x z_scale

**Tests for ConsensusScalingGate:**
- 4/4 same direction → 1.0x
- 0/4 agree → 0.5x
- Contrarian (1/4) → 1.3x

**Commit:** `feat(P2-07): add EquityLeverageGate and ConsensusScalingGate`

---

## Task 8: End-to-End Parity Test

**Goal:** Verify LiveRunner produces identical signals, positions, and sizing to AlphaRunner on the same historical data.

**Files:**
- Create: `tests/integration/test_convergence_parity.py`
- May need: helper utilities for replaying bars through both paths

**What to build:**
- Load 500 bars of historical ETHUSDT data from CSV
- Run through AlphaRunner path → collect signals, positions, sizes
- Run through LiveRunner path (with all new builders/gates enabled) → collect same
- Assert:
  - Signal direction matches on every bar
  - Position size diff < 1% (float precision)
  - PnL diff < 0.1%
  - Regime labels match (for BTC composite symbols)

**Test scenarios:**
- ETHUSDT 1h: 500 bars, fixed params
- BTCUSDT 1h: 500 bars, composite regime enabled
- ETHUSDT dual (1h+15m): AGREE mode with 200 bar overlap

**Commit:** `test(P2-08): add end-to-end convergence parity test (AlphaRunner vs LiveRunner)`

---

## Task 9: Switch run_bybit_alpha.py to LiveRunner

**Goal:** Make run_bybit_alpha.py use LiveRunner by default, with `--legacy` flag for AlphaRunner fallback.

**Files:**
- Modify: `scripts/ops/run_bybit_alpha.py`
- Modify: `scripts/ops/alpha_runner.py` (add deprecation warning)
- Create: `tests/unit/scripts/test_run_bybit_alpha_switch.py`

**What to build:**
- Default path: construct LiveRunnerConfig from SYMBOL_CONFIG, build LiveRunner
- `--legacy` flag: fall back to current AlphaRunner path
- Map SYMBOL_CONFIG params to LiveRunnerConfig fields:
  - `composite_regime_symbols` from symbols with `use_composite_regime=True`
  - `multi_interval_symbols` from symbols with `interval="15"`
  - `enable_rust_components=True`, `enable_adaptive_stop=True`, etc.
- Add `@deprecated` decorator/warning to AlphaRunner class

**Tests:**
- Default invocation creates LiveRunner (mock)
- `--legacy` creates AlphaRunner (mock)
- SYMBOL_CONFIG correctly mapped to LiveRunnerConfig

**Commit:** `feat(P2-09): switch run_bybit_alpha.py to LiveRunner, add --legacy fallback`

---

## Final Verification

- [ ] All 9 steps committed
- [ ] `make test` all green
- [ ] LiveRunner on 500 bars matches AlphaRunner signals (Step 8)
- [ ] systemd service starts with LiveRunner (manual test on Demo)
- [ ] `--legacy` flag switches back to AlphaRunner
