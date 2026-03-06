# SOL Model Research V2

## Phase 1: Data Expansion
- [x] Download SOL on-chain metrics (Coin Metrics API: sol) — 403, only 7 basic metrics
- [x] Run IC analysis on on-chain features for SOL — N/A (no data)
- [x] Add viable on-chain features to SOL candidate pool — N/A

## Phase 2: BTC-Lead IC Scan (NEW — SUCCESS)
- [x] Run 35-feature BTC-lead IC scan on SOL 1h data (47,837 bars)
- [x] Identify SOL-specific alpha from BTC features — 15/31 PASS (|IC| > 0.02)
- [x] Top 3: btc_ret_24 (IC=-0.047), btc_rsi_14 (-0.036), btc_mean_reversion_20 (-0.035)
- [x] Extend CrossAssetComputer: 10→17 features (added BTC technicals)
- [x] Update train_v7_alpha.py to pass BTC high/low to CrossAssetComputer
- [x] Update feature_hook.py for live pipeline

## Phase 3: WF Validation
- [x] BTC-lead in candidate pool: 8/17 FAIL (features never selected by greedy IC)
- [x] BTC-lead as FIXED features (no-hpo): **13/17 PASS**, Avg Sharpe=+0.12, Return=+122.3%
- [x] Baseline comparison (no-hpo): 11/17 FAIL, Avg Sharpe=-0.78, Return=+107.4%
- [x] Update strategy_config.py with new SOL config

## Phase 4: Production Deployment
- [x] Train production SOL models with new 13-feature fixed set
- [x] Bootstrap significance test — P(Sharpe>0)=80.6%, CI=[-2.70, 7.03]
- [x] OOS Sharpe=+2.14, IC=0.155 (vs old Sharpe=-1.34, IC=0.038)
- [x] Models saved to models_v8/SOLUSDT_gate_v2/
