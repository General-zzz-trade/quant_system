# Pre-Live Safety Audit (2026-03-17)

## 1. API Keys
- [x] No hardcoded keys in source code (verified via grep)
- [x] .env.example documents all required vars
- [x] .env in .gitignore
- [x] systemd service loads EnvironmentFile=/quant_system/.env
- [x] create_adapter() raises RuntimeError if keys missing

## 2. Order Safety
- [x] MAX_ORDER_NOTIONAL = $500 in config.py
- [x] Checked before every send_market_order in alpha_runner.py
- [x] RustCircuitBreaker: 3 failures in 120s → block orders
- [x] RustOrderStateMachine: dedup check (active_count > 2 → block)
- [x] orderLinkId set on every order (Bybit server-side dedup)

## 3. Risk / Drawdown
- [x] RustRiskEvaluator(max_drawdown_pct=0.15) shared across all runners
- [x] RustKillSwitch: global scope, any runner kill → all stop
- [x] PnLTracker: unified tracking, drawdown_pct property
- [x] production.local.yaml: dd_warning=8%, dd_reduce=12%, dd_kill=15%

## 4. Thread Safety
- [x] threading.Lock (_trade_lock) guards signal + order execution
- [x] Lock acquired in both check_realtime_stoploss and process_bar
- [x] Re-check state under lock before executing

## 5. Position Reconciliation
- [x] _reconcile_position() every 10 bars
- [x] Compares runner state vs exchange positions
- [x] On divergence: syncs to exchange truth + WARNING log
- [x] RustStateStore tracks authoritative position state

## 6. Graceful Shutdown
- [x] systemd service: KillSignal=SIGTERM
- [x] runner/operator_control.py handles shutdown command
- [x] Live runner has checkpoint/restore for crash recovery

## 7. Burn-in Gate
- [x] enable_burnin_gate: true in production.local.yaml
- [x] Phase A (Paper, 7d) → Phase B (Shadow, 7d) → Phase C (Testnet, 3d)
- [x] BurninGate blocks live trading until phases pass
- [x] scripts/ops/burnin_status.py shows current status

## 8. Backtest-Live Parity
- [x] WARMUP_BARS = 800 (> zscore_window + warmup)
- [x] Hour_key uses actual timestamps (not sequential index)
- [x] Fees: 7.5bps/side matching Bybit taker rate
- [x] 12/12 Rust constraint parity tests pass

## VERDICT: READY FOR SHADOW MODE (Phase A)
