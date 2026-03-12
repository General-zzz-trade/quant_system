# Quant System Full Audit Report

> Status: historical audit snapshot (2026-03-06)
> Current runtime / contract / governance truth now lives in:
> [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md),
> [`docs/runtime_contracts.md`](/quant_system/docs/runtime_contracts.md),
> [`docs/execution_contracts.md`](/quant_system/docs/execution_contracts.md),
> [`docs/model_governance.md`](/quant_system/docs/model_governance.md),
> and [`research.md`](/quant_system/research.md)

Date: 2026-03-06

## Executive Summary

11-dimension audit across 975 Python files, 138k lines of code.
Overall: production-grade architecture with good foundations, but several
issues need addressing before real-money trading.

**Critical: 4 | High: 8 | Medium: 15 | Low: 12**

---

## CRITICAL ISSUES (Fix immediately)

### C1. Testnet API credentials in .env (Security)
- **File**: `.env` lines 6-7
- **Issue**: Hardcoded Binance testnet API key/secret in version-controlled file
- **Fix**: Revoke credentials, remove from .env, use .env.example template only

### C2. Health server timing attack (Security)
- **File**: `monitoring/health_server.py:44`
- **Issue**: `auth == f"Bearer {token}"` uses `==` instead of `hmac.compare_digest()`
- **Fix**: Replace with `hmac.compare_digest(auth, f"Bearer {token}")`

### C3. Poller HTTP timeout blocks feature pipeline (Performance)
- **Files**: `funding_poller.py:68`, `onchain_poller.py:71`, `macro_poller.py:75`, `deribit_iv_poller.py:67,79`
- **Issue**: Synchronous `urllib.request.urlopen(timeout=10-15)` on daemon threads. If API hangs, feature_hook source() calls on main thread wait for lock release -> 10-15s pipeline stall
- **Fix**: Reduce timeout to 3s; or decouple with lock-free atomic reference pattern

### C4. Regression tests are empty placeholders (Testing)
- **Files**: `tests/regression/test_known_bug_cases.py`, `test_pnl_regression.py`, `test_strategy_regression.py`
- **Issue**: All contain only `assert True`. No actual regression protection for z-score lookahead fix, funding cost fix, DD breaker sign fix
- **Fix**: Implement regression tests for each previously fixed bug

---

## HIGH ISSUES (Fix this week)

### H1. Model signing bypass via env var (Security)
- **File**: `infra/model_signing.py:30-38`
- **Issue**: `QUANT_ALLOW_UNSIGNED_MODELS=1` disables all pickle signature verification. Production must never use this.
- **Fix**: Remove env var bypass in production; enforce QUANT_MODEL_SIGN_KEY

### H2. Liquidation poller unbounded deque (Performance/Memory)
- **File**: `execution/adapters/binance/liquidation_poller.py:40`
- **Issue**: `self._events: Deque = deque()` has no maxlen. In high-volatility markets, can accumulate thousands of events between get_current() calls
- **Fix**: Add `maxlen=10000`

### H3. No position checkpoint on restart (Disaster Recovery)
- **File**: `runner/live_paper_runner.py`
- **Issue**: Paper trader starts flat on restart. All z-score buffers, hold counters, gate history lost. 7-day warmup required again.
- **Fix**: Checkpoint bridge state (z-score buf, close_history, position, hold_counter) to disk periodically; restore on restart

### H4. SOL config missing bear_model_path (Configuration)
- **File**: `infra/config/examples/testnet_sol_gate_v2.yaml`
- **Issue**: No bear_model_path specified. Strategy F regime-switch will silently skip bear model.
- **Fix**: Add `bear_model_path: models_v8/SOLUSDT_bear_c` (or confirm SOL is long-only intentionally)

### H5. 11 of 13 Prometheus alerts reference non-existent metrics (Monitoring)
- **File**: `deploy/prometheus/alerts.yml`
- **Issue**: Alerts for `quant_orders_total`, `quant_margin_ratio`, `quant_kill_switch_active`, `quant_market_data_age_seconds`, etc. reference metrics that are never exported
- **Fix**: Export missing metrics in EngineMonitoringHook / PrometheusExporter

### H6. Alerting not wired to production channels (Monitoring)
- **Files**: `monitoring/alerts/channels.py`, `monitoring/health.py`
- **Issue**: TelegramAlertSink exists but is never instantiated. SystemHealthMonitor defaults to ConsoleAlertSink. No human gets alerts.
- **Fix**: Wire TelegramAlertSink with bot_token env var in LiveRunner startup

### H7. No disk space monitoring (Disaster Recovery)
- **Issue**: Logs and SQLite DB grow unbounded. No rotation configured in Python (logrotate only in systemd deploy). Docker Compose has no limits.
- **Fix**: Use RotatingFileHandler in logging setup; add disk space check in SystemHealthMonitor

### H8. Signal processing logic duplicated 4 times (Maintenance)
- **Files**: `backtest_alpha_v8.py`, `train_v7_alpha.py`, `train_unified.py`, `batch_backtest.py`
- **Issue**: 4 independent `_pred_to_signal()` implementations with subtle behavioral differences (global vs rolling z-score, different warmup handling). Causes backtest-live divergence.
- **Fix**: Consolidate to single shared implementation

---

## MEDIUM ISSUES (Fix this month)

### M1. Undeclared Python dependencies (Dependencies)
- scipy (used in 6 IC analysis scripts), aiohttp, websockets, cvxpy not declared in pyproject.toml
- **Fix**: Add to optional dependency groups

### M2. Python version mismatch (Dependencies)
- Dockerfile uses 3.12, pyproject.toml says >=3.11
- **Fix**: Align to same version

### M3. pybind11 unpinned upper bound (Dependencies)
- `>=2.12` allows breaking 3.x changes
- **Fix**: Pin `>=2.12,<3.0`

### M4. No NaN-ready check before model inference (Data)
- Feature hook passes features dict directly to model; first 65 bars have many NaN features
- Model never saw NaN during training if features were imputed
- **Fix**: Add warmup bar counter; skip inference until minimum features available

### M5. Stale poller data used silently (Data)
- When API is down, pollers return last cached value with no freshness indicator
- **Fix**: Add last_updated timestamp to each poller; log warning if data older than 2x polling interval

### M6. No correlation ID propagation (Monitoring)
- TradeAuditLog schema supports correlation_id but live code never populates it
- **Fix**: Generate UUID on signal generation; propagate through order lifecycle

### M7. Monthly gate boundary: backtest vs live subtle difference (Logic)
- Backtest computes SMA once globally; live accumulates hourly. Minor divergence at hour boundaries.
- **Fix**: Acceptable for now; document the expected divergence

### M8. Data validation gaps (Data)
- MarketReducer accepts price <= 0 without error; no OHLCV range validation
- **Fix**: Add assertions: price > 0, volume >= 0, high >= low

### M9. Model version management chaos (Maintenance)
- 11 feature versions (V0-V11) scattered across code with no central registry
- models_v8/ directory naming mixes model stage + version
- **Fix**: Create MANIFEST.json per model dir; document feature versions

### M10. 50% of scripts likely dead code (Maintenance)
- 14+ training scripts (train_v0 through v5, various one-off research)
- **Fix**: Move inactive scripts to `archive/` directory

### M11. enriched_computer.push() is 242 lines (Maintenance)
- 7-level nesting, 40+ instance attributes, extremely hard to modify safely
- **Fix**: Break into focused sub-methods (funding_update, oi_update, etc.)

### M12. tests_unit/ legacy directory with 84 duplicate files (Testing)
- Overlaps with tests/unit/; causes confusion about which tests are canonical
- **Fix**: Reconcile and remove duplicates

### M13. No end-to-end test with real model inference (Testing)
- Integration tests use mock models only
- **Fix**: Add test that loads actual trained model and runs full pipeline

### M14. Environment variables undocumented (Configuration)
- QUANT_MODEL_SIGN_KEY, QUANT_ALLOW_UNSIGNED_MODELS not in .env.example
- **Fix**: Document all required env vars

### M15. Exception silently swallowed in health monitor (Monitoring)
- `monitoring/health.py:222`: `except Exception: pass` — could lose critical alerts
- **Fix**: Log exception and try fallback sink

---

## LOW ISSUES (Track for future)

### L1. Poller float assignment without lock (Logic)
- funding_poller, fgi_poller, deribit_iv_poller write to `self._rate` without lock
- CPython GIL makes this safe in practice, but not spec-guaranteed
- Risk: Only matters if migrating to PyPy or free-threaded Python

### L2. Cross-asset deque->list() conversions (Performance)
- `cross_asset_computer.py:108,120-122`: `list(self.close_history)[-20:]` copies on every bar
- ~0.5-1ms overhead per bar, not critical

### L3. Debug logging string formatting (Performance)
- `bridge.py:304-307`: Formatting always evaluated even if DEBUG disabled
- Negligible if production uses INFO level

### L4. No JSON size limit on WebSocket messages (Security)
- `json.loads(raw)` without size check could cause memory issues with malformed messages
- Low risk: exchange WS messages are bounded in practice

### L5. Download scripts have identical structure (Maintenance)
- 14 download scripts with same fetch/cache/save pattern
- Could share a DataDownloader base class

### L6. No CLI config overrides (Architecture)
- Must edit YAML to change parameters; no --symbol, --risk-pct flags
- Acceptable for current usage pattern

### L7. No chaos/load testing (Testing)
- No tests for 100k event backpressure, random failure injection
- Nice-to-have for production readiness

### L8. Walk-forward fold embargo gap (Data)
- Internal CV has embargo (horizon + 2 bars) but WF fold train/test boundary has no gap
- Minimal impact since WF folds are months apart

### L9. psycopg2 LGPL license (Dependencies)
- Used in optional timescale backend; LGPL requires care if forking
- Not relevant if not using TimescaleDB

### L10. hold_counter lost on restart (Architecture)
- Signal decay counter resets to 0 on restart; position may close prematurely
- Mitigated by model design (signals are regime-persistent)

### L11. No secondary data source failover (Disaster Recovery)
- Binance-only; no fallback venue for market data
- Acceptable for current scope (single exchange)

### L12. Grafana dashboards not checked in (Monitoring)
- Dashboard generator exists but no pre-built JSON dashboards
- Minor: generate-on-deploy pattern works

---

## VERIFIED FALSE POSITIVES (Agent claims debunked)

### FP1. "Feature order mismatch" — FALSE
- Agent claimed live features arrive in wrong order
- Reality: `lgbm_alpha.py:50` uses `features.get(f, nan) for f in self.feature_names` — dict lookup by name, order irrelevant

### FP2. "Bear model semantics reversed" — FALSE
- Agent claimed bear model returns "long" when crash is likely (should be "short")
- Reality: Bear classifier `side="long"` is a convention meaning "crash signal detected". Bridge.py:273 correctly interprets this: `if bear_sig.side == "long"` → apply bear scoring thresholds. Logic is correct, naming is just unusual.

### FP3. "Z-score circular buffer off-by-one" — FALSE
- Agent claimed `backtest_alpha_v8.py` has wraparound bug in circular buffer
- Reality: Before buffer wraps (`buf_count < zscore_window`), `buf_idx` equals `buf_count`, so `buf[:buf_count]` correctly gets all inserted values. After wrap, code uses full `buf` array. No bug.

---

## ARCHITECTURE STRENGTHS (No action needed)

- Clean event-driven pipeline with immutable state
- Single-threaded event loop ensures determinism
- C++ (23x) and Rust (50-300x) acceleration with Python fallbacks
- Proper WebSocket reconnection (exponential backoff + jitter, 10 retries)
- Poller failure isolation (daemon threads, exception handling)
- Model hot-reload via SIGHUP without restart
- K8s deployment with probes, PDB, leader election, external secrets
- 3540+ tests passing, parity tests for all C++/Rust code
- HMAC model signing for pickle security
- SQL injection prevention (parameterized queries)
- Rate limiting (dual token-bucket, Rust-accelerated)

---

## PRIORITY EXECUTION PLAN

### Phase 1: Security (1 day)
- [ ] C1: Revoke testnet credentials, clean .env
- [ ] C2: Fix health server timing attack
- [ ] H1: Document model signing requirements

### Phase 2: Stability (2 days)
- [ ] C3: Reduce poller HTTP timeouts to 3s
- [ ] H2: Bound liquidation deque (maxlen=10000)
- [ ] H4: Fix SOL config bear_model_path
- [ ] M4: Add warmup bar counter before inference
- [ ] M5: Add poller data freshness tracking
- [ ] M15: Fix silent exception in health monitor

### Phase 3: Monitoring (2 days)
- [ ] H5: Export missing Prometheus metrics
- [ ] H6: Wire Telegram alerts
- [ ] M6: Add correlation ID propagation
- [ ] M14: Document env vars

### Phase 4: Testing (2 days)
- [ ] C4: Implement regression tests for known bugs
- [ ] M12: Clean up tests_unit/ duplicates
- [ ] M13: Add end-to-end test with real model

### Phase 5: Maintenance (3 days)
- [ ] H8: Consolidate _pred_to_signal()
- [ ] M9: Create model version registry
- [ ] M10: Archive dead scripts
- [ ] M11: Refactor enriched_computer.push()

### Phase 6: Infrastructure (2 days)
- [ ] H3: Implement bridge state checkpointing
- [ ] H7: Add disk space monitoring
- [ ] M1-M3: Fix dependency declarations
