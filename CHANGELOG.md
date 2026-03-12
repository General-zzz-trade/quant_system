# Changelog

All notable changes to the quant trading system.

## [Unreleased]

### Added
- OOD (out-of-distribution) detection for alpha models
- Concept drift adaptation with automatic recommendations
- Monte Carlo path simulation (bootstrap + parametric)
- Sensitivity analysis framework
- Factor significance tests with multiple testing corrections
- ArgoCD application manifests and canary deployment configuration
- System documentation: README, operations manual, API docs

### Changed
- Established explicit runtime truth, runtime contract, production runbook, execution contract, and model governance documents as current sources of truth
- Rewrote `research.md` against the current codebase rather than historical README assumptions
- Consolidated script-layer signal post-processing into `scripts/signal_postprocess.py`
- Added script catalog / status classification for current, experimental, and legacy-reference entrypoints
- Strengthened live/backtest parity around `deadzone`, `min_hold`, `trend_hold`, `monthly_gate`, and `vol_target`

### Fixed
- `LiveRunner.stop()` cleanup now runs correctly even after `_running` is cleared
- `LiveRunner` startup reconcile now checks account balance drift
- `LiveRunner` perf tuning now tolerates `nohz_full=(null)`
- Missing `data/quality/live_validator.py` restored so default live build path resolves cleanly

## [0.9.0] - 2026-02-28

### Added
- Multi-timeframe strategy framework (aggregator + ensemble)
- SLO/SLI monitoring with error budget tracking
- Data quality alerts (NaN rates, drift detection, constant features)
- Backtest-live PnL comparison tool
- Data lineage tracking with full chain trace
- Backup management with snapshot/restore
- Kelly Criterion allocator (multi-asset, fractional Kelly)
- K8s security hardening (non-root, read-only FS, seccomp, capability drop)
- CI security scanning (Trivy container scan, pip-audit dependency audit)
- External Secrets Operator integration with rotation CronJob
- PodDisruptionBudget and NetworkPolicy for K8s
- Shadow execution mode (simulated orders with real market data)
- Preflight health checks before live trading
- HTTP health endpoint for K8s probes
- Structured JSON logging with configurable log levels

## [0.8.0] - 2026-02-27

### Added
- RiskGate integration into live execution pipeline (position/order notional limits)
- CorrelationComputer activation in live runner with real-time updates
- AttributionTracker for event-level signal attribution
- CorrelationGate order-level risk check (blocks high-correlation entries)
- ModuleReloader for hot-swap of decision modules
- RegimeAwareDecisionModule with regime policy gating
- Alert rules: stale data, high drawdown, kill switch, latency SLA, high correlation

### Fixed
- Decision bridge now only triggers on MARKET events (not FILL/ORDER)
- Fill recording adapter correctly intercepts execution results

## [0.7.0] - 2026-02-26

### Added
- C++ pybind11 acceleration (6 modules)
  - `rolling_window.hpp`: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP (8-15x speedup)
  - `cross_sectional.hpp`: momentum_rank (8x), rolling_beta (76x), relative_strength (14x)
  - `portfolio_math.hpp`: sample_cov (39x), ewma_cov (90x), rolling_corr (60x)
  - `factor_math.hpp`: exposures (22x), factor_model_cov (45x), specific_risk (21x)
  - `feature_selection.hpp`: correlation_select (5x), mutual_info_select (26x)
  - `linalg.hpp`: Black-Litterman posterior (34x), Gauss-Jordan matrix inverse
- Bitget exchange adapter (partial implementation)
- Automatic Python fallback when C++ extensions are not built
- NaN sentinel pattern for efficient Optional[float] handling in C++

## [0.6.0] - 2026-02-25

### Added
- Production hardening: execution safety, observability, backtest integrity
- Order timeout with background timer threads
- Backpressure queue for execution adapter
- Jitter randomization on order submission
- Per-endpoint rate limiting for exchange APIs
- Walk-forward validation with overfit detection
- Combinatorial cross-validation for strategy evaluation
- Backtest-live PnL divergence tracking

## [0.5.0] - 2026-02-24

### Added
- KillSwitch with multi-scope support (GLOBAL, ACCOUNT, STRATEGY, SYMBOL)
- KillSwitchBridge wrapping execution adapters
- MarginMonitor with warning/critical thresholds
- ReconcileScheduler for periodic position reconciliation
- GracefulShutdown with SIGTERM/SIGINT handling
- SystemHealthMonitor (stale data, drawdown alerts)
- LatencyTracker for signal-to-fill measurement
- AlertManager with rule-based alerting and cooldowns

## [0.4.0] - 2026-02-23

### Added
- EngineCoordinator with frozen v1.0 dispatch chain
- StatePipeline as single source of truth for state mutations
- EventDispatcher with PIPELINE/DECISION/EXECUTION/DROP routing
- DecisionBridge connecting strategy modules to engine
- ExecutionBridge as the sole order submission path
- FeatureComputeHook for real-time feature injection
- LiveInferenceBridge for ML model prediction in live loop

## [0.3.0] - 2026-02-22

### Added
- Event type system: MarketEvent, SignalEvent, IntentEvent, OrderEvent, FillEvent, RiskEvent, ControlEvent, FundingEvent
- EventHeader with causal chain tracking (parent/root event IDs)
- StateSnapshot immutable point-in-time capture
- AccountState, MarketState, PositionState, PortfolioState, RiskState
- BinanceRestClient and WebSocket market data runtime
- KlineProcessor for real-time kline aggregation

## [0.2.0] - 2026-02-21

### Added
- Portfolio optimization: Black-Litterman, mean-variance, risk parity objectives
- Constraint system: long-only, max weight, sector exposure, turnover
- Factor model covariance estimation
- Risk budgeting and rebalancing framework
- Alpha model protocol with LightGBM and XGBoost implementations
- Decision engine with signal model, kill overlay, intent validation

## [0.1.0] - 2026-02-20

### Added
- Initial project structure with event-driven architecture
- Core modules: engine, event, state, execution, risk, decision
- Backtest runner with CSV OHLCV input
- Moving average cross strategy (reference implementation)
- Simulated execution adapter for backtesting
- Basic risk aggregation framework
