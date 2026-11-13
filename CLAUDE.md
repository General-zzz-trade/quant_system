# quant_system — project instructions

## Current status (2026-11-13) — read first

- **Not approved for live trading.** A code-only audit (2026-11-13) found fatal defects in the
  live data path, the exit path and the research methodology. No backtest or Sharpe figure in
  older docs, configs or commit messages is an acceptance result; treat them as unverified.
- **Venue focus: Binance USDT-M perpetuals only.** OKX/Bybit code is kept but frozen (safety fixes
  only). Hard leverage cap: total notional ≤ 3× account equity, enforced in code.
- Default mode is dry-run. Execution needs `--enable-execution` plus explicit persistent runtime
  state (`RISK_SCOPE`, `RUNTIME_STATE_DIR`). Binance defaults to testnet.
- CI never deploys; Deploy is manual only.
- Authoritative docs: `docs/local_development.md`, `docs/offline_validation.md`,
  `docs/execution_recovery.md`.

## Commands

```bash
make rust                          # Build Rust crate (maturin + pip install)
make test                          # Core local gate
python3 -m pytest tests/unit/ -x -q
cargo test                         # Rust unit tests
ruff check --select E,W,F .        # Lint (matches CI gate)
python scripts/check_local_safety.py          # stdlib-only fault-injection checks
python -m runner.offline demo --out <empty dir>   # offline replay/research acceptance
```

Rust builds must enable the `python` feature: `maturin build --release --features python`.
After a build, the `_quant_hotpath/` directory at the project root shadows the installed
package — copy the new `.so` there or remove the stale copy.

## Architecture (as actually wired)

- Entry point: `runner/alpha_main.py` (`--venue`, `--dry-run` default, `--enable-execution`).
  `runner/shadow_main.py` is a separate signal-observation entry with no account access.
- Bar flow: WS kline → `MarketEvent` → `EngineCoordinator.emit` → `FeatureComputeHook`
  (RustFeatureEngine + CSV sources) → `StatePipeline` (RustStateStore) → `DecisionBridge` →
  `AlphaDecisionModule.decide` → `OrderEvent` → `ExecutionBridge` (runtime/data/risk gates) →
  venue execution adapter → `ConfirmedExecutor` (durable SQLite intents, query-never-resend) →
  `FillEvent` → `DurableFillConsumer` (state + strategy position + receipt in one transaction).
- Strategy position is fill-driven: `decide()` never sets `_signal`/`_current_qty` for executing
  runners; `sync_confirmed_position()` does, from confirmed fills.
- Entry risk gates fail closed; reductions (qty=0 or reduce_only) bypass entry gates by design.
- Research: `runner/offline/` is the only methodologically sound replay (time split, next-bar
  fills, fees/slippage/funding, same decision module). `train_v12.py` metrics are in-sample-tuned.

## Rules for changes

- Never add an automatic path that raises a risk limit, removes a cap, or restarts a live
  service. Limits change only by a human editing config.
- Model artifacts (pickle **and** JSON, including `config.json`) load only through
  `infra/model_signing.py` (`load_verified_pickle`, `load_verified_json`, `resolve_artifact`).
- Safety state must fail closed: unreadable pause/kill/drawdown state blocks new exposure.
- Unknown equity/price sizes to zero, never to a minimum lot.
- Do not tune parameters on a final test window; record any final-test look in the protocol.

## Gotchas

- Fd8 fixed point: Python `float * 100_000_000` ↔ Rust i64. Prefer the `*_f` float accessors.
- `pip install` needs `--break-system-packages --ignore-installed` (system Python 3.12, Debian
  packages without RECORD files).
- Adapter bar dicts carry `"time"` in **seconds** and `"start"` in **milliseconds**.
- Several `scripts/*` entries are dangling symlinks into `/home/ubuntu/dev`; they are not part of
  this repository.
