# quant_system — project instructions

## Current status (2026-11-13) — read first

- **Not approved for live trading. No strategy has a verified edge.** Older Sharpe/return figures in docs,
  configs, `models_v8/` or commit messages are unverified claims, not results.
- **The only candidate:** the risk-managed long BTC/ETH overlay (`lab/overlay_study.py`).
  - It passed a pre-registered development-window screen, which earns forward paper trading only
    (`research/results/overlay_2026-11-13/`).
  - Its `final_spec.json` keeps `"passed": false`, so `runner.binance_target` refuses mainnet.
- **Closed lines:**
  - Hourly ML (`runner.offline research`): the validation-only re-check failed 2026-11-13
    (`research/results/ml_recheck_2026-11-13/`).
  - The legacy LGBM models in `models_v8/`: the training method selected parameters on its test
    window, and the binaries are not in the repo.
  - ETH 4h breakout at 10x isolated / 1-3x exposure (`lab/breakout_study.py`): failed its
    pre-registered development screen 2026-11-14 (cost x2 return -11%, max drawdown 82%;
    `research/results/breakout_2026-11-14/`). Not to be re-tuned on that window.
- **Data windows:**
  - 2021-01-10..2025-11-16 is development data and has been used heavily.
  - 2025-11-16..2026-11-12 has been viewed several times and must not be used for any selection.
  - The next final test is forward data after 2026-11-13.
- **Venue:** Binance USDT-M perpetuals only. OKX, Bybit and Polymarket code is frozen: safety changes
  only, no features.
- **Defaults:** paper/dry-run; Binance testnet; CI never deploys; deploy is manual.

## Commands

```bash
make test                          # core gate: core + execution pytest, cargo test, ruff (must be all green)
make test-all                      # entire pytest suite (must be all green)
make rust                          # build the Rust crate (maturin, --features python) and pip install it
ruff check --select E,W,F .        # lint, whole repo
python scripts/check_local_safety.py               # stdlib-only offline fault-injection suite
python -m lab.overlay_study --out <empty dir>      # overlay screen (development window only)
python -m runner.offline research --validation-only --symbol BTCUSDT --start 2021-12-02 \
    --train-end 2024-11-20T04:00 --validation-end 2025-11-16T14:00 \
    --data-dir data_files/binance_public --out <empty dir>
python -m data.downloads.binance_public --start 2020-01-01   # -> new data_files/binance_public_<end>/
```

- **Rust builds** need the `python` feature: `maturin build --release --features python`.
- **Shadowing:** a `_quant_hotpath/` directory at the project root shadows the installed package.

## Architecture (as actually wired)

**Runtimes:**
- `runner/binance_target.py`: runtime for the overlay strategy.
  - Each closed 1h bar: `lab.strategy.latest_target` → `lab.rebalance.plan` → orders.
  - Paper mode simulates fills with the research broker (`lab.simulate.Sleeve`). Execution mode uses
    `BinanceExecutionAdapter` behind the risk gates.
  - Market data comes from Binance REST; it reads no local data files.
- `runner/binance_main.py`: ML runtime (`AlphaDecisionModule` with a frozen Ridge bundle). The ML line
  is closed.
- `runner/alpha_main.py`: legacy runner for Bybit/OKX.
  - Dry-run in every unit. It refuses `--venue binance`, and execution is allowed for Bybit only
    (`runner/portfolio_risk.py`).
  - Models load once at startup. There is no hot reload; SIGHUP is only logged.
- `runner/shadow_main.py`: signal observation only, no account access.

**Binance execution:**
- `ConfirmedExecutor` keeps durable SQLite intents and never resends; it queries instead.
- Only exchange trade records become fills.
- `ProtectiveStopManager` keeps one exchange `STOP_MARKET closePosition` per confirmed position.

**Risk gates:**
- `CompositeRiskGate` checks entries only. Reductions (`qty == 0` or `reduce_only`) bypass it by design.
- Kill file (`shared/kill_file.py`); a missing file means inactive.
- `DrawdownGuard`: daily and weekly latch that resets at UTC period boundaries.
- `AccountExposureGate`: 3x hard cap, checked at entry time only.
- `OrderLimiter`: in memory, resets on restart.

**Research:**
- `lab/`: target-weight strategies with fixed-date windows.
- `runner/offline/`: ML replay through the production decision module; the split is at two fixed
  timestamps.
- Every study writes its gate before results exist. Protocols live in `research/protocols/`.

**Open exit-side gaps — fix before any testnet or live execution:**
- Stops and flattening follow the consumed ledger, not the venue position.
- Kill and drawdown flattening wait while an entry order is working.
- A failed stop placement is only logged.
- One failed REST call exits the process.
- `binance_target` only:
  - A failed flatten is not retried until the next bar, because the order id is deterministic per bar.
  - Paper sleeve state is in memory and is lost on restart.
  - There is no paper-vs-replay compare tool.
- `binance_main` only:
  - After 5 failed exits, the runtime rewrites an operator's `flatten: true` kill file to `false`.
  - Execution sizing uses 3x leverage, while research uses 1x.

## Rules for changes

- **No automatic risk escalation or restarts:**
  - Never add an automatic path that raises a risk limit, removes a cap, restarts a live service or
    swaps a model into a running process.
  - Limits change only when a person edits config.
- **Signed artifacts:** model artifacts (pickle and JSON, including `config.json` and strategy specs)
  load only through `infra/model_signing.py`.
- **Fail closed:** unreadable pause, kill or drawdown state blocks new exposure.
- **Zero sizing:** unknown equity or price sizes to zero, never to a minimum lot.
- **Research protocol:**
  - Pre-register the spec, gate and windows in a commit before running a study on real data.
  - Never tune on a viewed final-test window, and record every final-test look.
- **Data snapshots are immutable:**
  - Each download goes into a new dated directory.
  - `data_files/binance_public` is the 2026-11-13 snapshot that every 2026-11-13 study references.
- **Delivery:**
  - Work on a branch and commit per step. Push or merge only when the user asks.
  - GitHub keeps only `master`. Changes reach it through a PR with all CI checks green, then the merged
    branch is deleted.

## Gotchas

- **Fd8 fixed point:** state `.close` and `.last_price` are raw integers (price × 1e8). Use `close_f`
  and `last_price_f`.
- **Bar timestamps:** adapter bar dicts carry `"time"` in seconds and `"start"` in milliseconds.
- **pip:** `pip install` needs `--break-system-packages --ignore-installed` (system Python 3.12).
- **Result files:** `research/results/` matches a `.gitignore` pattern but is tracked. Add new results
  with `git add -f`.
- **Out-of-repo dependencies:**
  - `scripts/training/train_4h_daily.py` runs a script outside the repo (`/home/ubuntu/dev`).
  - Systemd units assume the checkout is at `/quant_system`.
- **Retired units:** removed systemd units are listed in `RETIRED_UNITS` in `infra/sync_systemd.sh`.
  Hosts only stop them after someone runs `sudo bash infra/sync_systemd.sh`.
- **gh 2.45:** `gh pr edit` fails silently (Projects classic deprecation). Use
  `gh api -X PATCH repos/General-zzz-trade/quant_system/pulls/N -F body=@file`.
- **Known pre-existing defect:** `monitoring/daily_reconcile_report.py` has a circular import.
- **Docs may lag the code:** when they disagree, the code wins; fix the doc.
