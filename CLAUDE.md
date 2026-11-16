# quant_system — project instructions

## Current status (2026-11-13) — read first

- **Not approved for live trading. No strategy has a verified edge.** Older Sharpe/return figures in docs,
  configs, `models_v8/` or commit messages are unverified claims, not results.
- **Forward-test candidates (none has a verified edge):**
  - The risk-managed long BTC/ETH overlay (`lab/overlay_study.py`).
    - It passed a pre-registered development-window screen, which earns forward paper trading only
      (`research/results/overlay_2026-11-13/`). Paper trading is not deployed (owner's decision 2026-11-14).
    - Its `final_spec.json` keeps `"passed": false`, so `runner.binance_target` refuses mainnet.
  - Three candidates frozen 2026-11-14 (`lab/candidates_study.py`,
    `research/results/forward_candidates_2026-11-14/`): the ETH/BTC pair lead, a weekly directional Ridge and a
    liquidation-flush reversal.
    - They only met a "not losing" floor on development data. That is not evidence of an edge: the directional
      model's walk-forward IC is negative.
  - All four are judged at the looks in `research/forward_looks.jsonl` (2027-02-13 report only, 2027-05-13
    decision). For the three new ones a pass only extends the test to 12 months
    (`research/protocols/2026-11-14-forward-candidates.md`).
- **Closed lines:**
  - Hourly ML (`runner.offline research`): the validation-only re-check failed 2026-11-13
    (`research/results/ml_recheck_2026-11-13/`). Its code was deleted 2026-11-15 (M4); the last version is
    the git tag `archive/ml-line-2026-11-15`.
  - The legacy LGBM models in `models_v8/`: the training method selected parameters on its test
    window, and the binaries are not in the repo.
  - ETH 4h breakout at 10x isolated / 1-3x exposure (`lab/breakout_study.py`): failed its
    pre-registered development screen 2026-11-14 (cost x2 return -11%, max drawdown 82%;
    `research/results/breakout_2026-11-14/`). Not to be re-tuned on that window.
  - ETH/BTC relative value, 10 candidates incl. Ridge/LightGBM (`lab/pair_study.py`): failed
    2026-11-14 (deflated Sharpe probability 0.35, below the static short pair;
    `research/results/pair_2026-11-14/`). Per-leg 10x isolated is infeasible for pairs.
  - Mid-frequency on USDT perpetuals (`lab/midfreq_study.py`): 15 symbols, 15 min to 4 h holds, 7 candidates,
    failed 2026-11-14. Every candidate loses 78-85% at taker costs; gross edge per unit traded is 0.004-0.035%
    against a 0.02% maker fee (`research/results/midfreq_2026-11-14/`). No engine work for it.
  - Risk management of the frozen overlay (`lab/overlay_risk_study.py`): 4 predeclared variants (EWMA
    volatility, correlation-aware portfolio cap, drawdown brake, all three) failed 2026-11-15. Only the brake cut
    the drawdown enough, and its Sharpe fell to 0.59 from 0.72 (`research/results/overlay_risk_2026-11-15/`).
    Not to be re-tuned on that window; the frozen overlay is unchanged.
- **Research discipline in code:**
  - `research/trials.jsonl` (`lab/trials.py`) records every configuration evaluated on real
    data; gates use the deflated Sharpe over the whole family. The directional family holds 83 trials.
  - `lab/validation.py`: purged walk-forward splits.
  - `lab/forward.py`: data from 2026-11-13 on is read only at looks registered in
    `research/forward_looks.jsonl` before their end date
    (`research/protocols/2026-11-14-forward-tests.md`).
- **Data windows:**
  - 2021-01-10..2025-11-16 is development data and has been used heavily.
  - 2025-11-16..2026-11-12 has been viewed several times and must not be used for any selection.
  - The next final test is forward data after 2026-11-13.
- **Venue:** Binance USDT-M perpetuals only. The Bybit, OKX and Polymarket code was deleted on 2026-11-15.
- **Defaults:** paper/dry-run; Binance testnet; CI never deploys; deploy is manual.

## Commands

```bash
make test                          # core gate: money-path and execution pytest, ruff, import contracts, mypy
make test-all                      # entire pytest suite (must be all green); CI runs it as 4 shards (python-test-all)
ruff check --select E,W,F .        # lint, whole repo
python scripts/check_local_safety.py               # stdlib-only offline fault-injection suite
python3 -m scripts.account_check                   # live account, read-only GET (the one sanctioned reader of the credentials file)
python3 scripts/dev/mutate.py "<tests>" <mutants.json> [jobs]   # mutation self-check; baseline must pass
python -m lab.overlay_study --out <empty dir>      # overlay screen (development window only)
python -m data.downloads.binance_public --start 2020-01-01   # -> new data_files/binance_public_<end>/
python -m lab.paper_compare --paper-dir <paper dir> --data-dir <snapshot dir>   # paper vs research, fill by fill
python -m trading.infrastructure.state_dir --state-dir <dir> init     # once, before execution mode (RISK_SCOPE set)
python -m trading.infrastructure.kill --state-dir <dir> activate --reason "..." [--flatten]   # or clear / status
python -m trading.infrastructure.entry_pause --state-dir <dir> status            # symbols paused after a stop
python -m trading.infrastructure.entry_pause --state-dir <dir> release --symbol BTCUSDT --reason "..."  # a person only
python -m trading.infrastructure.manual --state-dir <dir> orders     # open orders and the venue's view
python -m trading.infrastructure.manual --state-dir <dir> settle --intent <id> --reason "..."     # runtime stopped
python -m trading.infrastructure.manual --state-dir <dir> position --symbol BTCUSDT --reason "..."  # ledger := venue
```

- **No native code:** the Rust extension `_quant_hotpath` was deleted on 2026-11-15 (its last version is in the
  tag `archive/ml-line-2026-11-15` and in git history); the repository is pure Python.

## Architecture (as actually wired)

**Runtimes:**
- `runner/binance_target.py`: entry point of the overlay strategy, a thin shell (S10) over
  `trading/infrastructure/cli/target.py` (start-up checks, composition of both modes, loop). It supplies what
  `trading` may not import: the signed spec loader and the TCA log writer. It loads no `lab` module.
  - The loop is `trading/application/runtime.py`; it wires `market_poller.py`, `exit_supervisor.py` and `status.py`
    (S3c). In execution mode the three sides are isolated: one that raises blocks every entry until it works again
    and the others still run; one symbol's failing bar blocks only that symbol. One symbol's hourly step is
    `trading/application/session.py`;
    `trading/infrastructure/target.py` wires the sessions.
  - Each closed 1h bar: target (`trading/domain/strategy/overlay.py`) → `trading/domain/rebalance.plan` → orders.
  - Paper mode runs the same execution path (gates, journal, lifecycle, stops, reconciliation) on the simulated
    exchange (`trading/adapters/simulated`: the research brokers, stops by bar high/low; ADR-0013 plan B).
    `ModePolicy` (`trading/infrastructure/modes.py`) is the one list of the differences. Paper uses the code's
    default limits, reads no account setting, has no account check; every intended divergence from research
    (entry refused, stop, pause, kill switch, drawdown flatten) is a `paper_divergence` event; paper has no
    single-order notional cap. Its journal and its state (exchange snapshot, cursors) are in the paper directory's
    `state.sqlite`, saved in the same transactions (S10 decision 4); paper creates it at the first start of a paper
    record. Paper's default directory is `data/runtime/paper` (execution's `data/runtime/binance_target`); paper
    refuses any directory with a manifest (an execution directory), a reset included, and execution refuses a
    paper database. `paper_state.json` is a copy never read back (deleting it resets nothing;
    `--reset-paper-state` sets the database aside).
  - Execution mode sends `OrderIntent`s through `OwnerExecutor` → `ExecutionService` → `OrderLifecycle`,
    behind the risk gates. Its lot rules are the exchange's filters; research and paper keep their constants.
  - Market data comes from Binance REST; it reads no local data files.
- The other runtimes (ML runtime, Bybit/OKX runner, shadow process) and the Docker deploy path were deleted
  on 2026-11-15 (M4). systemd runs three units: `binance-target`, `health-watchdog`, `log-archive`.

**Binance execution:**
- `OrderLifecycle` (`trading/application/execution/lifecycle.py`) keeps durable SQLite intents and never resends;
  it queries instead. `ConfirmedExecutor` and `BinanceExecutionAdapter`, the old interfaces over it, are test
  support now (`tests/support/`): some characterization tests still drive the lifecycle through them.
- Only exchange trade records become fills.
- Exchange results are typed records (`trading/domain/venue.py`). Binance status words and error codes are
  translated in `trading/adapters/binance/adapter.py`; `tests/architecture` refuses them in domain/application.
- `ProtectiveStopManager` keeps one exchange `STOP_MARKET closePosition` per confirmed position, sized and priced from
  the venue position (ADR-0012, S5b); none when the ledger is 0. Kill, drawdown and breached-stop flattening close
  the whole venue position. A reduction's fill beyond the ledger is booked as a foreign fill (`FOREIGN_FILL`).

**Configuration (ADR-0017, S4):** `trading/infrastructure/config.py` is the only module in trading/ and runner/
that reads the environment. The entry point reads it once into a frozen `RuntimeConfig` (credentials apart in
`Secrets`), records it at start-up without secrets (`config` event, `config_digest` in the status) and passes it.

**Risk gates:**
- `CompositeRiskGate` checks entries only. Reductions (`qty == 0` or `reduce_only`) bypass it by design.
- Problems found in a cycle are typed issues (ADR-0014). What each kind does to trading (block a symbol's or
  every entry, flatten, pause, alert only) is one table in `trading/domain/issues.py`. The text is for people:
  `tests/architecture/test_issue_rules.py` forbids text searches in `trading/application`.
- Events are written with an `EventKind` (`trading/domain/events.py`); the JSONL log refuses anything else. Every
  order and stop status change is an `order`/`stop` event (`RecordedOrderJournal`); the journal stays the state.
- The event log is guarded (`trading/application/event_guard.py`): a failed write never stops trading (exits,
  stops and reconciliation go on) but blocks every entry until a write succeeds (`EVENT_LOG_FAILED`).
- Every symbol has a state (`trading/domain/symbol_state.py`: warming, ready, blocked, paused, flattening, halted),
  set by the exit side each cycle and shown in the status. Only `ready` enters (a backstop after the gate).
- State: an execution state directory holds one versioned database, `state.sqlite` (ADR-0009). Only
  `state_dir init` creates it and its tables; the stores open it read-write and never create anything.
- Host lock (ADR-0016 point 5, S6): one execution process per account identity (sha256(network:API key)) on this
  host, `/var/lib/quant/locks/<identity>.lock`, whatever the state directory or `RISK_SCOPE`. The runtime and the
  manual tools take it; `state_dir init` creates the directory. Another host cannot be stopped from here.
- Kill file: the target runtime reads `<state dir>/kill_switch.json` (`trading/adapters/files/kill_switch.py`);
  in an initialized state directory a missing file means active.
- `DrawdownGuard`: daily and weekly latch that resets at UTC period boundaries.
- `AccountExposureGate`: 3x hard cap, checked at entry time only.
- `OrderLimiter` (target runtime, `trading/application/risk/order_limiter.py`): daily counts stored per UTC day
  in the state database; unreadable counts refuse orders. Drawdown limits are capped at 10 % daily / 25 %
  weekly at start-up.

**Research:**
- `lab/`: target-weight strategies with fixed-date windows.
- `trading/adapters/simulated/bar_broker.py`: the research broker (`BarBroker`), its configuration and the research
  lot constants, which `lab.simulate` and paper use (moved from `runner/offline/broker.py` in S10; the old path is
  deleted). The maker limit offset is defined once:
  `trading.domain.orders.LIMIT_OFFSET_BPS`.
- `lab/paper_compare.py` (S11, G6): a paper directory against `lab.simulate` on a dated data snapshot, fill by fill
  (paper records no bars). Exit 0 "一致", 1 "不一致" (first differing fill), 2 "无法比较" (data missing, or not
  what paper saw). A difference is an intended divergence only through a fact of the record plus its
  `paper_divergence` event. Prints fills only, no returns.
- Every study writes its gate before results exist. Protocols live in `research/protocols/`.

**Open exit-side gaps — fix before any testnet or live execution:**
- Unverified: whether Binance accepts a second `closePosition` stop in the same direction. ADR-0012 places the new
  stop before it cancels the old one; if Binance refuses (the old order API had -4130), a new revision's stop is
  never placed and entries stay blocked while the old stop protects. Check on the testnet before M5.
- A failed stop placement blocks entries (step 12) and is sent to Telegram once it has lasted 2 minutes.
- At start-up a warm-up error other than a market-data failure (incomplete or non-contiguous history) still ends
  the process: a person looks before it runs. Once running, failures are isolated (S3c and its audit).
- `binance_target` only:
  - Alerts go to Telegram only when `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are both set; otherwise to the
    process log (status `alerts: "log only"`).

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
  - One PR per migration step, made of small commits (each under 400 handwritten lines, one thing).
  - Locally run the tests related to the change plus ruff, `mypy trading/`, `lint-imports` and
    `tests/architecture`. The whole suite runs in CI (`python-test-all`, four shards from `scripts/ci_shard.py`).
  - GitHub keeps only `master`. Changes reach it through a PR with all CI checks green, then the merged
    branch is deleted.

## Gotchas

- **Fd8 fixed point:** state `.close` and `.last_price` are raw integers (price × 1e8). Use `close_f`
  and `last_price_f`.
- **Bar timestamps:** adapter bar dicts carry `"time"` in seconds and `"start"` in milliseconds.
- **pip:** `pip install` needs `--break-system-packages --ignore-installed` (system Python 3.12).
- **Result files:** `research/results/` matches a `.gitignore` pattern but is tracked. Add new results
  with `git add -f`.
- **Checkout path:** systemd units assume the checkout is at `/quant_system`.
- **Retired units:** removed systemd units are listed in `RETIRED_UNITS` in `infra/sync_systemd.sh`.
  Hosts only stop them after someone runs `sudo bash infra/sync_systemd.sh`.
- **gh 2.45:** `gh pr edit` fails silently (Projects classic deprecation). Use
  `gh api -X PATCH repos/General-zzz-trade/quant_system/pulls/N -F body=@file`.
- **Docs may lag the code:** when they disagree, the code wins; fix the doc.
