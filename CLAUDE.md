# quant_system — project instructions

## Current status (2026-11-17) — read first

- **Not approved for live trading. No strategy has a verified edge.** Older Sharpe/return figures in docs,
  configs, `models_v8/` or commit messages are unverified claims, not results.
- **Engineering:** every migration step before M5 (S1–S12) is merged and audited (2026-11-16). The runtime
  extension (`docs/arch/runtime-lab-strategies-plan.md`) is done: R1–R7 (#119–#130), its audit fixes (#126, #134)
  and the docs (R9, #139). 2026-11-17: the spec's `after_stop: pause_then_reset` (#142, audited), the pilot's start
  check `account_check --pilot` (#146), the funding kept in the state database (#147, three audits) and mainnet's
  BTC/ETH hard limit (#153); the lab moved to BTC/ETH (#154).
- **Path to live (owner, 2026-11-16):** no paper and no testnet stage. A strategy that passes the lab's backtest
  checks goes to a small-capital mainnet pilot with hard limits (-3 % a day: `MAX_DAILY_DRAWDOWN_PCT=3`, which a person sets, the code default is 5; a 15 % cumulative loss of the strategy
  capital flattens and stops; kill switch); the owner scales it. This replaces the 4-week testnet run of M5; the checks that could only be made on the testnet
  (a second closePosition stop) are made in the pilot. Still required first: the owner marks the signed spec
  passed (code never does), no position outside the strategy ledger on the account, an alert channel (the alert file) and the heartbeat,
  the host install, and the strategy capital `STRATEGY_CAPITAL_USDT` (mainnet refuses to start without it). The
  cumulative funding is kept in the state database since 2026-11-17 (state database version 2; an existing directory
  is brought there with `state_dir migrate`, the runtime stopped; this host's `/var/lib/quant/state/mainnet` was still
  version 1 on 2026-11-17 and waits for the owner's go-ahead).
- **Before a lab survivor gets money (owner, 2026-11-16 evening):** an idea that passes the lab's checks is frozen and
  tracked on new data for 4 weeks first: one registered look a week (`research/forward_looks.jsonl`, as the forward
  protocol says), each on a new snapshot; it goes to the pilot only if it still holds on all four. Nothing is
  deployed for this. This changes "passes the backtest checks, then the pilot".
- **Calibration pilot (owner, 2026-11-16 evening):** separately, the existing BTC/ETH overlay may run on mainnet to
  collect real fills (prices, slippage, stop fills, refusals) for checking the backtest against reality. The owner
  chose to use the 150 USDT now on the account, not 300: `STRATEGY_CAPITAL_USDT=150`, so the loss stop is at
  22.5 USDT. Later that evening the owner chose 10x isolated margin (`BINANCE_EXCHANGE_LEVERAGE=10`,
  `BINANCE_MARGIN_TYPE=ISOLATED`), a 4.5 % stop (`STOP_PCT=0.045`; 10x isolated allows at most 4.75 %) and positions
  of 2 to 3 times the capital (`STRATEGY_LEVERAGE`, which the owner writes; the hard cap was 3 and is 10 since
  2026-11-17, see below). The backtest of that
  choice (`research/results/overlay_pilot_leverage_2026-11-16/`, development window) had 2x best: +69 %, Sharpe
  0.54, max drawdown 23 %; 3x: +19 %, Sharpe 0.30, max drawdown 46 %. Exchange minimums limit an order's notional,
  not its margin: with leverage BTC trades too (the 1x sizing was what kept it out, not the exchange).
  It has no verified edge and none is expected; the 15 % loss stop bounds the cost. Every "still required first"
  item above applies, the account must hold nothing outside the strategy ledger (a manual position on BTCUSDT or
  ETHUSDT would be netted with the strategy's; the loss stop itself counts only the ledger's positions), and the
  owner marks the spec.
- **Shared account (owner, 2026-11-16):** the strategy keeps sharing the main account. It sizes from
  min(account equity, `STRATEGY_CAPITAL_USDT` + its own cumulative realized PnL after USDT fees); the owner chose
  that this base grows with the strategy's profits. The PnL includes funding (tech lead, 2026-11-16): the USDT
  funding income of each strategy symbol since the strategy's first fill on it, from Binance's income history, read
  at most every 10 minutes; funding never read or older than 2 hours sizes the base to zero. Binance keeps that
  history for three months only, so every funding row read is kept in the state database (version 2, tech lead
  2026-11-17) and each refresh reads from 24 hours before the kept span ends; only a gap longer than Binance's three
  months (the runtime down that long) leaves the funding unknown, which sizes the base to zero and leaves the loss
  stop's total unknown too (every entry blocked).
  Every `target.*` owner in the scope's ledger counts, also one whose symbol the current spec no longer trades.
  A zero base plans no partial reduction, only a whole close (the zero-sizing rule). Fees paid in BNB are not in
  the PnL (BNB fee payment is off on the account, checked 2026-11-16).
  The drawdown latch, the -3 % daily stop and the account exposure cap stay on the whole account's equity (owner's
  choice: with a large account and a small strategy they rarely trigger on the strategy's losses alone).
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
  - The lab champion of 2026-11-17, `E3_vol_target_trend` (long only, sized by volatility; 8 % stop, re-entry after
    the signal resets), unverified: walk-forward Sharpe 0.62, holdout +8.2 % over 6 months, but a deflated Sharpe
    probability of 0.03 over 1092 configurations (`research/lab/runs/2026-11-17/judge.md`). Frozen
    (`research/lab/runs/2026-11-17/forward_e3.json`) and tracked from 2026-11-18 at 4 weekly registered looks
    (2026-11-25, 12-02, 12-09, 12-16; `lab/um_forward.py`). It still holds while the lowest hourly equity since the
    start stays above 92.5 % of the capital (owner 2026-11-17); only after 4 holds may a pilot be prepared, with a
    spec that carries `after_stop: pause_then_reset` (`research/protocols/2026-11-17-lab-forward-e3.md`).
    - Its frozen spec lists ten whole-market symbols (ETH, SOL, XAU, CL, XAG, XRP, DOGE, ZEC, HYPE, 1000PEPE) and no
      BTC; mainnet refuses any symbol but BTCUSDT and ETHUSDT (#153), so as frozen it cannot become a pilot. The looks
      have no scheduled task; whether tracking goes on as a research record is open (owner, 2026-11-17).
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
  - Cross-sectional long/short on the 10 most liquid crypto perps (`lab/xs_study.py`, owner 2026-11-17 "开广度和成本"):
    momentum, carry, low volatility, reversal and a BTC trend/range switch, long 3 / short 3 daily at 1500 USDT
    notional, 2x, the 8 % stop; failed 2026-11-17, every walk-forward lost 69-99 % (`research/lab/runs/2026-11-17e/`).
    Gross of costs and stops momentum and carry made Sharpe 0.74 (a diagnostic written after the results): the
    trading lost the money. Reversal was worse in ranges than in trends, so the switch did not help. Not to be
    re-tuned on the development window. Maker entries were not better by their registered rule (2026-11-17d), so
    the lab stays at market entries; the replay can take maker entries (`execution`, #161).
  - Parameter search on the pilot (owner 2026-11-17, "来开始探索交易的参数数值设置多少最佳", scopes A and B,
    `lab/pilot_params.py`). Both failed their registered adoption rules, so nothing changes; neither is to be re-tuned
    on the development window.
    - A, the risk and execution settings: 60 settings of position multiple, stop/exchange leverage, after-stop rule and
      latches (`research/results/pilot_params_2026-11-17/`). The year-by-year choice made Sharpe 0.29, against 0.78 for
      today's settings on the same years, so today's settings stay.
    - B, the overlay's signal settings: 72 settings of lookbacks, target volatility, band and signal form
      (`research/results/overlay_signal_2026-11-17/`). The year-by-year choice made Sharpe 0.36, against 0.78 for the
      frozen spec, and lost at double costs, so the frozen spec stays.
    - In both, results swing with the path: a cost change alone moves the latch and stop timing.
  - The panel model as a filter on a channel breakout (owner 2026-11-17 "跑回测验证", `lab/vol_breakout.py`): 20-day
    breakout, 10-day/14-day exits, the 8 % stop with pause_then_reset, top-100 perps, 10 slots of 150 USDT at 2x;
    the filter (top fifth by trend_14) and predicted-volatility sizing, 2x2. Failed 2026-11-18
    (`research/results/vol_breakout_2026-11-17/`): accepted breakouts beat rejected ones by 1.4 points a trade, Welch t
    0.46 < 2; sizing did not help; every configuration lost money (fixed 150 USDT a trade: -1308 to -2474 USDT), the
    shorts most. Not to be re-tried with other rules on the development window. Its account sum let the losses of
    hundreds of contracts pass the capital (money below zero in two configurations): with far fewer slots than
    contracts, replay by slot or let the stake shrink with the account.
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
- **Research scope (owner 2026-11-17, evening):** "我要求不只做BTC和ETH，全市场的永续合约USDT都可以做": research may use every
  USDT perpetual (the whole-market panel study). The mainnet limit below stays until a strategy passes and the owner
  approves the runtime change (dynamic universe; specs list at most 10 symbols today).
  - The panel study (`research/results/panel_2026-11-17/`, development window): one model pooled over the 100 most
    traded crypto perps of each day predicts a trend coming within 14 days (AUC 0.69) and 7-day volatility, not
    direction. Its models are frozen (`forward_spec.json`) and judged at 6 monthly looks, 2027-01-01 to 2027-06-01
    (decision 2027-06-01; `research/protocols/2026-11-17-panel-forward.md`); the look runner is still to be written.
    A model that holds is only a component: no strategy uses it yet.
  - Its first use, the volatility-expansion breakout study, failed 2026-11-18 (see Closed lines).
  - Daily BTC/ETH prediction report (owner 2026-11-18: "目前做改成做预测机器", "我负责做下单",
    "先做BTC和ETH，直接就在此对话推送"): `lab/daily_predict.py` runs the frozen models on the newest closed UTC day
    over a rolling public-data feed (`data/downloads/perps_feed.py`, `/root/lab_cache/prediction_feed/`) and posts
    trend probability, predicted volatility and a sizing rule (stop 2.5 predicted daily sd, a fixed risk share, at
    most 2x) in this conversation each morning. The owner places any order; the system sends none and reads no key. It predicts no
    direction. It reads data after 2026-11-13 for predictions only (no outcome, no tuning); the models are judged
    only at the registered looks. It refuses a day whose data is not whole, or if the frozen code or models changed.
- **Venue:** Binance USDT-margined perpetuals, **BTCUSDT and ETHUSDT only** again since 2026-11-17 (owner: "目前如何去改成做
  BTC和ETH，不再全市场这样做"; the whole-market scope of 2026-11-16 is reverted, in the lab too (`lab.LIVE_BTC_ETH`);
  delivery and USDC-margined contracts are not used). Mainnet refuses to start with a spec listing any other symbol (`MAINNET_SYMBOLS` in `trading/domain/strategy/spec.py`,
  checked at the start check and where the runtime is composed; `account_check --pilot` names it); the testnet and
  paper still take any USDT perpetual a signed spec lists, at most 10, with the strategy it names (R1–R7). The only
  spec so far is the BTC/ETH overlay. The Bybit, OKX and Polymarket code was deleted on 2026-11-15.
- **Research lab (tech lead, 2026-11-16):** a scheduled crew in this session (Asia/Shanghai): scout 20:54, judge
  22:54, engineer 02:51, morning brief 07:17 (push), weekly review Sunday. **Paused by the owner on 2026-11-17**: the
  five tasks are disabled, not deleted (the nightly runs re-mined one fixed snapshot: more trials, no new data); the
  monthly re-selection stays on. Do not re-enable them without the owner. It follows the owner's lab prompt
  as written: ideas are scanned and tested the same night and a champion and two challengers are named each
  morning, without pre-registering each batch (this replaces the research-protocol rule below for the lab;
  the four registered forward tests stay as they are). What stays: real data with source and timestamp only,
  never invented numbers, every estimate labelled; fees, funding, slippage and delisted contracts in every
  test; every configuration tested is recorded in `research/lab/trials.jsonl`; lab data ends 2026-11-12 (forward
  guard; snapshot `data_files/binance_usdt_perps_2026-11-12`), except the monthly re-selection (owner 2026-11-17:
  monthly, on a new snapshot that may read data after 2026-11-13, `lab/um_monthly.py`; the four registered forward
  tests stay frozen); the prompt's final check (unseen data, at least 50 trades, double fees, neighbouring settings, no
  lookahead, and the live terms: `LiveTerms` in `lab/um_lab.py`; since 2026-11-17 `LIVE_BTC_ETH` (owner: BTCUSDT and
  ETHUSDT only, 150 USDT split in half, positions up to 2x each half on isolated margin at the largest exchange leverage
  the stop allows; the whole-market runs used `LIVE`, 300 USDT over ten slots at 1x), the 8 % stop, a 4-hour pause
  after a stop that ends by itself) and "WHY THIS COULD FAIL" on every card; a lab champion is labelled unverified. No paper and no
  testnet stage (owner, 2026-11-16, see Path to live); real money waits for the owner's "go live"; the existing audited runtime
  and process are used (no AgenKit), and risk and execution stay deterministic code (the sentinel's stops:
  -3 % a day, the 15 % cumulative loss stop, a stale feed).
  - 2026-11-17 (`research/lab/runs/2026-11-17/`): the run on the 2020+ snapshot named E3 (above). Across the years
    what helps most is how a position is re-entered after a stop, not the stop's distance: waiting for the signal
    to reset beats the 4-hour pause for 8 of the top 10 ideas (`stop_stability.json`, `pause_then_reset_results.md`).
  - The whole-market rounds 2026-11-17b (`runs/2026-11-17b/plan.md`) and the volatility-tightened stop (`vol_stop.md`)
    were stopped unviewed when the owner moved to BTC/ETH (`terminated.md`, `vol_stop_terminated.md`); their replays
    stay registered in `trials.jsonl` and count in later deflated Sharpe probabilities. Their tools stay
    (`lab/um_portfolio.py`, the higher elimination line).
  - From 2026-11-17 the nightly crew and the monthly re-selection test BTCUSDT and ETHUSDT only, at `lab.LIVE_BTC_ETH`.
  - Monthly re-selection on new data (owner 2026-11-17): a scheduled task on the 8th of each month, 13:23
    Asia/Shanghai (`lab/um_monthly.py`, BTC/ETH since 2026-11-17); a champion's spec is drafted with `scripts.draft_strategy_spec` (always
    `passed: false`) and tracked 4 weeks before a person publishes it.
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
python -m data.downloads.us_equity_daily --out <new dir> --tradfi <tradfi_contracts.json> --nav SPY,GLD   # US underlyings, daily
python -m lab.paper_compare --paper-dir <paper dir> --data-dir <snapshot dir>   # paper vs research, fill by fill
python3 -m scripts.account_check --pilot --spec <signed spec>   # the pilot's start check: what is still missing
python3 -m scripts.live_fills --state-db <state.sqlite> --history <dir>   # pilot fills vs the backtest's assumptions
python3 -m scripts.live_fills --state-db <db> --history <dir> --events <events.jsonl> --json-out <f> --markdown-out <f>   # daily report
python -m lab.um_monthly plan|screen|final --snapshot <dir> --run research/lab/runs/<date>   # monthly re-selection
python -m lab.um_forward download|look --frozen <file> --study <name> --end <date> ...   # a champion's weekly look
python3 -m scripts.draft_strategy_spec --run <run dir> --snapshot <dir>   # a champion's spec draft (passed false)
python -m data.downloads.perps_feed update --feed /root/lab_cache/prediction_feed/feed   # public data, no key
python -m lab.daily_predict --feed /root/lab_cache/prediction_feed/feed --log /root/lab_cache/prediction_feed/predictions.jsonl   # the daily BTC/ETH report (updates the feed first)
python -m trading.infrastructure.state_dir --state-dir <dir> init     # once, before execution mode (RISK_SCOPE set)
python -m trading.infrastructure.state_dir --state-dir <dir> restore --snapshot <file> --reason "..."  # lost DB only
python -m trading.infrastructure.state_backup --state-dir <dir> --backup-dir <dir>   # what state-backup@ runs hourly
sudo bash infra/install_host.sh                    # host install/update: packages, dirs, units; starts nothing
python -m trading.infrastructure.kill --state-dir <dir> activate --reason "..." [--flatten]   # or clear / status
python -m trading.infrastructure.entry_pause --state-dir <dir> status            # paused symbols, ends_at if scheduled
python -m trading.infrastructure.entry_pause --state-dir <dir> release --symbol BTCUSDT --reason "..."  # a person only
python -m trading.infrastructure.manual --state-dir <dir> orders     # open orders and the venue's view
python -m trading.infrastructure.manual --state-dir <dir> settle --intent <id> --reason "..."     # runtime stopped
python -m trading.infrastructure.manual --state-dir <dir> position --symbol BTCUSDT --reason "..."  # ledger := venue
```

- **No native code:** the Rust extension `_quant_hotpath` was deleted on 2026-11-15 (its last version is in the
  tag `archive/ml-line-2026-11-15` and in git history); the repository is pure Python.

## Architecture (as actually wired)

**Runtimes:**
- `runner/binance_target.py`: entry point of the target runtime, a thin shell (S10) over
  `trading/infrastructure/cli/target.py` (start-up checks, composition of both modes, loop). It supplies what
  `trading` may not import: the signed spec loader and the TCA log writer. It loads no `lab` module.
  - The loop is `trading/application/runtime.py`; it wires `market_poller.py`, `exit_supervisor.py` and `status.py`
    (S3c). In execution mode the three sides are isolated: one that raises blocks every entry until it works again
    and the others still run; with a per-symbol strategy one symbol's failing bar blocks only that symbol. One
    symbol's hourly step is `trading/application/session.py`; `trading/infrastructure/target.py` wires the sessions
    with the spec's strategy. The one architecture description is `docs/arch/target.md`.
  - Each closed 1h bar: target from the signed spec's strategy → `trading/domain/rebalance.plan` → orders.
    - One strategy interface for research and the runtime (`trading/domain/strategy/base.py`: `Strategy`, `Window`,
      `PerSymbol`, `references`, `cross_sectional`): the BTC/ETH overlay (`overlay.py`) and the lab's 29 ideas
      (`ideas.py`). The spec (`spec.py`, schema 1 = the overlay, schema 2 = a named strategy) is parsed strictly:
      at most 10 symbols, unknown fields refused; negative targets trade as 0 unless it sets `allow_short`.
    - A per-symbol strategy decides each symbol as its bar arrives. A cross-sectional one, or one that reads other
      symbols (`references`), decides every symbol at one close with one call (`market_poller.py`, R3/R7d): while a
      symbol's bar is missing nobody decides, and a symbol still missing past the late-entry limit is blocked.
    - Each symbol waits for and checks funding at its own settlement interval, read from the exchange (R4);
      unknown, it takes no entries.
    - The exit side reads the whole account's positions once per reconciliation (`exit_supervisor.reconcile`,
      R5); a symbol missing from that snapshot, or an unreadable snapshot, reads its own.
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
  on 2026-11-15 (M4). systemd units: `binance-target` (paper), `binance-target-testnet`
  (execution, restarts at most 3 times an hour, network fixed to the testnet) and `binance-target-mainnet`
  (`Restart=no`, never at boot), `quant-alert@` (the failure alert), `state-backup@<network>.timer` (hourly),
  `health-watchdog`, `log-archive` (S8). One execution network per host; state under `/var/lib/quant` (2026-11-16).

**Binance execution:**
- `OrderLifecycle` (`trading/application/execution/lifecycle.py`) keeps durable SQLite intents and never resends;
  it queries instead. `ConfirmedExecutor` and `BinanceExecutionAdapter`, the old interfaces over it, are test
  support now (`tests/support/`): some characterization tests still drive the lifecycle through them.
- Only exchange trade records become fills.
- A placement is `rejected` only when Binance refused the request itself: a 4xx (not 403/408/418/429) whose error code
  is in `_REQUEST_REFUSED` (`adapter.py`, #157). Anything else - timeouts, 5xx, 408/418/429, -1000/-1001/-1006/-1007,
  a missing or unlisted code - is `error`: the outcome is unknown, the order is looked up for the whole not-found
  grace (90 s) with the symbol's entries locked, and a stop is not replaced meanwhile. A stop refused with -2021 or
  -4142 is breached (#158).
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
  `state_dir init` creates it and its tables (`state_dir restore` puts a checked snapshot back when it is lost or
  broken, and tightens it: drawdown latched - the next start flattens and stays flat until the week ends -,
  today's count used up, every symbol paused); the stores open it read-write and never create anything.
- Alerts and heartbeat (ADR-0016 points 1-3, S7): execution (testnet and mainnet) refuses to start without an
  alert channel and an https `HEARTBEAT_URL` (a secret), and starts only after the channel confirmed "execution
  starting", before any account request. The channel is `ALERT_FILE` (owner 2026-11-16: no Telegram; one JSON line
  per alert in `/var/lib/quant/alerts/`, confirmed when the line is written; a Claude Code session reads it about
  once an hour and may only reduce risk on its own: kill switch, stop the service) or the Telegram pair
  `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`; both together are refused. The heartbeat pings at most once a minute while
  waiting for market data and between cycles; it never stops the loop. Paper logs alerts only.
- Host lock (ADR-0016 point 5, S6): one execution process per account identity (sha256(network:API key)) on this
  host, `/var/lib/quant/locks/<identity>.lock`, whatever the state directory or `RISK_SCOPE`. The runtime and the
  manual tools take it; `state_dir init` creates the directory. Another host cannot be stopped from here.
- Kill file: the target runtime reads `<state dir>/kill_switch.json` (`trading/adapters/files/kill_switch.py`);
  in an initialized state directory a missing file means active. The runtime never clears it.
- Strategy loss stop (owner 2026-11-16, `trading/application/risk/loss_stop.py`; audit fixes of the same day): with
  a strategy capital, once realized PnL after fees + funding + unrealized PnL is 15 % of the capital below zero
  (`STRATEGY_LOSS_STOP_PCT`, a person can only tighten it), the runtime sets the kill file with flatten: the one
  write it makes there.
  - Unrealized PnL is the strategy's own: each `target.*` owner's ledger quantity x (the venue's mark for the symbol
    - the ledger's entry price). A manual position on a strategy symbol neither trips the stop nor hides a loss; a
    held ledger position without a venue row or a usable mark is unknown. Every `target.*` owner of the scope counts.
  - It is read at the start of every reconciliation, at most once a minute: in a cycle, in the warm-up and while a
    restart waits for market data. The flatten happens in the cycle of the trip.
  - Once reached it stays reached while the process lives: every reconciliation sets a cleared switch again and
    entries stay refused meanwhile (issue kind `strategy_loss_stop`: block every entry, cancel resting entries; a
    lasting one is alerted). Clearing alone trips it again, also after a restart (the warm-up reads the loss before
    any entry); going on means a new pilot in a new state directory.
  - A loss that cannot be known closes nothing and blocks every entry. Funding that cannot be refreshed does not
    blind the stop: it counts the last total read in this process, whatever its age (the sizing base keeps the
    2-hour rule). Since 2026-11-17 that total is the one kept in the state database, also right after a restart;
    it is unknown when the kept rows do not reach back to a first fill or when a symbol's next read would start more
    than 88 days back (24 hours before its kept rows end, or at its first fill when none are kept): the rest can never
    be read, so the total would be partial.
- `DrawdownGuard`: daily and weekly latch that resets at UTC period boundaries.
- `AccountExposureGate`: the account exposure cap, checked at entry time only: `BINANCE_MAX_LEVERAGE` (3 unless a
  person sets it, `DEFAULT_ACCOUNT_LEVERAGE`), never above `MAX_TOTAL_LEVERAGE`.
- Leverage ceiling (owner 2026-11-17): `MAX_TOTAL_LEVERAGE` (`trading/domain/risk/exposure.py`) is 10, it was 3. The
  owner asked for positions of 10x the capital and chose "放宽到 10 倍" after the 2-10x study
  (`research/results/overlay_pilot_leverage_2026-11-17/`: every 10x setting lost 74-95 % on the development window,
  2x stayed best). Only the ceiling of `STRATEGY_LEVERAGE` and `BINANCE_MAX_LEVERAGE` moved; both still default to
  1 and 3, a person writes them, and `STRATEGY_LEVERAGE` stays at most the exchange leverage setting (ceiling 10).
  The loss stop's ceiling (15 %) and the drawdown ceilings (10 % / 25 %) did not change (owner, same day).
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
  (paper records no bars); research is rebuilt from the strategy the paper record names (R7d). Exit 0 "一致",
  1 "不一致" (first differing fill), 2 "无法比较" (data missing, or not what paper saw). A difference is an
  intended divergence only through a fact of the record plus its `paper_divergence` event. Prints fills only,
  no returns.
- `lab/um_judge.py`: the research lab's judge as a tested tool (periods, the yearly symbol lists, a configuration =
  an idea's parameters plus a stop variant, walk-forward choice, score, final check; the holdout period is replayed
  only in `final`, for survivors); it replays strategies through the interface, the registered ideas from
  `lab.um_lab.lab_ideas()` (R7c). `lab.simulate` takes `reenter_after_stop="signal_reset"` (after a stop, no entry
  until the target has been 0). How a run uses it: `research/lab/README.md`.
- Every study writes its gate before results exist. Protocols live in `research/protocols/`.

**Open exit-side gaps — check before or in the pilot (the testnet stage was cancelled 2026-11-16):**
- Unverified: whether Binance accepts a second `closePosition` stop in the same direction. ADR-0012 places the new
  stop before it cancels the old one; if Binance refuses (the old order API had -4130), a new revision's stop is
  never placed and entries stay blocked while the old stop protects. To be checked in the pilot.
- A failed stop placement blocks entries (step 12) and is sent to the alert channel once it has lasted 2 minutes.
- At start-up a warm-up error other than a market-data failure (incomplete or non-contiguous history) still ends
  the process: a person looks before it runs. Once running, failures are isolated (S3c and its audit).

## Rules for changes

- **No automatic risk escalation or restarts:**
  - Never add an automatic path that raises a risk limit, removes a cap, restarts a live service or
    swaps a model into a running process.
  - Limits change only when a person edits config.
  - Two exceptions, both the owner's choices of 2026-11-16: the strategy's sizing base grows with its own realized
    PnL and funding income (`CappedEquity`, `StrategyFunding`), never above the account's equity and under the
    account exposure cap; and a stop's entry pause
    (triggered or breached stop) ends by itself after 4 hours (`STOP_PAUSE_SECONDS`, the same in the lab's
    `LiveTerms`). A person can end it earlier; a restore's pause still waits for a person. A spec with
    `"after_stop": "pause_then_reset"` (owner 2026-11-17, only ever stricter) also keeps a flat symbol's entries
    blocked after that pause until its strategy has decided a target of 0 since the stop (the lab's
    `reenter_after_stop="pause_then_reset"`); a 0 seen before a restart is not remembered, so the symbol waits for the
    next one.
- **Signed artifacts:** model artifacts (pickle and JSON, including `config.json` and strategy specs)
  load only through `infra/model_signing.py`.
- **Fail closed:** unreadable pause, kill or drawdown state blocks new exposure.
- **Zero sizing:** unknown equity or price sizes to zero, never to a minimum lot.
- **Research protocol:**
  - Pre-register the spec, gate and windows in a commit before running a study on real data (except the research
    lab of 2026-11-16, see Current status).
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

- **Research shares the host with the runtime:** run every backtest, screen or mutation run through
  `scripts/dev/lab_nice.sh` (nice 19, idle disk); the execution units run at `Nice=-5`, `CPUWeight=1000` (2026-11-17).
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
