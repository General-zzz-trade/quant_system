"""Binance BTC/ETH target-position runtime (1h clock, 4h/daily features): start-up checks, the composition of either
mode and the loop. ``runner/binance_target.py`` forwards here (the thin entry of ADR-0017, done in S10 part 3) and
supplies the two parts ``trading`` may not import: the signed spec loader and the transaction-cost log writer.

    python -m runner.binance_target --spec models_binance/strategy/final_spec.json            # paper
    python -m runner.binance_target --spec ... --enable-execution                             # testnet

Each closed 1h bar, per symbol:
1. advance the window (paper: the simulated exchange matches the bar and the exit side consumes its fills
   first, as research does),
2. compute the target weight with the function research uses on a trailing window (exactly the research value),
3. rebalance with ``trading.domain.rebalance.plan`` (band, reduce-only vs entry),
4. send intents through the gates (inputs fresh, kill switch, drawdown, 3x gross exposure, order limits) to the
   durable executor.

Paper runs the same execution path on the simulated exchange (ADR-0013 plan B) and reproduces the research
simulation's fills exactly (tested). Mainnet execution requires the frozen spec's research gate to have passed.
"""
from __future__ import annotations

import argparse
import logging
import signal
import time
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from trading.adapters.binance.rest import BinanceNonRetryableError, BinanceRestError
from trading.adapters.files.event_log import JsonlEventLog
from trading.adapters.files.kill_switch import KillFile
from trading.adapters.files.host_lock import HostLock
from trading.adapters.notify.heartbeat import HttpHeartbeat
from trading.adapters.files.lease import ExecutionLease
from trading.adapters.files.state_files import JsonPaperStore, JsonStatusFile
from trading.adapters.simulated.exchange import SimulatedExchange
from trading.adapters.sqlite.paper_state import SqlitePaperState
from trading.adapters.sqlite.schema import STATE_DB
from trading.application import runtime as _runtime
from trading.application.alerts import NotifyingEventLog
from trading.application.equity import EquitySource
from trading.application.event_guard import GuardedEventLog
from trading.application.ports import Clock, EventLog, KillSwitch, Notifier
from trading.application.runtime import POLL_SECONDS
from trading.domain.events import EventKind
from trading.domain.risk.liquidation import check_stop_inside_liquidation
from trading.domain.strategy.overlay import StrategySpec
from trading.domain.strategy.base import Strategy
from trading.domain.strategy.spec import parse_spec
from trading.infrastructure import target
from trading.infrastructure.bootstrap import build_adapters
from trading.infrastructure.config import (RuntimeConfig, Secrets, load_config, make_heartbeat, make_notifier,
                                           require_operator_channels, require_strategy_capital, risk_limits,
                                           with_credentials)
from trading.infrastructure.modes import PAPER, policy_for
from trading.infrastructure.paper import (PAPER_STATE, PAPER_STATE_DIR, BarTimeClock, DivergencePauses,
                                          PaperDivergenceLog, PaperStateStore, SymbolVenueEquity, paper_exchange,
                                          paper_state_db, refuse_execution_directory)
from trading.infrastructure.state_dir import is_managed, kill_file_path
from trading.infrastructure.state_dir import verify as verify_state_dir

logger = logging.getLogger("binance_target")
START_ALERT_SECONDS = 15.0          # how long the start waits for the alert channel to confirm (ADR-0016 point 2)

EXECUTION_STATE_DIR = Path("data/runtime/binance_target")    # as in the state_dir command's examples
SpecLoader = Callable[[Path], dict[str, Any]]
TcaLogFactory = Callable[[Path], EventLog]


def preflight(spec_path: Path, *, execution: bool, testnet: bool, stop_pct: D, state_dir: Path | None = None,
              config: RuntimeConfig | None = None, load_spec: SpecLoader) -> dict[str, Any]:
    """Every start-up check that needs no exchange: the spec's signature and kind, the research gate for
    mainnet, the risk scope, the drawdown ceiling, the stop distance and - for execution - a state directory a
    person initialized for this scope (step 13c). Run before any account is connected (invariant L6). Since
    ADR-0010 connecting is read-only; the account itself is checked next, by ``refuse_foreign_holdings``."""
    frozen = load_spec(spec_path)
    if execution:
        net = config.testnet if config is not None else testnet
        if not net and frozen.get("passed") is not True:
            raise RuntimeError("research gate not passed; mainnet execution refused")
        if config is None:      # the scope and the drawdown ceiling (step 13b) are checked by load_config
            config, _ = load_config(execution=True, testnet=testnet, stop_pct=stop_pct)
        check_stop_inside_liquidation(float(config.stop_pct), config.leverage or 0, config.margin)
        if state_dir is not None:
            verify_state_dir(state_dir, config.scope)
    elif state_dir is not None:
        refuse_execution_directory(state_dir)      # before anything is created in it (S10 audit H1)
    return frozen


class TargetRuntime(_runtime.TargetRuntime):
    """The runtime of either mode, composed: it loads the signed spec (through ``load_spec``) and picks the mode's
    policy and parts - in execution the connected account, in paper the simulated exchange (S10 part 3). The loop
    is trading.application.runtime.TargetRuntime; the sessions are wired in trading.infrastructure.target."""

    def __init__(self, *, spec_path: Path, state_dir: Path, mode: str, market: Any, account: Any = None,
                 clock: Clock = time.time, testnet: bool = True, stop_pct: D = D("0.08"),
                 reset_paper: bool = False, notifier: Notifier | None = None, config: RuntimeConfig | None = None,
                 secrets: Secrets | None = None, load_spec: SpecLoader,
                 tca_log: TcaLogFactory | None = None) -> None:
        policy = policy_for(mode)
        if config is None:      # read once, here, when the entry point did not (tests construct the runtime)
            config, secrets = load_config(execution=mode == "execution", testnet=testnet, stop_pct=stop_pct)
        self.config = config
        self.frozen = load_spec(spec_path)
        # R2: one record for either spec schema; the sessions get the spec's strategy and execution settings.
        self.target_spec = parse_spec(self.frozen)
        spec, strategy = self.target_spec.execution, self.target_spec.strategy()
        if mode == "execution":
            target.refuse_unverified_mainnet(testnet=config.testnet, spec_passed=self.frozen.get("passed"))
        self.account, self.clock, self._tca_log = account, clock, tca_log
        alert_prefix = ""
        if mode == "execution":                   # alerts reach a person in execution mode (step 12)
            alert_prefix = config.scope
            notifier = notifier if notifier is not None else make_notifier(secrets or Secrets(None, None))
        else:
            notifier = None
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # A failing event log never stops trading; it blocks every entry until it works again (S3 audit, finding 3).
        events = GuardedEventLog(JsonlEventLog(self.state_dir / "target_events.jsonl"))
        journal: Any = events
        if notifier is not None:
            journal = NotifyingEventLog(journal, notifier, prefix=alert_prefix)
        self.equity = EquitySource(clock)
        symbols = list(self.target_spec.symbols)
        account_check = venue = gate_clock = paper_store = None
        if mode == "execution":
            wiring, kill = self._execution(spec, symbols, journal, strategy)
            assert config.leverage is not None
            account_check = target.account_checker(account, state_dir=self.state_dir, scope=config.scope,
                                                   symbols=symbols, leverage=config.leverage, margin=config.margin)
        else:
            journal = PaperDivergenceLog(journal)
            venue, gate_clock = paper_exchange(spec, symbols), BarTimeClock()
            paper_store, wiring, kill = self._paper(spec, symbols, journal, venue, gate_clock, reset_paper, strategy)
            account = venue
        self.exchange = venue
        super().__init__(
            spec=spec, policy=policy, market=market, sessions=wiring.sessions, journal=journal, equity=self.equity,
            clock=clock, status_sink=JsonStatusFile(self.state_dir / "target_status.json"),
            market_errors=(BinanceRestError, OSError), final_market_errors=(BinanceNonRetryableError,),
            account=account, kill=kill, drawdown=wiring.drawdown, paper_store=paper_store,
            spec_passed=self.frozen.get("passed"), pauses=wiring.pauses, notifier=notifier,
            loss_stop=wiring.loss_stop,
            alert_prefix=alert_prefix, event_log_failure=events.failure, config_digest=config.digest(),
            account_check=account_check, venue=venue, gate_clock=gate_clock,
            exit_observer=journal.risk_exit if venue is not None else None)
        if paper_store is not None:      # from now on every journal transaction carries the paper state
            paper_store.state.follow(self.paper_revision, self.paper_document)
        # ADR-0017: what this run uses, recorded once, without any secret.
        journal.write(EventKind.CONFIG, **config.record(), digest=config.digest())
        logger.info("Configuration %s: %s", config.digest()[:12], config.record())

    def _paper(self, spec: StrategySpec, symbols: list[str], journal: PaperDivergenceLog, venue: SimulatedExchange,
               gate_clock: BarTimeClock, reset: bool,
               strategy: Strategy) -> tuple[PaperStateStore, target.ExecutionWiring, KillSwitch]:
        """Plan B (ADR-0013): the execution path over the simulated exchange. No credentialed adapter, no signed
        request, no account setting from the environment: the code's default limits (S10 decision 3), the venue's
        unlevered account for the stop check, a kill file that is not managed unless the directory is. The paper
        state is saved with the order journal, in one database (S10 decision 4)."""
        path, aside = paper_state_db(self.state_dir, reset=reset, stamp_ms=int(self.clock() * 1000))
        copy = JsonPaperStore(self.state_dir / PAPER_STATE, clock=self.clock)
        archived = copy.archive() if reset else None
        if archived or aside:
            journal.write(EventKind.PAPER_RESET, archived=archived, state_db=aside)
        store = PaperStateStore(SqlitePaperState(path, clock=self.clock), copy)
        kill = KillFile(kill_file_path(self.state_dir), missing_is_active=is_managed(self.state_dir))
        inputs = target.ExecutionInputs(account=venue, scope=self.config.scope, stop_pct=self.config.stop_pct,
                                        limits=replace(risk_limits({}), max_order_notional=None), kill=kill,
                                        leverage=1, margin=None)     # no single-order notional cap (2026-11-16)
        parts = target.PaperParts(symbol_equity=lambda symbol: SymbolVenueEquity(venue, symbol),
                                  gate_clock=gate_clock, sleeper=lambda seconds: None,
                                  pauses=lambda pauses: DivergencePauses(pauses, journal), state=store.state)
        wiring = target.execution_sessions(spec, symbols, state_dir=self.state_dir, journal=journal,
                                           equity=self.equity, inputs=inputs, clock=self.clock, policy=PAPER,
                                           paper=parts, strategy=strategy,
                                           allow_short=self.target_spec.allow_short)
        for symbol, session in wiring.sessions.items():
            session.sleeve = SimpleNamespace(broker=venue.broker(symbol))    # read-only, for people and tests
        return store, wiring, kill

    def _execution(self, spec: StrategySpec, symbols: list[str], journal: EventLog,
                   strategy: Strategy) -> tuple[target.ExecutionWiring, KillSwitch]:
        config = self.config
        assert config.limits is not None and config.leverage is not None
        # In a managed directory a missing kill file reads as active (ADR-0005 5b). The runtime never clears it; the
        # strategy loss stop is the one thing that sets it (owner 2026-11-16).
        kill = KillFile(kill_file_path(self.state_dir), missing_is_active=is_managed(self.state_dir))
        tca = self._tca_log(self.state_dir / "tca_binance.jsonl") if self._tca_log is not None else None
        account: Any = self.account          # the connected Binance adapter (build_adapters)
        inputs = target.ExecutionInputs(account=account, scope=config.scope, stop_pct=config.stop_pct,
                                        limits=config.limits, kill=kill, leverage=config.leverage, margin=config.margin,
                                        strategy_capital=config.strategy_capital, funding=account,
                                        strategy_loss_stop=config.strategy_loss_stop,
                                        tca_log=tca)
        return target.execution_sessions(spec, symbols, state_dir=self.state_dir, journal=journal,
                                         equity=self.equity, inputs=inputs, clock=self.clock,
                                         strategy=strategy, allow_short=self.target_spec.allow_short), kill


def main(argv: list[str] | None = None, *, load_spec: SpecLoader,
         TargetRuntime: Callable[..., _runtime.TargetRuntime], description: str | None = __doc__) -> None:
    """The entry point. ``load_spec`` is the signed spec loader and ``TargetRuntime`` builds the runtime (the entry
    module's subclass, which supplies the loader and the TCA log); the spec path goes only to those two."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=None,
                        help=f"default {PAPER_STATE_DIR} for paper, {EXECUTION_STATE_DIR} for execution")
    parser.add_argument("--stop-pct", type=D, default=D("0.08"), help="Exchange-side catastrophic stop distance")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="execution", action="store_false", help="Paper trading (default)")
    mode.add_argument("--enable-execution", dest="execution", action="store_true")
    parser.set_defaults(execution=False)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--reset-paper-state", action="store_true",
                        help="Archive the saved paper state and start a new paper record")
    args = parser.parse_args(argv)
    # Paper and execution never share a directory by default (S10 audit H1); paper refuses an execution one.
    args.state_dir = args.state_dir or (EXECUTION_STATE_DIR if args.execution else PAPER_STATE_DIR)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # ADR-0017: the configuration is read here, once; everything after receives it.
    config, secrets = load_config(execution=args.execution, stop_pct=args.stop_pct)
    testnet = config.testnet
    frozen = preflight(args.spec, execution=args.execution, testnet=testnet, stop_pct=config.stop_pct,
                       state_dir=args.state_dir, config=config, load_spec=load_spec)
    # The same validated, stripped scope the journal and the manual tools use (audit after step 12b, finding 7).
    scope = config.scope
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lease = ExecutionLease(args.state_dir / STATE_DB, scope)
    host: HostLock | None = None
    notifier = None
    try:
        if args.execution:
            # Owner 2026-11-16: mainnet sizes from the owner's strategy capital; it refuses to start without one.
            require_strategy_capital(config)
            # ADR-0016 points 1 and 3: alerts and a heartbeat, checked before anything reaches the account.
            require_operator_channels(config, secrets)
            config, secrets = with_credentials(config, secrets)   # after every local check (as before ADR-0017)
            assert config.identity is not None
            # ADR-0016 point 5: one execution process per account identity on this host, whatever its directory.
            host = HostLock(config.identity, holder=f"binance_target --state-dir {args.state_dir}")
            notifier = make_notifier(secrets)
            network = "testnet" if testnet else "MAINNET"
            # ADR-0016 point 2: confirmed by the alert channel (Telegram's answer, or the line written to the
            # alert file), or no account request at all.
            if not notifier.deliver(f"{network} execution starting",
                                    f"scope {scope}, account identity {config.identity[:12]}, state directory "
                                    f"{args.state_dir}", timeout_s=START_ALERT_SECONDS):
                raise RuntimeError(f"the start-up alert was not confirmed by the alert channel "
                                   f"({notifier.channel}); execution not started")
        symbols = list(parse_spec(frozen).symbols)
        market, account = build_adapters(execution=args.execution, testnet=testnet, symbols=symbols,
                                         credentials=secrets.binance)
        if args.execution:   # ADR-0010: read-only, before the runtime is built; the runtime never sets leverage
            assert config.leverage is not None and account is not None and notifier is not None
            try:
                target.refuse_foreign_holdings(account, state_dir=args.state_dir, scope=scope,
                                               symbols=symbols, leverage=config.leverage,
                                               margin=config.margin)
            except RuntimeError as exc:                       # best effort: the refusal still stands
                notifier.deliver("execution start refused", str(exc)[:3000], timeout_s=START_ALERT_SECONDS)
                raise
        built = TargetRuntime(spec_path=args.spec, state_dir=args.state_dir,
                              mode="execution" if args.execution else "paper", market=market, account=account,
                              testnet=testnet, stop_pct=config.stop_pct, reset_paper=args.reset_paper_state,
                              config=config, secrets=secrets, **({"notifier": notifier} if notifier else {}))
        _run(built, max_cycles=args.max_cycles, heartbeat=make_heartbeat(config, secrets))
    finally:                 # a start-up that fails half way releases them too
        lease.close()
        if host is not None:
            host.close()


def _beat(heartbeat: HttpHeartbeat | None) -> None:
    """The heartbeat is a bystander (ADR-0016 point 3): nothing it does may stop the loop."""
    if heartbeat is None:
        return
    try:
        heartbeat.beat()
    except Exception as exc:
        logger.warning("heartbeat failed (%s)", type(exc).__name__)


def _run(built: _runtime.TargetRuntime, *, max_cycles: int | None, heartbeat: HttpHeartbeat | None = None) -> None:
    """The loop: warm up (market data down: exits keep working), then one cycle every POLL_SECONDS. The heartbeat
    beats on both, so waiting for market data never reads as a dead process (design audit L6)."""
    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    while not stop["flag"]:
        _beat(heartbeat)
        if built.warmup_or_hold():
            break
        time.sleep(POLL_SECONDS)
    cycles = 0
    while not stop["flag"]:
        _beat(heartbeat)
        built.cycle()
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        time.sleep(POLL_SECONDS)


__all__ = ["EXECUTION_STATE_DIR", "SpecLoader", "TargetRuntime", "TcaLogFactory", "main", "preflight"]
