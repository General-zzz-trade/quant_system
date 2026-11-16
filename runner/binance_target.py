"""Binance BTC/ETH target-position runtime: the command-line entry (a thin shell since S10 part 3, ADR-0017).

    python -m runner.binance_target --spec models_binance/strategy/final_spec.json            # paper
    python -m runner.binance_target --spec ... --enable-execution                             # testnet

The entry point, its start-up checks and the composition of both modes live in ``trading.infrastructure.cli.target``.
This module supplies the two parts ``trading`` may not import (.importlinter): the signed spec loader
(``infra.model_signing``; artifacts load only through it) and the transaction-cost log writer (``monitoring.tca``).
Its names keep their meaning for the tests and the systemd unit.
"""
from __future__ import annotations

from decimal import Decimal as D
from pathlib import Path
from typing import Any

from trading.adapters.simulated.exchange import BROKER_STATE as _BROKER_STATE
from trading.application.execution.fills import TargetFillConsumer
from trading.application.runtime import POLL_SECONDS
from trading.application.session import MAX_ENTRY_DELAY_MS, TargetSession
from trading.domain.events import EventKind
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure.cli import target as _cli
from trading.infrastructure.config import RuntimeConfig
from trading.infrastructure.paper import PAPER_STATE


def load_spec(path: Path) -> dict:
    from infra.model_signing import load_verified_json
    spec = load_verified_json(path)
    if spec.get("kind") != "btc_eth_multitimeframe_target" or spec.get("schema") != 1:
        raise ValueError("not a frozen BTC/ETH target-strategy spec")
    StrategySpec.from_record(spec["spec"])  # validates parameters
    return spec


def preflight(spec_path: Path, *, execution: bool, testnet: bool, stop_pct: D, state_dir: Path | None = None,
              config: RuntimeConfig | None = None) -> dict:
    """``trading.infrastructure.cli.target.preflight`` with the signed spec loader."""
    return _cli.preflight(spec_path, execution=execution, testnet=testnet, stop_pct=stop_pct, state_dir=state_dir,
                          config=config, load_spec=load_spec)


class _TcaLog:
    """The EventLog port over the TCA file next to the order journal (one append per fill). Moved here from the
    execution adapter shell (ADR-0006, 2026-11-15); the writer itself is still the legacy monitoring.tca."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, kind: EventKind, **fields: Any) -> None:
        from monitoring.tca import TCALogger
        tca = TCALogger("binance", path=self._path)
        try:
            tca.record_fill(**fields)
        finally:
            tca.close()


class TargetRuntime(_cli.TargetRuntime):
    """``trading.infrastructure.cli.target.TargetRuntime`` with the signed spec loader and the TCA log."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(load_spec=load_spec, tca_log=_TcaLog, **kwargs)


def main(argv: list[str] | None = None) -> None:
    _cli.main(argv, load_spec=load_spec, TargetRuntime=TargetRuntime)


__all__ = ["MAX_ENTRY_DELAY_MS", "PAPER_STATE", "POLL_SECONDS", "TargetFillConsumer", "TargetRuntime",
           "TargetSession", "_BROKER_STATE", "load_spec", "main", "preflight"]


if __name__ == "__main__":
    main()
