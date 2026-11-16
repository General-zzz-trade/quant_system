"""The strategy interface shared by research and the runtime (plan R1, docs/arch/runtime-lab-strategies-plan.md,
owner 2026-11-16).

A strategy turns the closed hourly bars and funding settlements of its symbols (one ``Window`` per symbol) into a
target weight - a signed fraction of the symbol's sizing equity - and a ready flag at every close:

- ``history(windows)``: every close, per symbol, as a frame of ``available_at``, ``target``, ``ready`` (research
  replays it);
- ``latest(windows)``: the last close, per symbol (the runtime acts on it). It equals history's last row for the same
  windows: one function decides in research and in live trading.

A per-symbol strategy (``PerSymbol``, ``cross_sectional`` false) reads only its own symbol's window, and the runtime
decides each symbol as its bar arrives. A cross-sectional one may read all of them: from R3 on the runtime gives it
every symbol's window at the same close in one call, and decides nothing while a symbol's bar is missing.
Strategies are code, reviewed and tested; a signed spec names one from ``STRATEGIES`` with its parameters. Every
input known at a close is in the windows; nothing later is (tests/unit/lab/test_lookahead.py checks the functions).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

import pandas as pd

from trading.domain.strategy.ideas import IDEAS as LAB_IDEAS, IdeaStrategy
from trading.domain.strategy.overlay import StrategySpec, compute_targets


@dataclass(frozen=True)
class Window:
    """One symbol's inputs at a close: its closed hourly bars (``available_at``, ``open_time``, OHLC, ``volume`` and,
    from R7a, the kline fields ``quote_volume``, ``trades``, ``taker_buy_volume``, ``taker_buy_quote_volume``: NaN
    when unknown) and its funding settlements (settlement time in ms -> rate) known by then."""
    hourly: pd.DataFrame
    funding: pd.Series


class Strategy(Protocol):
    name: str
    cross_sectional: bool                     # a symbol's target may depend on the other symbols' windows

    def window_hours(self) -> int: ...        # closed bars a decision needs (a trailing window gives the full value)

    def funding_window(self) -> int: ...      # funding settlements it needs

    def references(self) -> tuple[str, ...]: ...    # symbols it reads besides each traded one (R7a)

    def history(self, windows: Mapping[str, Window]) -> dict[str, pd.DataFrame]: ...

    def latest(self, windows: Mapping[str, Window]) -> dict[str, tuple[float, bool]]: ...


class PerSymbol:
    """A strategy whose decision for a symbol reads only that symbol's window: ``symbol_history`` is the whole of it."""
    name = "per_symbol"
    cross_sectional = False

    def window_hours(self) -> int:
        raise NotImplementedError

    def funding_window(self) -> int:
        raise NotImplementedError

    def symbol_history(self, window: Window) -> pd.DataFrame:
        raise NotImplementedError

    def references(self) -> tuple[str, ...]:
        return ()

    def history(self, windows: Mapping[str, Window]) -> dict[str, pd.DataFrame]:
        return {symbol: self.symbol_history(window) for symbol, window in windows.items()}

    def latest(self, windows: Mapping[str, Window]) -> dict[str, tuple[float, bool]]:
        return {symbol: (float(frame["target"].iloc[-1]), bool(frame["ready"].iloc[-1]))
                for symbol, frame in self.history(windows).items()}


class OverlayStrategy(PerSymbol):
    """The BTC/ETH multi-timeframe overlay (trading/domain/strategy/overlay.py), the first implementation."""
    name = "overlay"

    def __init__(self, spec: StrategySpec) -> None:
        self.spec = spec

    def window_hours(self) -> int:
        return self.spec.window_hours()

    def funding_window(self) -> int:
        return self.spec.funding_window()

    def symbol_history(self, window: Window) -> pd.DataFrame:
        return compute_targets(window.hourly, window.funding, self.spec)


# The overlay, and from R7b the research lab's ideas under their lab names (trading/domain/strategy/ideas.py).
STRATEGIES: dict[str, Callable[[Any], Strategy]] = {"overlay": OverlayStrategy,
                                                    **{name: partial(IdeaStrategy, name) for name in LAB_IDEAS}}


def strategy_from(name: str, params: Any) -> Strategy:
    """The strategy a spec names, built from its parameters; an unknown name is refused."""
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; known: {sorted(STRATEGIES)}")
    return STRATEGIES[name](params)


__all__ = ["OverlayStrategy", "PerSymbol", "STRATEGIES", "Strategy", "Window", "strategy_from"]
