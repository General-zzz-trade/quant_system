"""The strategy spec, version 2 (plan R2, docs/arch/runtime-lab-strategies-plan.md, owner 2026-11-16).

``parse_spec(document)`` turns a signed spec document into one record, ``TargetSpec``: the strategy's name and
parameters, the execution settings, the symbols (at most ``MAX_SYMBOLS``) and whether shorts are allowed.

- Schema 1, kind ``btc_eth_multitimeframe_target`` (the BTC/ETH overlay, the pilot's spec): the overlay strategy with
  the document's ``spec`` as both its parameters and its execution settings; shorts are allowed exactly when the
  overlay's ``long_only`` is false.
- Schema 2, kind ``target_strategy``: ``strategy`` {``name``, ``params``}, ``execution`` {``band``, ``execution``,
  ``ttl_bars``}, ``symbols``, ``allow_short``.

The symbol list comes from the lab and is published by the owner (2026-11-16): the runtime trades only the symbols a
signed spec lists. Anything the parser cannot use is refused by name.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from trading.domain.strategy.base import STRATEGIES, Strategy
from trading.domain.strategy.ideas import IDEAS, idea_params
from trading.domain.strategy.overlay import StrategySpec

MAX_SYMBOLS = 10                    # owner 2026-11-16: at most 10 symbols in the first version
KIND_V1, KIND_V2 = "btc_eth_multitimeframe_target", "target_strategy"
# How each strategy's parameters become what its constructor takes.
PARAMS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "overlay": lambda params: StrategySpec.from_record(dict(params)),
    **{name: partial(idea_params, name) for name in IDEAS},          # R7b: the lab's ideas
}


@dataclass(frozen=True)
class TargetSpec:
    strategy_name: str
    params: Any                     # the strategy's parsed parameters
    execution: StrategySpec         # the execution settings the sessions use: band, execution mode, maker lifetime
    symbols: tuple[str, ...]
    allow_short: bool

    def strategy(self) -> Strategy:
        return STRATEGIES[self.strategy_name](self.params)

    def record(self) -> dict[str, Any]:
        params = self.params.record() if hasattr(self.params, "record") else self.params
        return {"strategy": self.strategy_name, "params": params, "symbols": list(self.symbols),
                "allow_short": self.allow_short, "band": self.execution.band, "execution": self.execution.execution,
                "ttl_bars": self.execution.ttl_bars}


def _is_usdt_perpetual(symbol: Any) -> bool:
    """A USDT-margined perpetual's name: upper-case letters and digits ending in USDT; no delivery suffix."""
    return (isinstance(symbol, str) and symbol.isascii() and symbol.isalnum() and symbol == symbol.upper()
            and symbol.endswith("USDT") and 6 <= len(symbol) <= 24)


def _symbols(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError("the spec's symbols must be a list")
    if not raw:
        raise ValueError("the spec lists no symbol")
    if len(raw) > MAX_SYMBOLS:
        raise ValueError(f"the spec lists {len(raw)} symbols; at most {MAX_SYMBOLS}")
    for symbol in raw:
        if not _is_usdt_perpetual(symbol):
            raise ValueError(f"not a USDT-margined perpetual symbol: {symbol!r}")
    if len(set(raw)) != len(raw):
        raise ValueError("a symbol is listed twice")
    return tuple(raw)


def _params(name: Any, raw: Any) -> Any:
    if name not in STRATEGIES or name not in PARAMS:
        raise ValueError(f"unknown strategy {name!r}; known: {sorted(STRATEGIES)}")
    if not isinstance(raw, Mapping):
        raise ValueError(f"the strategy {name!r} needs its params")
    try:
        return PARAMS[name](raw)
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"the strategy {name!r} refuses its params: {exc}") from None


def parse_spec(document: Mapping[str, Any]) -> TargetSpec:
    schema, kind = document.get("schema"), document.get("kind")
    if schema == 1:
        if kind != KIND_V1:
            raise ValueError(f"a schema-1 spec must be of kind {KIND_V1!r}, not {kind!r}")
        overlay = _params("overlay", document.get("spec"))
        return TargetSpec("overlay", overlay, overlay, _symbols(document.get("symbols")), not overlay.long_only)
    if schema != 2:
        raise ValueError(f"unknown spec schema {schema!r}")
    if kind != KIND_V2:
        raise ValueError(f"a schema-2 spec must be of kind {KIND_V2!r}, not {kind!r}")
    strategy = document.get("strategy")
    if not isinstance(strategy, Mapping):
        raise ValueError("the spec names no strategy")
    params = _params(strategy.get("name"), strategy.get("params"))
    execution = document.get("execution")
    if not isinstance(execution, Mapping) or any(k not in execution for k in ("band", "execution", "ttl_bars")):
        raise ValueError("the spec's execution needs band, execution and ttl_bars")
    try:
        settings = StrategySpec(band=float(execution["band"]), execution=str(execution["execution"]),
                                ttl_bars=int(execution["ttl_bars"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the spec's execution settings are refused (band, execution, ttl_bars): {exc}") from None
    allow_short = document.get("allow_short")
    if not isinstance(allow_short, bool):
        raise ValueError("the spec's allow_short must be true or false")
    spec = TargetSpec(str(strategy["name"]), params, settings, _symbols(document.get("symbols")), allow_short)
    unlisted = [symbol for symbol in spec.strategy().references() if symbol not in spec.symbols]
    if unlisted:     # R7a: the runtime reads only the symbols it trades
        raise ValueError(f"the strategy {spec.strategy_name!r} reads {', '.join(unlisted)}, which the spec does not "
                         "list; list it among the symbols")
    return spec


__all__ = ["KIND_V1", "KIND_V2", "MAX_SYMBOLS", "PARAMS", "TargetSpec", "parse_spec"]
