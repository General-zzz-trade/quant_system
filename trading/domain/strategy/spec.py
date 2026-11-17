"""The strategy spec, version 2 (plan R2, docs/arch/runtime-lab-strategies-plan.md, owner 2026-11-16).

``parse_spec(document)`` turns a signed spec document into one record, ``TargetSpec``: the strategy's name and
parameters, the execution settings, the symbols (at most ``MAX_SYMBOLS``) and whether shorts are allowed.

- Schema 1, kind ``btc_eth_multitimeframe_target`` (the BTC/ETH overlay, the pilot's spec): the overlay strategy with
  the document's ``spec`` as both its parameters and its execution settings; shorts are allowed exactly when the
  overlay's ``long_only`` is false.
- Schema 2, kind ``target_strategy``: ``strategy`` {``name``, ``params``}, ``execution`` {``band``, ``execution``,
  ``ttl_bars``}, ``symbols``, ``allow_short``, and optionally ``after_stop`` (tech lead 2026-11-17): ``"pause"``, the
  default and schema 1's rule, or ``"pause_then_reset"``, which after a stop's pause also waits, while flat, until the
  strategy's target has been 0 since the stop.

The symbol list comes from the lab and is published by the owner (2026-11-16): the runtime trades only the symbols a
signed spec lists. Anything the parser cannot use is refused by name.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

from trading.domain.issues import AFTER_STOP_PAUSE, AFTER_STOP_RULES
from trading.domain.strategy.base import STRATEGIES, Strategy
from trading.domain.strategy.ideas import IDEAS, idea_params
from trading.domain.strategy.overlay import StrategySpec

MAX_SYMBOLS = 10                    # owner 2026-11-16: at most 10 symbols in the first version
# Owner 2026-11-17 ("目前如何去改成做BTC和ETH，不再全市场这样做", "加硬限制"): mainnet trades these two only. The
# testnet and paper stay general (R1-R7) for tests; the start check refuses a mainnet spec listing any other symbol.
MAINNET_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT"})
KIND_V1, KIND_V2 = "btc_eth_multitimeframe_target", "target_strategy"
def _idea_params(name: str) -> Callable[[Mapping[str, Any]], Any]:
    return lambda params: idea_params(name, params)


# How each strategy's parameters become what its constructor takes.
PARAMS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "overlay": lambda params: StrategySpec.from_record(dict(params)),
    **{name: _idea_params(name) for name in IDEAS},                  # R7b: the lab's ideas
}


@dataclass(frozen=True)
class TargetSpec:
    strategy_name: str
    params: Any                     # the strategy's parsed parameters
    execution: StrategySpec         # the execution settings the sessions use: band, execution mode, maker lifetime
    symbols: tuple[str, ...]
    allow_short: bool
    after_stop: str = AFTER_STOP_PAUSE

    def strategy(self) -> Strategy:
        return STRATEGIES[self.strategy_name](self.params)

    def record(self) -> dict[str, Any]:
        params = self.params.record() if hasattr(self.params, "record") else self.params
        out = {"strategy": self.strategy_name, "params": params, "symbols": list(self.symbols),
               "allow_short": self.allow_short, "band": self.execution.band, "execution": self.execution.execution,
               "ttl_bars": self.execution.ttl_bars}
        return out if self.after_stop == AFTER_STOP_PAUSE else {**out, "after_stop": self.after_stop}

    def identity(self) -> dict[str, Any]:
        """What a paper record stores to know its strategy: the overlay as traded (its own record, as before R2, so a
        schema-1 record is unchanged; with ``allow_short`` when the spec's rule is not the one its ``long_only``
        implies), any other strategy by name, parameters, execution settings and shorts rule (R2 and R5/R6 audits);
        either with ``after_stop`` when it is not the default, so a record without it is unchanged."""
        record = self._identity()
        return record if self.after_stop == AFTER_STOP_PAUSE else {**record, "after_stop": self.after_stop}

    def _identity(self) -> dict[str, Any]:
        if self.strategy_name == "overlay":
            traded: StrategySpec = replace(self.params, band=self.execution.band, execution=self.execution.execution,
                                           ttl_bars=self.execution.ttl_bars)
            record = traded.record()
            if self.allow_short != (not traded.long_only):     # R5/R6 audit: the shorts rule, when the spec sets it
                record["allow_short"] = self.allow_short
            return record
        return {"strategy": self.strategy_name, "params": dict(self.params), "band": self.execution.band,
                "execution": self.execution.execution, "ttl_bars": self.execution.ttl_bars,
                "allow_short": self.allow_short}


def _is_usdt_perpetual(symbol: Any) -> bool:
    """A USDT-margined perpetual's name: upper-case letters and digits ending in USDT; no delivery suffix."""
    return (isinstance(symbol, str) and symbol.isascii() and symbol.isalnum() and symbol == symbol.upper()
            and symbol.endswith("USDT") and 5 <= len(symbol) <= 24)


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


V2_KEYS = frozenset({"schema", "kind", "strategy", "execution", "symbols", "allow_short", "passed", "after_stop"})
EXECUTION_KEYS = ("band", "execution", "ttl_bars")


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _strict_overlay(raw: Mapping[str, Any], settings: StrategySpec) -> None:
    """Schema 2 (R2 audit): every overlay parameter is one the overlay has, of its own type; its execution settings,
    when it carries them, are the spec's execution block."""
    defaults = {f.name: f.default for f in fields(StrategySpec)}
    for key, value in raw.items():
        if key not in defaults:
            raise ValueError(f"the overlay has no parameter {key!r}")
        default = defaults[key]
        if key == "lookbacks":
            ok = isinstance(value, (list, tuple)) and bool(value) and all(type(x) is int and x > 0 for x in value)
        elif isinstance(default, bool):
            ok = type(value) is bool
        elif isinstance(default, int):
            ok = type(value) is int
        elif isinstance(default, float):
            ok = _number(value)
        else:
            ok = type(value) is str
        if not ok:
            raise ValueError(f"the overlay parameter {key!r} has the wrong type: {value!r}")
        if key in EXECUTION_KEYS and value != getattr(settings, key):
            raise ValueError(f"the overlay parameter {key!r} differs from the spec's execution block")


def _execution(raw: Any) -> StrategySpec:
    if not isinstance(raw, Mapping) or set(raw) != set(EXECUTION_KEYS):
        raise ValueError("the spec's execution needs exactly band, execution and ttl_bars")
    band, mode, ttl = raw["band"], raw["execution"], raw["ttl_bars"]
    if not _number(band) or type(mode) is not str or type(ttl) is not int:
        raise ValueError("the spec's execution settings have the wrong types (band a number, execution a word, "
                         "ttl_bars a whole number)")
    try:
        return StrategySpec(band=float(band), execution=mode, ttl_bars=ttl)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the spec's execution settings are refused (band, execution, ttl_bars): {exc}") from None


def parse_spec(document: Mapping[str, Any]) -> TargetSpec:
    schema, kind = document.get("schema"), document.get("kind")
    if type(schema) is not int:
        raise ValueError(f"unknown spec schema {schema!r}")
    if schema == 1:
        if kind != KIND_V1:
            raise ValueError(f"a schema-1 spec must be of kind {KIND_V1!r}, not {kind!r}")
        if "after_stop" in document:          # audit of PR #142, finding 5: refused, not silently ignored
            raise ValueError("after_stop needs a schema-2 spec")
        overlay = _params("overlay", document.get("spec"))
        return TargetSpec("overlay", overlay, overlay, _symbols(document.get("symbols")), not overlay.long_only)
    if schema != 2:
        raise ValueError(f"unknown spec schema {schema!r}")
    if kind != KIND_V2:
        raise ValueError(f"a schema-2 spec must be of kind {KIND_V2!r}, not {kind!r}")
    unknown = sorted(set(document) - V2_KEYS)
    if unknown:                       # R2 audit: a field the runtime does not understand is refused, not ignored
        raise ValueError(f"the spec has fields the runtime does not understand: {', '.join(map(str, unknown))}")
    if not isinstance(document.get("passed"), bool):
        raise ValueError("the spec's passed must be true or false")
    strategy = document.get("strategy")
    if not isinstance(strategy, Mapping):
        raise ValueError("the spec names no strategy")
    if set(strategy) - {"name", "params"}:
        raise ValueError("the spec's strategy holds only name and params")
    settings = _execution(document.get("execution"))
    if strategy.get("name") == "overlay" and isinstance(strategy.get("params"), Mapping):
        _strict_overlay(strategy["params"], settings)
    params = _params(strategy.get("name"), strategy.get("params"))
    allow_short = document.get("allow_short")
    if not isinstance(allow_short, bool):
        raise ValueError("the spec's allow_short must be true or false")
    after_stop = document.get("after_stop", AFTER_STOP_PAUSE)
    if type(after_stop) is not str or after_stop not in AFTER_STOP_RULES:
        raise ValueError(f"the spec's after_stop must be one of {', '.join(AFTER_STOP_RULES)}, not {after_stop!r}")
    spec = TargetSpec(str(strategy["name"]), params, settings, _symbols(document.get("symbols")), allow_short,
                      after_stop)
    unlisted = [symbol for symbol in spec.strategy().references() if symbol not in spec.symbols]
    if unlisted:     # R7a: the runtime reads only the symbols it trades
        raise ValueError(f"the strategy {spec.strategy_name!r} reads {', '.join(unlisted)}, which the spec does not "
                         "list; list it among the symbols")
    return spec


__all__ = ["EXECUTION_KEYS", "KIND_V1", "KIND_V2", "MAINNET_SYMBOLS", "MAX_SYMBOLS", "PARAMS", "TargetSpec", "V2_KEYS",
           "parse_spec"]
