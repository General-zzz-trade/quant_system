"""Every feature, signal and target function research and the forward tests use passes the shared look-ahead
check (lab/causality.py; design 2026-11 M5, step S12).

Registration rule:
- every module under lab/ and trading/domain/strategy/, and trading/domain/market.py, is a feature module
  (FEATURE_MODULES) or is listed in OTHER_MODULES with the reason it computes no feature;
- every public function defined in a feature module is checked (CASES) or listed in NOT_CHECKED with a reason.
A new feature function, or a new module, fails the two registry tests until it is registered.

Data: the synthetic snapshot of test_pair_study (280 days of both symbols with funding, premium and metrics),
loaded by lab.pair.load_inputs exactly as a study loads the real one.
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab import breakout, pair
from lab import candidates as c
from lab import candidates_study as cs
from lab import overlay_risk_study as rs
from lab import pair_study as ps
from lab.causality import check, variants, violations
from lab.overlay_study import SPEC as OVERLAY
from tests.unit.lab.test_pair_study import write_snapshot
from trading.domain import market
from trading.domain.strategy import overlay as ov

ROOT = Path(__file__).resolve().parents[3]
HOUR = 3_600_000
DAY = 24 * HOUR
T0 = 1_609_459_200_000          # 2021-01-01
DAYS = 280
START = T0 + 220 * DAY          # first decision and first walk-forward block of the cases
END = T0 + DAYS * DAY
CUTS = (T0 + 230 * DAY, T0 + 245 * DAY + 13 * HOUR, T0 + 262 * DAY + 4 * HOUR)    # midnight, an odd hour, a 4h close
BTC, ETH = "BTCUSDT", "ETHUSDT"
ALL_LAYERS = replace(OVERLAY, lookbacks=(20, 60, 120), signal="scaled", long_only=False, funding_filter=True,
                     vol_spike_cut=True, confirm_4h=True)
RIDGE_PD_7D = next(x for x in ps.CANDIDATES if x["id"] == "ridge_PD_7d")
_rng = np.random.default_rng(5)
MODEL_B = c.fit_ridge(_rng.normal(size=(200, len(c.B_FEATURES))), _rng.normal(size=200))        # frozen models
MODEL_PAIR = c.fit_ridge(_rng.normal(size=(200, len(cs.PAIR_FEATURES))), _rng.normal(size=200))


# -- cases: market inputs -> a timed output, the way research and the forward replays call each function -------------
def _rates(m, s):
    return market.funding_series(m["funding"][s])


def _directional(m):
    """lab.candidates_forward.evaluate_directional up to the targets."""
    data = cs.b_dataset(m)
    return data, {s: pd.Series(c.predict_frozen(MODEL_B, data[s]["x"]), index=data[s]["feats"].index)
                  for s in pair.SYMBOLS}


def _weekly(m):
    feats = c.symbol_features(m, ETH)
    prediction = pd.Series(c.predict_frozen(MODEL_B, feats[list(c.B_FEATURES)].to_numpy(float)), index=feats.index)
    return c.weekly_targets(m["hourly"][ETH], prediction, feats["sigma"], start=START)


def _pair_sides(m):
    """lab.candidates_forward.evaluate_pair up to the decisions."""
    feats = pair.build(m)[1]
    prediction = pd.Series(c.predict_frozen(MODEL_PAIR, feats[list(cs.PAIR_FEATURES)].to_numpy(float)),
                           index=feats.index)
    return ps.decisions(prediction, 7, start=START, end=END)


def _pair_ml(m):
    closes, feats = pair.build(m)
    return ps.ml_predictions(feats, closes, RIDGE_PD_7D, oos_start=START, limit=END)[0]       # blocks' IC: labels


CASES = [
    ("market.frame_from_bars", market.frame_from_bars, lambda m: market.frame_from_bars(m["bars"][BTC])),
    ("market.aggregate.4h", market.aggregate, lambda m: market.aggregate(m["hourly"][BTC], 4)),
    ("market.aggregate.1d", market.aggregate, lambda m: market.aggregate(m["hourly"][BTC], 24)),
    ("market.group_closes", market.group_closes, lambda m: market.group_closes(m["hourly"][BTC], 24)),
    ("market.funding_series", market.funding_series, lambda m: _rates(m, BTC)),
    ("market.funding_window", market.funding_window,
     lambda m: market.funding_window({t: float(r) for t, (r, _) in m["funding"][BTC].items()}, None, 10 ** 6)),
    ("overlay.daily_weights", ov.daily_weights, lambda m: ov.daily_weights(m["hourly"][BTC], OVERLAY)),
    ("overlay.four_hour_features", ov.four_hour_features, lambda m: ov.four_hour_features(m["hourly"][BTC])),
    ("overlay.funding_bands", ov.funding_bands, lambda m: ov.funding_bands(_rates(m, BTC))),
    ("overlay.compute_targets.frozen", ov.compute_targets,
     lambda m: ov.compute_targets(m["hourly"][BTC], _rates(m, BTC), OVERLAY)),
    ("overlay.compute_targets.all_layers", ov.compute_targets,
     lambda m: ov.compute_targets(m["hourly"][ETH], _rates(m, ETH), ALL_LAYERS)),
    ("breakout.features", breakout.features, lambda m: breakout.features(m["hourly"][ETH], breakout.BreakoutSpec())),
    ("candidates.daily_close", c.daily_close, lambda m: c.daily_close(m["hourly"][ETH])),
    ("candidates.daily_vol", c.daily_vol, lambda m: c.daily_vol(c.daily_close(m["hourly"][ETH]))),
    ("candidates.symbol_features", c.symbol_features, lambda m: c.symbol_features(m, ETH)),
    ("candidates.predict_frozen", c.predict_frozen, lambda m: _directional(m)[1]),
    ("candidates.weekly_targets", c.weekly_targets, _weekly),
    ("candidates.flush_signal", c.flush_signal, lambda m: c.flush_signal(m["hourly"][BTC], m["metrics"][BTC])),
    ("candidates.flush_targets", c.flush_targets, lambda m: c.flush_targets(m["hourly"][ETH], m["metrics"][ETH])),
    ("pair.daily_closes", pair.daily_closes, lambda m: pair.daily_closes(m["hourly"])),
    ("pair.price_features", pair.price_features, lambda m: pair.price_features(pair.daily_closes(m["hourly"]))),
    ("pair.derivative_features", pair.derivative_features,
     lambda m: pair.derivative_features(m, pair.daily_closes(m["hourly"]).index.to_numpy(dtype=np.int64))),
    ("pair.build", pair.build, pair.build),
    ("pair_study.ml_predictions", ps.ml_predictions, _pair_ml),
    ("pair_study.rule_signal.mom", ps.rule_signal, lambda m: ps.rule_signal(pair.build(m)[1], "mom")),
    ("pair_study.rule_signal.rev", ps.rule_signal, lambda m: ps.rule_signal(pair.build(m)[1], "rev")),
    ("pair_study.decisions", ps.decisions, _pair_sides),
    ("pair_study.merged_hourly", ps.merged_hourly,
     lambda m: ps.merged_hourly(m["hourly"]).assign(available_at=lambda f: f["open_time"] + HOUR)),
    ("candidates_study.b_dataset", cs.b_dataset,                   # its labels (y, label_end) look ahead by design
     lambda m: {s: d["feats"] for s, d in cs.b_dataset(m).items()}),
    ("candidates_study.b_walk_forward", cs.b_walk_forward,
     lambda m: cs.b_walk_forward(cs.b_dataset(m), first_test=START, limit=END)[0]),
    ("candidates_study.b_targets", cs.b_targets, lambda m: cs.b_targets(m, *_directional(m), START)),
    ("overlay_risk_study.daily_frame.simple", rs.daily_frame, lambda m: rs.daily_frame(m["hourly"][BTC], "simple")),
    ("overlay_risk_study.daily_frame.ewma", rs.daily_frame, lambda m: rs.daily_frame(m["hourly"][BTC], "ewma")),
    ("overlay_risk_study.portfolio_multiplier", rs.portfolio_multiplier,
     lambda m: rs.portfolio_multiplier({s: rs.daily_frame(m["hourly"][s], "simple") for s in (BTC, ETH)})),
    ("overlay_risk_study.hourly_targets.combined", rs.hourly_targets,
     lambda m: rs.hourly_targets(m["hourly"], rs.VARIANTS["combined"])),
]

# -- the registry ---------------------------------------------------------------------------------------------------
FEATURE_MODULES = ("trading.domain.market", "trading.domain.strategy.overlay", "lab.breakout", "lab.candidates",
                   "lab.candidates_study", "lab.overlay_risk_study", "lab.pair", "lab.pair_study")
OTHER_MODULES = {
    "lab": "package marker",
    "trading.domain.strategy.base": ("the strategy interface (runtime extension R1); its implementations' targets "
                                     "are the checked functions (OverlayStrategy returns compute_targets exactly, "
                                     "tests/unit/trading/test_strategy_interface.py)"),
    "lab.breakout_study": "study driver of a closed line; its signal is lab.breakout",
    "lab.candidates_forward": "forward replay; it calls the checked functions of the candidate modules",
    "lab.causality": "the check itself",
    "lab.data": "file loaders; the aggregations it re-exports are trading.domain.market",
    "lab.forward": "forward-look guard and replay driver",
    "lab.metrics": "statistics of finished equity curves",
    "lab.midfreq": ("closed line (2026-11-14) on a 5-minute cross-sectional panel the check does not model; "
                    "its code only reproduces the recorded result"),
    "lab.midfreq_study": "study driver of a closed line",
    "lab.orders": "builds one order record",
    "lab.overlay_study": "study driver and report; the overlay's targets are trading.domain.strategy.overlay",
    "lab.paper_compare": "compares a paper record's fills with lab.simulate's on the same bars; no feature",
    "lab.rebalance": "re-export of the rebalancing planner (a target and a position in, orders out)",
    "lab.simulate": "the shared simulator",
    "lab.sizing": "sizing rule: numbers in, a number out",
    "lab.strategy": "re-export of trading.domain.strategy.overlay",
    "lab.study": "study driver (stage search) over the overlay's targets",
    "lab.trials": "trial registry",
    "lab.um_judge": ("the research lab's judge: periods, replays through lab.um_lab, walk-forward choice, score and "
                     "final check; its one rule over market data, the yearly universe, is checked in test_um_judge "
                     "(a list uses only bars from before its date); each idea it runs is checked with lab.causality "
                     "by the lab, and its final check fails an idea whose check was not run"),
    "lab.um_lab": ("the research lab's harness: snapshot loading, replays through the shared broker, portfolio "
                   "and walk-forward; its liquidity filter is checked in test_um_lab, each idea it runs is checked "
                   "with lab.causality by the lab (research/lab/README.md)"),
    "lab.validation": "walk-forward splits; their purge is tested in test_research_infra",
    "trading.domain.strategy": "package marker",
}
LABEL = "label: looks ahead by definition; training keeps only labels that ended before the test block"
FIT = "fits the rows its caller selects; the callers' selections are checked (b_walk_forward, ml_predictions)"
FINAL_FIT = "one final fit on the labels that ended by `limit` (pinned in test_candidates_study)"
NUMBER = "a number or one row in, a number out: no time axis"
SIMULATION = "simulation, not a feature (tests/architecture/test_research_fills.py)"
REPORT = "report or gate on finished results"
DRIVER = "study driver"
NOT_CHECKED = {
    "trading.domain.strategy.overlay.latest_target": "the last row of compute_targets: no time axis",
    "lab.breakout.desired_side": NUMBER, "lab.breakout.stop_distance": NUMBER,
    "lab.candidates.sized_weight": NUMBER, "lab.candidates.fit_ridge": FIT, "lab.candidates.directional_labels": LABEL,
    "lab.pair.load_inputs": "reads the snapshot files and cuts them at `limit`", "lab.pair.labels": LABEL,
    "lab.pair_study.fit_predict": FIT, "lab.pair_study.simulate_pair": SIMULATION, "lab.pair_study.yearly": REPORT,
    "lab.pair_study.describe": REPORT, "lab.pair_study.public": REPORT, "lab.pair_study.gate_checks": REPORT,
    "lab.pair_study.run_study": DRIVER, "lab.pair_study.main": DRIVER,
    "lab.candidates_study.run_sleeves": SIMULATION, "lab.candidates_study.fit_pair": FINAL_FIT,
    "lab.candidates_study.fit_directional": FINAL_FIT, "lab.candidates_study.flush_segments": REPORT,
    "lab.candidates_study.floor": REPORT, "lab.candidates_study.run_study": DRIVER, "lab.candidates_study.main": DRIVER,
    "lab.overlay_risk_study.brake_multiplier": NUMBER, "lab.overlay_risk_study.simulate_portfolio": SIMULATION,
    "lab.overlay_risk_study.yearly_max_drawdown": REPORT, "lab.overlay_risk_study.gate_checks": REPORT,
    "lab.overlay_risk_study.select": REPORT, "lab.overlay_risk_study.run_study": DRIVER,
    "lab.overlay_risk_study.main": DRIVER,
}


def _research_modules() -> set[str]:
    paths = [*(ROOT / "lab").rglob("*.py"), *(ROOT / "trading/domain/strategy").rglob("*.py"),
             ROOT / "trading/domain/market.py"]
    names = set()
    for path in paths:
        parts = path.relative_to(ROOT).with_suffix("").parts
        names.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def _public_functions(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    return {f"{module_name}.{name}" for name, fn in inspect.getmembers(module, inspect.isfunction)
            if fn.__module__ == module_name and not name.startswith("_")}


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    market_dir, premium_dir = write_snapshot(tmp_path_factory.mktemp("lookahead"), DAYS)
    return pair.load_inputs(market_dir, premium_dir, END)


@pytest.fixture(scope="module")
def probes(inputs):
    return variants(inputs, CUTS)


# -- tests ----------------------------------------------------------------------------------------------------------
def test_every_research_module_is_a_feature_module_or_says_why_not():
    assert not set(FEATURE_MODULES) & set(OTHER_MODULES)
    assert _research_modules() == set(FEATURE_MODULES) | set(OTHER_MODULES)


def test_every_public_function_of_a_feature_module_is_checked_or_says_why_not():
    checked = {f"{fn.__module__}.{fn.__name__}" for _, fn, _ in CASES}
    public = set().union(*(_public_functions(m) for m in FEATURE_MODULES))
    assert not checked & set(NOT_CHECKED)
    assert public == checked | set(NOT_CHECKED)


@pytest.mark.parametrize("case", [pytest.param(case, id=name) for name, _, case in CASES])
def test_no_output_changes_when_inputs_known_later_change_or_disappear(inputs, probes, case):
    assert check(case, inputs, probes) == []


# -- the check itself: it passes what is known in time and catches what is not ----------------------------------------
def _next_close(m):
    return m["hourly"][BTC].assign(next_close=lambda f: f["close"].shift(-1))           # the next bar's close


def _whole_sample_zscore(m):
    close = c.daily_close(m["hourly"][BTC])
    return (close - close.mean()) / close.std()                                          # mean of the whole sample


def _open_interest_without_the_lag(m):
    rows = m["metrics"][BTC]
    oi = pd.Series(rows["sum_open_interest"].to_numpy(float), index=rows["timestamp"].to_numpy(dtype=np.int64))
    times = m["hourly"][BTC]["available_at"].to_numpy(dtype=np.int64)
    return pd.Series(pair._last_value(oi, times, HOUR), index=times)                    # a row stamped t, used at t


def _ready_unless_last_day(m):
    hourly = m["hourly"][BTC]
    return hourly.assign(ready=np.arange(len(hourly)) < len(hourly) - 24)                # knows where the data ends


@pytest.mark.parametrize("case,where,per_cut", [
    (_next_close, "output['next_close']", 2),
    (_whole_sample_zscore, "output", 2),
    (_open_interest_without_the_lag, "output", 2),
    (_ready_unless_last_day, "output['ready']", 1),                 # only removing data shows it
])
def test_the_check_catches_an_injected_lookahead(inputs, case, where, per_cut):
    found = violations(case, inputs, CUTS)
    assert len(found) == per_cut * len(CUTS) and all(line.endswith(f"changes {where}") for line in found)


def test_values_known_exactly_at_the_cut_are_not_changed(inputs):
    assert violations(lambda m: m["hourly"][BTC], inputs, CUTS) == []
    assert violations(lambda m: m["funding"][ETH], inputs, CUTS) == []
    assert violations(lambda m: m["metrics"][ETH], inputs, CUTS) == []


def test_the_check_refuses_to_be_vacuous(inputs):
    with pytest.raises(ValueError, match="never changes the output"):
        violations(lambda m: pd.Series([1.0], index=[T0]), inputs, CUTS)
    with pytest.raises(ValueError, match="no input is known after"):
        violations(lambda m: m["hourly"][BTC], inputs, [END + DAY])
    with pytest.raises(ValueError, match="no output is known at"):
        violations(lambda m: m["hourly"][BTC], inputs, [T0 - DAY])
    with pytest.raises(TypeError, match="cannot tell when"):
        violations(lambda m: m["hourly"][BTC]["close"].to_numpy(), inputs, CUTS)
