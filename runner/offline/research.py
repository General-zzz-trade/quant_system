"""Predeclared candidate selection on validation, then ONE frozen final test.

Protocol (all decided before any final-test result exists):

1. Chronological split: train 60%, validation 20%, final test 20%.
2. Candidates = feature set x label horizon x Ridge alpha x deadzone, listed
   in ``CANDIDATES``. Ridge alpha is chosen per (feature set, horizon) by
   validation IC; deadzone by validation replay Sharpe (daily, net of costs).
   Training labels always end before the validation start.
3. The chosen spec is retrained on train+validation (labels end before the
   final-test start) and written to ``protocol.json`` together with every
   candidate's validation result, the data fingerprints and code identity.
4. Only then is the final test replayed, plus a 2x cost stress and buy-hold /
   50-bar trend baselines under identical fills and costs.

The final test may be looked at once. Re-running this study after changing
anything because of a final-test result invalidates that test; the protocol
records ``final_test_views`` so repeated looks are visible.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np

from data.quality.causal_bars import aggregate_closed, dataset_manifest, fingerprint
from .broker import ReplayConfig
from .models import FEATURE_SETS, design, feature_rows, fit_samples, normalize_spec, train_bundle
from .replay import run_replay
from .report import code_identity, metrics, write_json, write_result

CANDIDATES = {
    "feature_set": ("ohlcv", "derivatives"),
    "horizon": (1, 4, 24),
    "alpha": (1.0, 10.0, 100.0),
    "deadzone": (1.0, 1.5, 2.0),
}
MIN_VALIDATION_FILLS = 20
WARMUP_BARS = 3000  # replay warmup before each evaluation window (features, z-score, regimes)
HOURS_PER_YEAR = 365 * 24

# Predeclared 2026-11-13 (research/protocols/2026-11-13-overlay-and-ml-recheck.md), before any
# validation-only run on real data. The chosen candidate is the best of the grid ON the validation
# window, so these thresholds are deliberately strict; passing only earns a forward paper test.
VALIDATION_GATE = {
    "validation_sharpe_min_exclusive": 0.5,          # daily Sharpe of the chosen candidate, costs x1
    "validation_cost_x2_sharpe_min_exclusive": 0.0,  # same candidate, maker/taker/slippage doubled
    "validation_fills_per_year_max": 150,
}


def hold_for(horizon: int) -> tuple[int, int]:
    """Predeclared holding constraints tied to the label horizon."""
    min_hold = min(max(horizon, 6), 24)
    return min_hold, 6 * min_hold


def _ic(pred: np.ndarray, label: np.ndarray) -> float:
    if len(pred) < 30 or pred.std() < 1e-15 or label.std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(pred, label)[0, 1])


def _window(bars, start_index, end_index):
    """Bars for a replay evaluating [start_index, end_index) with bounded warmup."""
    begin = max(0, start_index - WARMUP_BARS)
    return bars[begin:end_index], bars[start_index].open_time


def _sharpe(m) -> float:
    value = m.get("daily_sharpe")
    return float(value) if value is not None else float("nan")


def select_alpha(bars, train_end, val_end, *, symbol, context, feature_set, horizon):
    """Fit each alpha on train labels, score IC on validation labels (no replay)."""
    cutoff, val_cutoff = bars[train_end].open_time, bars[val_end].open_time
    rows = feature_rows(bars[:val_end], symbol, context)
    spec0 = {"feature_set": feature_set, "horizon": horizon}
    train = design(bars[:val_end], cutoff, symbol=symbol, spec=spec0, rows=rows)
    everything = design(bars[:val_end], val_cutoff, symbol=symbol, spec=spec0, rows=rows)
    first_val = len(train)
    # Validation samples: features observed at/after the validation start.
    validation = [s for s in everything[first_val:] if s[2] - horizon * 3_600_000 >= cutoff]
    scores = {}
    for alpha in CANDIDATES["alpha"]:
        spec = normalize_spec({**spec0, "alpha": alpha})
        model = fit_samples(train, spec, training_sha256="selection")
        x = np.stack([s[0] for s in validation]) if validation else np.zeros((0, len(model["features"])))
        y = np.asarray([s[1] for s in validation])
        pred = ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["weights"])
        scores[alpha] = {"validation_ic": _ic(pred, y), "train_samples": len(train),
                         "validation_samples": len(validation)}
    finite = {a: v for a, v in scores.items() if math.isfinite(v["validation_ic"])}
    best = max(finite, key=lambda a: finite[a]["validation_ic"]) if finite else CANDIDATES["alpha"][0]
    return best, scores


def _stress(config):
    return replace(config, maker_bps=str(Decimal(config.maker_bps) * 2),
                   taker_bps=str(Decimal(config.taker_bps) * 2),
                   slippage_bps=str(Decimal(config.slippage_bps) * 2))


def passes_validation_gate(chosen, stress_metrics, validation_years) -> tuple[bool, list[str]]:
    """Predeclared go/no-go for a validation-only re-evaluation (see ``VALIDATION_GATE``)."""
    if chosen is None:
        return False, ["no validation candidate met the eligibility rule"]
    reasons = []
    sharpe = chosen["validation_sharpe_daily"]
    if not (math.isfinite(sharpe) and sharpe > VALIDATION_GATE["validation_sharpe_min_exclusive"]):
        reasons.append(f"validation daily Sharpe {sharpe} <= {VALIDATION_GATE['validation_sharpe_min_exclusive']}")
    stress = _sharpe(stress_metrics) if stress_metrics else float("nan")
    if not (math.isfinite(stress) and stress > VALIDATION_GATE["validation_cost_x2_sharpe_min_exclusive"]):
        reasons.append(f"validation Sharpe under 2x costs {stress} <= "
                       f"{VALIDATION_GATE['validation_cost_x2_sharpe_min_exclusive']}")
    per_year = chosen["fills"] / validation_years
    if per_year > VALIDATION_GATE["validation_fills_per_year_max"]:
        reasons.append(f"{per_year:.0f} fills per year > {VALIDATION_GATE['validation_fills_per_year_max']}")
    return not reasons, reasons


def run_research(bars, funding, out, *, symbol, context=None, config=None, synthetic=False,
                 validation_only=False):
    """``validation_only`` stops after candidate selection: no final model, no final-test replay."""
    config = config or ReplayConfig()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory; preserve previous experiments")
    n = len(bars)
    if n < 5000:
        raise ValueError("need at least 5000 hourly bars for candidate selection")
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    cutoff = bars[train_end].open_time
    candidates, selections, bundles = [], [], {}
    for feature_set in CANDIDATES["feature_set"]:
        if feature_set == "derivatives" and context is None:
            continue
        for horizon in CANDIDATES["horizon"]:
            try:
                alpha, alpha_scores = select_alpha(bars, train_end, val_end, symbol=symbol, context=context,
                                                   feature_set=feature_set, horizon=horizon)
                spec = normalize_spec({"feature_set": feature_set, "horizon": horizon, "alpha": alpha})
                bundle = train_bundle(bars[:train_end], cutoff, symbol=symbol, context=context, spec=spec)
                bundles[(feature_set, horizon)] = bundle
            except ValueError as exc:
                selections.append({"feature_set": feature_set, "horizon": horizon, "error": str(exc)})
                continue
            selections.append({"feature_set": feature_set, "horizon": horizon, "alpha": alpha,
                               "alpha_scores": {str(k): v for k, v in alpha_scores.items()}})
            min_hold, max_hold = hold_for(horizon)
            window, start_ms = _window(bars, train_end, val_end)
            for deadzone in CANDIDATES["deadzone"]:
                cfg = replace(config, deadzone=deadzone, min_hold=min_hold, max_hold=max_hold)
                result = run_replay(window, bundle, funding, start_ms=start_ms, config=cfg,
                                    synthetic=synthetic, symbol=symbol, context=context)
                m = metrics(result)
                candidates.append({"spec": spec, "deadzone": deadzone, "min_hold": min_hold,
                                   "max_hold": max_hold, "validation_sharpe_daily": _sharpe(m),
                                   **{k: m[k] for k in ("net_return_pct", "max_drawdown_pct", "fills",
                                                        "average_gross_exposure")}})
    eligible = [c for c in candidates if c["fills"] >= MIN_VALIDATION_FILLS
                and math.isfinite(c["validation_sharpe_daily"])]
    chosen = max(eligible, key=lambda c: c["validation_sharpe_daily"]) if eligible else None
    holdout_start = bars[val_end].open_time
    protocol = {
        "schema": 2, "symbol": symbol, "dataset": dataset_manifest(bars, synthetic=synthetic),
        "code": code_identity(), "base_config": config.record(),
        "split": {"train_end_ms": cutoff, "validation_end_ms": holdout_start, "final_test_start_ms": holdout_start},
        "candidate_grid": {k: list(v) for k, v in CANDIDATES.items()},
        "candidate_count": len(candidates), "selection_rule":
            f"alpha by validation IC; deadzone/spec by validation daily Sharpe with >= {MIN_VALIDATION_FILLS} fills",
        "alpha_selection": selections, "validation_candidates": candidates, "chosen": chosen,
        "funding_sha256": fingerprint([[t, str(r), str(m)] for t, (r, m) in sorted(funding.items())]),
        "final_test_views": 1,
        "limitations": [
            "Ridge reference model on incremental production features; not the legacy LGBM/XGB ensemble",
            "OHLC fills cannot prove maker queue position; exits pay slippage",
            "Exposure capped at 1x equity in replay; live cap is 3x",
            "Earlier undocumented research may have used the same history",
        ],
    }
    summary = {"symbol": symbol, "synthetic": synthetic, "profitability_verified": False, "stages": {},
               "chosen": chosen}
    if validation_only:
        return _finish_validation_only(bars, funding, out, protocol, summary, bundles, chosen, config,
                                       train_end=train_end, val_end=val_end, symbol=symbol, context=context,
                                       synthetic=synthetic)
    if chosen is None:
        protocol["outcome"] = "NO_CANDIDATE: no validation candidate met the eligibility rule; final test not run"
        write_json(out / "protocol.json", protocol)
        write_json(out / "summary.json", summary)
        _report(out, protocol, summary)
        return summary
    final_bundle = train_bundle(bars[:val_end], holdout_start, symbol=symbol, context=context, spec=chosen["spec"])
    protocol["final_model_sha256"] = final_bundle["sha256"]
    final_cfg = replace(config, deadzone=chosen["deadzone"], min_hold=chosen["min_hold"],
                        max_hold=chosen["max_hold"])
    protocol["final_config"] = final_cfg.record()
    # Freeze protocol and model BEFORE the final test is evaluated.
    write_json(out / "protocol.json", protocol)
    write_json(out / "model.json", final_bundle)
    window, start_ms = _window(bars, val_end, n)
    stress = replace(final_cfg, maker_bps=str(Decimal(final_cfg.maker_bps) * 2),
                     taker_bps=str(Decimal(final_cfg.taker_bps) * 2),
                     slippage_bps=str(Decimal(final_cfg.slippage_bps) * 2))
    for name, mode, cfg in (("final_test", "strategy", final_cfg), ("final_test_cost_x2", "strategy", stress),
                             ("buy_hold", "buy_hold", final_cfg), ("trend_50", "trend", final_cfg)):
        result = run_replay(window, final_bundle, funding, start_ms=start_ms, config=cfg, mode=mode,
                            synthetic=synthetic, symbol=symbol, context=context)
        write_result(out / name, result)
        m = metrics(result)
        summary["stages"][name] = {**m, "sharpe_daily": _sharpe(m)}
    write_json(out / "summary.json", summary)
    _report(out, protocol, summary)
    return summary


def _finish_validation_only(bars, funding, out, protocol, summary, bundles, chosen, config, *, train_end, val_end,
                            symbol, context, synthetic):
    """Validation-window stress test of the chosen candidate and the predeclared gate; nothing later."""
    stress = None
    if chosen is not None:
        spec = chosen["spec"]
        cfg = replace(config, deadzone=chosen["deadzone"], min_hold=chosen["min_hold"], max_hold=chosen["max_hold"])
        window, start_ms = _window(bars, train_end, val_end)
        stress = metrics(run_replay(window, bundles[(spec["feature_set"], spec["horizon"])], funding,
                                    start_ms=start_ms, config=_stress(cfg), synthetic=synthetic, symbol=symbol,
                                    context=context))
    validation_years = (val_end - train_end) / HOURS_PER_YEAR
    ok, reasons = passes_validation_gate(chosen, stress, validation_years)
    protocol.update(mode="validation_only", final_test_views=0, validation_gate=VALIDATION_GATE,
                    validation_gate_result={"passed": ok, "reasons": reasons,
                                            "chosen_cost_x2": None if stress is None else {
                                                "validation_sharpe_daily": stress.get("daily_sharpe"),
                                                "net_return_pct": stress["net_return_pct"],
                                                "fills": stress["fills"]}},
                    outcome=("PASSED validation gate: next test is forward paper" if ok else
                             "FAILED validation gate: " + "; ".join(reasons)))
    protocol.pop("final_model_sha256", None)
    summary.update(mode="validation_only", validation_gate={"passed": ok, "reasons": reasons})
    write_json(out / "protocol.json", protocol)
    write_json(out / "summary.json", summary)
    write_json(out / "gate.json", {"kind": "validation_only", "passed": ok, "reasons": reasons})
    lines = [f"# {summary['symbol']} 仅验证集重评（不运行最终测试）", "",
             f"候选数：{protocol['candidate_count']}；验证门槛：通过" if ok else
             f"候选数：{protocol['candidate_count']}；验证门槛：未通过", ""]
    lines += [f"- {r}" for r in reasons]
    (Path(out) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def passes_gate(summary) -> tuple[bool, list[str]]:
    """Predeclared go/no-go rule for moving a model toward live trading."""
    reasons = []
    stages = summary.get("stages", {})
    final, stress = stages.get("final_test"), stages.get("final_test_cost_x2")
    if not final:
        return False, ["no final test was run"]
    sharpe = final.get("sharpe_daily")
    if not (isinstance(sharpe, float) and math.isfinite(sharpe) and sharpe > 1.0):
        reasons.append(f"final-test daily Sharpe {sharpe} <= 1.0")
    if not stress or not stress["net_return_pct"] > 0:
        reasons.append("net return not positive under 2x costs")
    if final["fills"] < 100:
        reasons.append(f"only {final['fills']} fills (< 100)")
    for base in ("buy_hold", "trend_50"):
        b = stages.get(base)
        if b and final.get("average_gross_exposure", 0) > 0 and b.get("average_gross_exposure", 0) > 0:
            per_exposure = final["net_return_pct"] / final["average_gross_exposure"]
            base_per_exposure = b["net_return_pct"] / b["average_gross_exposure"]
            if per_exposure <= base_per_exposure:
                reasons.append(f"does not beat {base} per unit of exposure")
    return not reasons, reasons


def _report(out, protocol, summary):
    ok, reasons = passes_gate(summary)
    lines = [f"# {summary['symbol']} 研究结果（预先声明的候选选择 + 一次性最终测试）", "",
             f"数据指纹：`{protocol['dataset']['sha256']}`，候选数：{protocol['candidate_count']}", "",
             "## 验证集候选（只用于选择）", "",
             "| 特征 | 周期 | alpha | deadzone | 验证夏普(日) | 净收益% | 成交数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for c in protocol["validation_candidates"]:
        s = c["spec"]
        lines.append(f"| {s['feature_set']} | {s['horizon']} | {s['alpha']:g} | {c['deadzone']} | "
                     f"{c['validation_sharpe_daily']:.2f} | {c['net_return_pct']:.2f} | {c['fills']} |")
    lines += ["", "## 最终测试", ""]
    if not summary["stages"]:
        lines.append(protocol.get("outcome", "未运行"))
    else:
        lines += ["| 阶段 | 净收益 % | 最大回撤 % | 成交数 | 平均总敞口 | 夏普(日) |", "|---|---:|---:|---:|---:|---:|"]
        for name, m in summary["stages"].items():
            lines.append(f"| {name} | {m['net_return_pct']:.3f} | {m['max_drawdown_pct']:.3f} | {m['fills']} | "
                         f"{m['average_gross_exposure']:.3f} | {m['sharpe_daily']:.2f} |")
    lines += ["", f"## 上线门槛：{'通过' if ok else '未通过'}", ""] + [f"- {r}" for r in reasons]
    lines += ["", "最终测试只允许看一次；据此修改任何东西后重跑，该测试即失效。"]
    (Path(out) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (Path(out) / "gate.json").write_text(json.dumps({"passed": ok, "reasons": reasons}, ensure_ascii=False,
                                                    indent=2) + "\n", encoding="utf-8")


__all__ = ["CANDIDATES", "VALIDATION_GATE", "run_research", "passes_gate", "passes_validation_gate", "hold_for",
           "aggregate_closed", "FEATURE_SETS"]
