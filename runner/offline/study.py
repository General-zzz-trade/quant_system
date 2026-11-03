"""Chronological training, validation, frozen holdout and cost stress reports."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from data.quality.causal_bars import dataset_manifest, fingerprint
from .broker import ReplayConfig
from .models import train_bundle
from .replay import run_replay
from .report import code_identity, metrics, write_json, write_result


def run_study(bars, funding, out, *, synthetic=False, config=None):
    config = config or ReplayConfig()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory; preserve previous experiments")
    if len(bars) < 1200:
        raise ValueError("need at least 1200 hourly bars for this split; longer real history is recommended")
    n = len(bars)
    train_end, holdout_start = int(n * .6), int(n * .8)
    bundle = train_bundle(bars[:train_end], bars[train_end].open_time)
    # Freeze the protocol BEFORE opening final-test results. One predeclared candidate.
    protocol = {"schema": 1, "dataset": dataset_manifest(bars, synthetic=synthetic), "code": code_identity(),
                "config": config.record(), "train_end_ms": bars[train_end].open_time,
                "holdout_start_ms": bars[holdout_start].open_time, "model_sha256": bundle["sha256"],
                "candidate_count": 1, "selection": "fixed Ridge alpha=1; no parameter search",
                "funding_time_policy": "nominal UTC 8h boundary, before opening fills; source jitter may be <=1 second",
                "funding_sha256": fingerprint([[t, str(r), str(m)] for t, (r, m) in sorted(funding.items())]),
                "limitations": ["Reference OHLCV Ridge, not certification of existing live model artifacts",
                    "OHLC execution assumptions cannot prove maker queue fills",
                    "Historical independence cannot be established against undocumented prior experiments",
                    "Baseline exposure differs; compare average gross exposure, not just return",
                    "USDT linear perpetual, <=1x cap, fixed UTC 8h funding; no exchange liquidation certification"]}
    write_json(out / "protocol.json", protocol)
    write_json(out / "model.json", bundle)
    summary = {"synthetic": synthetic, "profitability_verified": False,
               "evidence": "engineering fixture only" if synthetic else "historical holdout under recorded assumptions",
               "stages": {}}
    # Expanding-origin folds only within the development portion (first 80%).
    folds = []
    for index in range(3):
        begin = int(n * (.4 + index * .4 / 3))
        end = int(n * (.4 + (index + 1) * .4 / 3))
        model = train_bundle(bars[:begin], bars[begin].open_time)
        result = run_replay(bars[:end], model, funding, start_ms=bars[begin].open_time,
                            config=config, synthetic=synthetic)
        fold_out = out / f"development_fold_{index + 1}"
        write_json(fold_out / "model.json", model)
        write_result(fold_out, result)
        folds.append({"start_ms": bars[begin].open_time, "end_ms": bars[end - 1].available_at,
                      "model_sha256": model["sha256"], **metrics(result)})
    summary["development_folds"] = folds
    validation = run_replay(bars[:holdout_start], bundle, funding, start_ms=bars[train_end].open_time,
                            config=config, synthetic=synthetic)
    write_result(out / "validation", validation)
    summary["stages"]["validation"] = metrics(validation)
    stress = replace(config, maker_bps=str(Decimal(config.maker_bps) * 2),
                     taker_bps=str(Decimal(config.taker_bps) * 2), slippage_bps=str(Decimal(config.slippage_bps) * 2))
    for name, mode, cfg in (("holdout", "strategy", config), ("holdout_cost_x2", "strategy", stress),
                             ("buy_hold", "buy_hold", config), ("trend_50", "trend", config)):
        result = run_replay(bars, bundle, funding, start_ms=bars[holdout_start].open_time,
                            config=cfg, mode=mode, synthetic=synthetic)
        write_result(out / name, result)
        summary["stages"][name] = metrics(result)
    write_json(out / "summary.json", summary)
    label = "合成行情工程验收：不代表真实收益" if synthetic else "历史样本外评估：不等于未来盈利保证"
    lines = [f"# {label}", "", f"数据指纹：`{protocol['dataset']['sha256']}`", "",
             "训练 60%、验证 20%、最终测试 20%，按时间划分。最终测试前冻结模型与参数。", "",
             "| 阶段 | 净收益 % | 最大回撤 % | 成交数 | 平均总敞口 |", "|---|---:|---:|---:|---:|"]
    for name, m in summary["stages"].items():
        lines.append(f"| {name} | {m['net_return_pct']:.3f} | {m['max_drawdown_pct']:.3f} | {m['fills']} | "
                     f"{m['average_gross_exposure']:.3f} |")
    lines += ["", "收益按逐时盯市权益计算，包含未实现盈亏、手续费和资金费；期末持仓未强制平仓。",
              "日收益样本不足 30 个时不报告夏普。基准使用相同成本与撮合规则，实际敞口可能不同。",
              "", "这些结果验证新建 OHLCV Ridge 研究基线；原有实盘模型尚未取得独立样本外认证。",
              "成本压力测试属于预先声明的敏感性分析，不能用于反复挑选最终测试集上的参数。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
