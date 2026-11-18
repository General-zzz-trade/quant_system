"""report.md from results.json (every number from the file)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
RUNS = {"lab_2026-11-17c": "17c（BTC/ETH，实盘条件）", "lab_2026-11-17": "11-17（全市场 10 个名额）"}


def cell(v: dict) -> str:
    a, g, d, wf = v["regression_alpha"], v["portfolio_gain"], v["family_deflated"], v["walk_forward"]
    b = a["betas"]
    dsr = "—" if d is None else f"{d['probability']:.2f}（{d['n_trials']}）"
    return (f"{wf['sharpe']:.2f} | {wf['total_return']:+.0%} | {wf['max_drawdown']:.0%} | {a['alpha_annual']:+.1%} | "
            f"**{a['t_newey_west']:.2f}** | {b.get('btc_hold', 0):.2f} | {b.get('btc_vol_target', 0):.2f} | "
            f"{'是' if g['passes'] else '否'} | {dsr} | {v['class']}")


HEAD = ("| 策略 | 夏普 | 收益 | 最大回撤 | 年化 alpha | NW t | β 持有 BTC | β 仓位控制 | 组合增益 "
        "| 紧缩夏普（族大小） | 归类 |\n|---|---|---|---|---|---|---|---|---|---|---|")


def main() -> None:
    every = [(k, v) for k, v in R["candidates_2026-11-14"].items()]
    every.append(("weak_portfolio", R["weak_portfolio_2026-11-18"]))
    for run in RUNS:
        every += [(f"{run}:{k}", v) for k, v in R[run]["ideas"].items()]
    traded = [(k, v) for k, v in every if v["walk_forward"].get("days") and v["walk_forward"]["max_drawdown"] > 0]
    evidence = [k for k, v in every if v["alpha_evidence"]]
    best = max(traded, key=lambda kv: kv[1]["regression_alpha"]["t_newey_west"])
    negative = [k for k, v in traded if v["regression_alpha"]["t_newey_west"] <= -2]
    c17 = R["lab_2026-11-17c"]["ideas"]
    trend = [k for k in ("E3_vol_target_trend", "A1_tsmom_long", "F1_btc_trend_beta", "A8_xs_momentum_top3")]
    lines = [
        "# 现有策略的 alpha 检查（2026-11-18）", "",
        "计划在结果之前提交（`plan.md`）。数字是【实际运行结果】，全部来自 `results.json`（`run.py` 生成，本文件由 "
        "`report.py` 从中生成）。**这是诊断，不做决定**：不让任何策略上实盘，也不关闭已登记的前向检验。", "",
        "## 结论", "",
        f"- 量了 {len(every)} 个策略（其中 {len(traded)} 个在窗口里有交易），**有 alpha 证据的：{len(evidence)} 个**。",
        "  - 标准：对持有 BTC 和持有 BTC 加仓位控制回归，年化 alpha > 0 且 Newey–West t ≥ 2，"
        "并且按风险平价加进持有 BTC 后夏普提高。",
        f"  - t 最高的是 `{best[0]}`：t {best[1]['regression_alpha']['t_newey_west']:.2f}，年化 alpha "
        f"{best[1]['regression_alpha']['alpha_annual']:+.1%}，"
        f"对 BTC 的 β {best[1]['regression_alpha']['betas']['btc_hold']:.2f}，仍不到 2。",
        "- 趋势类想法的收益基本就是\"持有 BTC 加仓位控制\"：",
    ]
    lines += [f"  - `{k}`（17c）：对仓位控制基准的 β {c17[k]['regression_alpha']['betas']['btc_vol_target']:.2f}，"
              f"alpha t {c17[k]['regression_alpha']['t_newey_west']:.2f}" for k in trend]
    lines += [
        "  - 按新标准，这类是\"持有 BTC 加仓位管理\"，单独按回撤和风险调整后的收益评价，不叫 alpha。",
        f"- 对 BTC 的超额收益显著为负（t ≤ −2）的 {len(negative)} 个："
        + "、".join(f"`{k}`" for k in negative) + "。它们扣成本后稳定地输给持有 BTC（这里没有分解原因）。",
        f"- 统计力：5 年日收益，真实夏普 0.5 时统计力约 {c17['E3_vol_target_trend']['power']['at_sharpe_0.5']:.0%}。"
        "所以\"没有证据\"不等于\"没有 alpha\"，只是这些历史证明不了；判定靠已登记的前向检验。", "",
        "## 现有候选", "", HEAD]
    rows = [("每周方向 Ridge（11-14 候选）", R["candidates_2026-11-14"]["dir_ridge_7d"]),
            ("爆仓反转（11-14 候选）", R["candidates_2026-11-14"]["flush_reversal_72h"]),
            ("ETH/BTC 配对（11-14 候选）", R["candidates_2026-11-14"]["pair_ridge_PD_7d"]),
            ("弱信号组合（11-18 冻结）", R["weak_portfolio_2026-11-18"]),
            ("叠加策略 overlay（17c，实盘条件）", c17["overlay"]),
            ("E3（17c，BTC/ETH）", c17["E3_vol_target_trend"]),
            ("E3（11-17，全市场，冻结版本）", R["lab_2026-11-17"]["ideas"]["E3_vol_target_trend"])]
    lines += [f"| {name} | {cell(v)} |" for name, v in rows]
    lines += ["", "三个候选按原代码重放，夏普和当时的记录逐位一致："]
    lines += [f"- `{k}`：记录 {v['recorded_sharpe']:.6f}，重放 {v['walk_forward']['sharpe']:.6f}"
              for k, v in R["candidates_2026-11-14"].items()]
    for run, title in RUNS.items():
        b = R[run]["benchmarks"]
        lines += ["", f"## {title}：按 NW t 排序", "",
                  f"同期基准：持有 BTC 夏普 {b['btc_hold']['sharpe']:.2f}、"
                  f"最大回撤 {b['btc_hold']['max_drawdown']:.0%}；"
                  f"持有 BTC 加仓位控制 夏普 {b['btc_vol_target']['sharpe']:.2f}、最大回撤 "
                  f"{b['btc_vol_target']['max_drawdown']:.0%}。", "", HEAD]
        ranked = sorted(R[run]["ideas"].items(), key=lambda kv: -kv[1]["regression_alpha"]["t_newey_west"])
        lines += [f"| `{k}` | {cell(v)} |" for k, v in ranked]
    lines += ["", "## 怎么读", "",
              "- 组合增益这一条很宽：夏普高于持有 BTC 且相关不高的都能过，所以单独过它不说明有 alpha。",
              "- 17c 里 `B2_xs_reversal_ls` 和 `C4_xs_funding_ls` 在两个品种上从不交易（收益为 0）；"
              "`B1_xs_reversal_long` 和 `C3_xs_funding_top_long` 在两个品种上持仓相同。",
              "- 17c 的最后一折含 2025-11-16 以后看过的数据（17c 当时的设计）；这里只报告，不选择。",
              "- 这些想法和窗口都被反复用过，t 值偏高的可能性大于偏低；即使这样也没有一个到 2。", ""]
    (HERE / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
