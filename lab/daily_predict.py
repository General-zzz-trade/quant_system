"""The daily BTC/ETH prediction report (owner 2026-11-18: "目前做改成做预测机器", "我负责做下单", "先做BTC和ETH，
直接就在此对话推送"): the frozen panel models (research/results/panel_2026-11-17/forward_spec.json) on the newest
closed UTC day, for BTCUSDT and ETHUSDT, with a sizing rule; the owner places any order.

    python -m lab.daily_predict --feed <dir> --log <file> [--capital 150 --risk 0.01]

- Frozen only: the code that builds the features must be byte-identical to the spec's ``code_commit``, and every
  model file must match its SHA-256; otherwise no report.
- It predicts whether a trend comes within 14 days and the next 7 days' daily volatility. It does not predict up
  or down (the direction target failed on development data); the 40-day change it shows is a rule, not a prediction.
- A day that is not whole (the feed not updated past it, BTC or ETH without its 24 bars, more than 5 % of the
  contracts failing to update, fewer than 50 contracts in the universe) gives reasons instead of numbers.
- Every day's output is appended to a log (one line per day, each naming the hash of the line before). Nothing here
  measures an outcome: the frozen models are judged only at the looks registered in research/forward_looks.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from lab import feature_pool as fp
from lab import panel_study as ps

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/results/panel_2026-11-17"
SPEC = STUDY / "forward_spec.json"
CALIBRATION = STUDY / "calibration.json"
FROZEN_CODE = ("lab/panel_study.py", "lab/feature_pool.py", "lab/feature_study.py",
               "research/results/panel_2026-11-17/run.py")
COINS = ("BTCUSDT", "ETHUSDT")
MODELS = ("trend_14 logistic", "trend_14 lgbm", "fwd_vol_7 ridge", "fwd_vol_7 lgbm", "fwd_vol_7 naive")
H, DAY = ps.H, ps.DAY
STOP_K = 2.5                 # the stop distance: this many predicted daily standard deviations
MAX_LEVERAGE = 2.0           # the size never exceeds twice the capital (the 2026-11-17 leverage study)
MIN_UNIVERSE, MAX_FAILED = 50, 0.05


@lru_cache(maxsize=1)
def _panel_run():
    spec = importlib.util.spec_from_file_location("panel_2026_11_17_run", STUDY / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FROZEN_EXCLUDE = module.EXCLUDE
    return module


def code_changes(paths: Sequence[str], current: Callable[[str], bytes], frozen: Callable[[str], bytes]) -> list[str]:
    """The paths whose bytes differ from the frozen commit's."""
    return [p for p in paths if current(p) != frozen(p)]


def check_models(spec: dict, frozen_dir: Path) -> list[str]:
    """The model files that are missing or differ from the SHA-256 the spec froze."""
    bad = []
    for files in spec["models"].values():
        for name, digest in files.items():
            path = Path(frozen_dir) / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                bad.append(name)
    return bad


def model_paths(spec: dict, frozen_dir: Path) -> dict[str, Path]:
    return {name: Path(frozen_dir) / next(f for f in spec["models"][name] if not f.endswith(".meta.json"))
            for name in MODELS}


def sizing(daily_vol: float, *, capital: float, risk: float, min_notional: float | None, k: float = STOP_K) -> dict:
    """The rule the report suggests: a stop ``k`` predicted daily standard deviations away, a size that loses
    ``risk`` of the capital at the stop, at most ``MAX_LEVERAGE`` times the capital."""
    stop = k * daily_vol
    raw = capital * risk / stop
    notional = min(raw, MAX_LEVERAGE * capital)
    return {"stop_pct": stop, "notional": notional, "leverage": notional / capital, "capped": raw > notional,
            "min_notional": min_notional, "below_minimum": min_notional is not None and notional < min_notional}


def problems(state: dict, t: int, last_days: dict[str, int | None], *, universe: int) -> list[str]:
    """Why the day at ``t`` cannot be reported (empty: it can)."""
    out = []
    update = state.get("last_update") or {}
    if int(update.get("now_ms", 0)) < t:
        out.append(f"数据还没更新到 {_date(t)}")
    trading = update.get("trading") or []
    failed = update.get("failed") or []
    if trading and len(failed) / len(trading) > MAX_FAILED:
        out.append(f"{len(failed)} 个合约没有更新成功（超过 {MAX_FAILED:.0%}）")
    for coin in COINS:
        if last_days.get(coin) != t:
            out.append(f"{coin} 当天的 K 线不全")
    if universe < MIN_UNIVERSE:
        out.append(f"当天的币池只有 {universe} 个（少于 {MIN_UNIVERSE}）")
    return out


def _date(t: int) -> str:
    return str(pd.Timestamp(int(t), unit="ms").date())


def append_log(path: Path, record: dict) -> dict:
    """Append one day's line naming the SHA-256 of the line before; a day already logged is refused."""
    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    if any(json.loads(line).get("t") == record["t"] for line in lines):
        raise ValueError(f"{record['t']} is already in the log")
    entry = {**record, "prev": hashlib.sha256(lines[-1].encode()).hexdigest() if lines else "genesis"}
    with path.open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True, default=float) + "\n")
    return entry


def verify_log(path: Path) -> list[int]:
    """The line numbers (from 1) whose link to the line before is broken."""
    lines = Path(path).read_text().splitlines()
    bad = []
    for k, line in enumerate(lines):
        want = "genesis" if k == 0 else hashlib.sha256(lines[k - 1].encode()).hexdigest()
        if json.loads(line).get("prev") != want:
            bad.append(k + 1)
    return bad


def _tercile(value: float, cal: dict | None) -> dict | None:
    if not cal:
        return None
    k = int(np.searchsorted(cal["edges"], value, side="right"))
    return {"band": ("低", "中", "高")[k], "rate": cal["rate"][k], "base_rate": cal["base_rate"]}


def predict_day(feed: Path, t: int, models: dict[str, Path], *, calibration: dict | None,
                exclude_more: Sequence[str] = ()) -> dict:
    """The frozen models on the day ending at ``t``, through the frozen panel code over ``feed``."""
    run = _panel_run()
    run.SNAP, run.LIMIT = Path(feed), int(t)
    run.EXCLUDE = frozenset(run.FROZEN_EXCLUDE) | frozenset(exclude_more)
    panel, _, info = run.build_panel(np.array([int(t)], dtype=np.int64))
    panel["log_rv_30"] = np.log(panel["rv_30"])
    x = panel[list(ps.PANEL_FEATURES) + ["log_rv_30"]].astype(np.float32)
    pred = {name: ps.predict_frozen(ps.load_frozen(path), x) for name, path in models.items()}
    symbols = panel["symbol"].to_numpy()
    out: dict = {"t": int(t), "universe": int(len(panel)), "info": info, "coins": {}}
    for coin in COINS:
        rows = np.flatnonzero(symbols == coin)
        h, funding = ps.load_symbol(feed, coin, int(t))
        daily = fp.daily(h)
        if not len(rows) or daily.empty:
            out["coins"][coin] = {"last_day": int(daily.index[-1]) if len(daily) else None}
            continue
        i = int(rows[0])
        close = daily["close"]
        logret = np.log(close).diff()
        p = float(pred["trend_14 logistic"][i])
        recent = funding[funding.index > t - 3 * DAY]
        out["coins"][coin] = {
            "last_day": int(daily.index[-1]), "close": float(close.iloc[-1]),
            "trend_logistic": p, "trend_lgbm": float(pred["trend_14 lgbm"][i]),
            "trend_rank": float((pred["trend_14 logistic"] < p).mean()),
            "tercile": _tercile(p, (calibration or {}).get(coin)),
            "log_vol_ridge": float(pred["fwd_vol_7 ridge"][i]), "log_vol_lgbm": float(pred["fwd_vol_7 lgbm"][i]),
            "vol_pred": float(np.exp(np.mean([pred["fwd_vol_7 ridge"][i], pred["fwd_vol_7 lgbm"][i]]))),
            "vol_naive": float(np.exp(pred["fwd_vol_7 naive"][i])),
            "rv_30": float(logret.iloc[-30:].std()),
            "ret_40": float(close.iloc[-1] / close.loc[close.index <= t - 40 * DAY].iloc[-1] - 1),
            "funding_last": float(funding.iloc[-1]) if len(funding) else float("nan"),
            "funding_3d": float(recent.sum())}
    return out


def _pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:+.{digits}f}%" if v < 0 else f"{v * 100:.{digits}f}%"


def render(report: dict) -> str:
    """The report as text for the conversation (Chinese)."""
    day = _date(report["t"])
    if report.get("problems"):
        return (f"**BTC/ETH 每日预测 {day}（UTC 0 点收盘）：今天不出预测**\n\n原因：\n"
                + "".join(f"- {p}\n" for p in report["problems"]))
    c = report["coins"]
    s = report["sizing"]

    def row(label: str, fn: Callable[[str], str]) -> str:
        return f"| {label} | " + " | ".join(fn(coin) for coin in COINS) + " |\n"

    def band(coin: str) -> str:
        b = c[coin]["tercile"]
        if not b:
            return "（没有历史对照）"
        return f"{b['band']}档：开发期该档 14 天内走出趋势 {b['rate']:.0%}（平均 {b['base_rate']:.0%}）"

    def size(coin: str) -> str:
        z = s[coin]
        text = f"{z['notional']:.0f} USDT（{z['leverage']:.2f} 倍）"
        if z["capped"]:
            text += "，已按 2 倍封顶"
        if z["below_minimum"]:
            text += f"，低于最小下单额 {z['min_notional']:.0f}，按规则下不了"
        return text

    lines = [f"**BTC/ETH 每日预测 {day}（UTC 0 点收盘）**\n\n",
             "| | BTC | ETH |\n|---|---|---|\n",
             row("收盘价", lambda k: f"{c[k]['close']:,.2f}"),
             row("14 天内走出趋势的概率（logistic / LightGBM）",
                 lambda k: f"{c[k]['trend_logistic']:.1%} / {c[k]['trend_lgbm']:.1%}"),
             row(f"在当天 {report['universe']} 个币里的位置", lambda k: f"高于 {c[k]['trend_rank']:.0%} 的币"),
             row("按自己的历史", band),
             row("预测未来 7 天的日波动", lambda k: f"{c[k]['vol_pred']:.2%}（简单预测 {c[k]['vol_naive']:.2%}）"),
             row("过去 30 天的日波动", lambda k: f"{c[k]['rv_30']:.2%}"),
             row("40 天涨跌（规则，不是预测）", lambda k: _pct(c[k]["ret_40"])),
             row("资金费：最近一次 / 3 天合计", lambda k: f"{c[k]['funding_last']:.4%} / {c[k]['funding_3d']:.4%}"),
             row(f"参考止损距离（{STOP_K} 倍预测日波动）", lambda k: f"{s[k]['stop_pct']:.1%}"),
             row(f"参考仓位（{report['capital']:.0f} USDT，每笔亏 {report['risk']:.0%}）", size),
             "\n- **不预测涨跌**：方向模型在开发期没过检验。做多做空由你定，40 天涨跌只是规则参考。\n",
             "- 趋势和波动的预测力来自开发期（2021–2025），冻结模型的前向检验在 2027-06-01 判定。\n",
             "- 每笔都在交易所挂止损；仓位按上面的规则算，不加码。\n"]
    return "".join(lines)


def _git_show(commit: str) -> Callable[[str], bytes]:
    return lambda path: subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True,
                                       check=True).stdout


def main(argv: list[str] | None = None) -> int:
    from data.downloads import perps_feed
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feed", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--capital", type=float, default=150.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--no-update", action="store_true")
    args = parser.parse_args(argv)
    spec = json.loads(SPEC.read_text())
    report: dict = {"problems": []}
    changed = code_changes(FROZEN_CODE, lambda p: (ROOT / p).read_bytes(), _git_show(spec["code_commit"]))
    bad = check_models(spec, STUDY / "frozen")
    state = json.loads((args.feed / perps_feed.STATE).read_text())
    if not args.no_update:
        state["last_update"] = perps_feed.update(args.feed)
    now = int(state["last_update"]["now_ms"]) if args.no_update else int(time.time() * 1000)
    t = now // DAY * DAY
    report["t"] = t
    if changed or bad:
        report["problems"] = [f"冻结的代码或模型被改过：{', '.join(changed + bad)}"]
    else:
        calibration = json.loads(CALIBRATION.read_text())["coins"] if CALIBRATION.exists() else None
        try:
            report.update(predict_day(args.feed, t, model_paths(spec, STUDY / "frozen"), calibration=calibration,
                                      exclude_more=state["last_update"].get("tradfi", [])))
            last = {k: v.get("last_day") for k, v in report["coins"].items()}
            report["problems"] = problems(state, t, last, universe=report["universe"])
        except Exception as exc:                                    # noqa: BLE001 - reported, not hidden
            report["problems"] = [f"计算失败：{type(exc).__name__}: {exc}"]
    if not report["problems"]:
        rules = state["last_update"].get("rules", {})
        report["capital"], report["risk"] = args.capital, args.risk
        minimum = {coin: (rules.get(coin) or {}).get("min_notional") for coin in COINS}
        report["sizing"] = {coin: sizing(report["coins"][coin]["vol_pred"], capital=args.capital, risk=args.risk,
                                         min_notional=float(minimum[coin]) if minimum[coin] else None)
                            for coin in COINS}
    record = {"t": t, "date": _date(t), "problems": report["problems"], "spec_sha256": hashlib.sha256(
        SPEC.read_bytes()).hexdigest(), "coins": report.get("coins"), "universe": report.get("universe"),
        "sizing": report.get("sizing"), "capital": args.capital, "risk": args.risk}
    try:
        append_log(args.log, record)
    except ValueError:
        pass                                                         # the day is already logged: report again only
    print(render(report))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
