"""Monthly forward looks of the frozen panel models (research/protocols/2026-11-17-panel-forward.md; tech lead
2026-11-17 "执行"). Six looks, ends 2027-01-01 .. 2027-06-01; looks 1-5 report only, look 6 judges.

    python -m lab.panel_forward --end 2027-01-01 --data data_files/binance_usdt_perps_2020_2026-12-31 \\
        --out research/results/panel_2026-11-17/looks/2027-01-01

- Refuses a look that is not registered in research/forward_looks.jsonl for the frozen spec (by its SHA-256),
  registered on or after its end, or whose end has not fully passed (``lab.forward.registered_look``); a registry
  that does not hold the protocol's six looks; frozen code or model files that differ from the frozen ones (the
  checks of ``lab.daily_predict``); a snapshot whose BTCUSDT or ETHUSDT lacks an hour of the forward window, or a
  decision day with fewer than ``min_universe`` contracts.
- Builds the panel with the frozen code (``run.build_panel`` of the study, bars to the end only), the universe rule
  of the freeze plus every contract the public exchangeInfo names TRADIFI_PERPETUAL at the look (``--tradfi`` reads
  it; the new ones are listed). Decision days: every UTC day from 2026-11-13 while the label ends before the look's
  end (t < end - the label's days). The frozen models predict as they are: no refit, no new fill medians.
- Measures, with the development functions: trend AUC; volatility R2 minus the frozen naive forecast's R2; direction
  the mean daily cross-sectional Spearman IC (days of at least 10 contracts). Pooled, per segment (look k adds the
  decision days [end of look k-1 - label days, end of look k - label days), the first from 2026-11-13) and on the
  BTCUSDT and ETHUSDT rows (reported only).
- Look 6 judges each model: trend_14 holds with a pooled AUC >= 0.60 and an AUC above 0.5 in at least 4 of the 6
  segments; fwd_vol_7 holds with a pooled R2 gain above 0 and a gain above 0 in at least 4 segments. fwd_vol_14 and
  the direction model are recorded only. A model that holds is a component, not a trading permission.
- Writes ``result.json`` and ``README.md`` into a new directory: a look is evaluated once; nothing is selected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from lab import daily_predict as dp
from lab import forward
from lab import panel_study as ps

H, DAY = ps.H, ps.DAY
LOOKS_IN_PROTOCOL = 6
HORIZON = {"trend_14": 14, "fwd_vol_7": 7, "fwd_vol_14": 14, "ret_14_scaled": 14}
JUDGED = {"trend_14": 0.60, "fwd_vol_7": 0.0}        # the pooled bar; a segment's bar is 0.5 (AUC) or 0 (gain)
SEGMENTS_NEEDED = 4


def segments(ends: Sequence[int], days: int) -> list[tuple[int, int]]:
    """Each look's new decision days for a label of ``days`` days, the first from the forward start."""
    out, low = [], forward.FORWARD_START
    for end in ends:
        high = end - days * DAY
        out.append((low, high))
        low = high
    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(y) and 0 < y.mean() < 1 else None


def _r2(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))


def score(target: str, y: np.ndarray, p: np.ndarray, naive: np.ndarray | None, t: np.ndarray) -> float | None:
    """The development measure of a target on these rows; None when it is not defined (one class, no IC day)."""
    if not len(y):
        return None
    if target == "trend_14":
        return _auc(y, p)
    if target == "ret_14_scaled":
        ic = ps.daily_rank_ic(t, p, y).dropna()
        return float(ic.mean()) if len(ic) else None
    if naive is None or len(y) < 2 or np.all(y == y[0]):
        return None
    return _r2(y, p) - _r2(y, naive)


def verdict(target: str, measured: dict) -> bool | None:
    """Look 6's rule for a judged target; None for one recorded only."""
    if target not in JUDGED:
        return None
    segment_bar = 0.5 if target == "trend_14" else 0.0
    pooled = measured["all"]
    kept = sum(1 for s in measured["segments"] if s is not None and s > segment_bar)
    above = pooled is not None and (pooled >= JUDGED[target] if target == "trend_14" else pooled > JUDGED[target])
    return bool(above and kept >= SEGMENTS_NEEDED)


def _whole(snapshot: Path, symbols: Sequence[str], start: int, end: int) -> None:
    hours = pd.RangeIndex(start, end, H)
    for symbol in symbols:
        path = snapshot / f"{symbol}_1h.csv.gz"
        if not path.is_file():
            raise ValueError(f"{symbol} is not in {snapshot}: the data is not whole")
        times = pd.read_csv(path, usecols=["open_time"])["open_time"]
        missing = hours.difference(pd.Index(times))
        if len(missing):
            raise ValueError(f"{symbol}: {len(missing)} hours of the forward window missing: the data is not whole")


def look(*, end: str, data: Path, out: Path, tradfi: Sequence[str], spec_path: Path = dp.SPEC,
         frozen_dir: Path | None = None, looks: Path = forward.LOOKS, now_ms: int | None = None,
         code: tuple[Callable[[str], bytes], Callable[[str], bytes]] | None = None,
         min_universe: int = dp.MIN_UNIVERSE) -> dict[str, Any]:
    spec_path, data, out = Path(spec_path), Path(data), Path(out)
    frozen_dir = Path(frozen_dir) if frozen_dir is not None else spec_path.parent / "frozen"
    digest = forward.spec_digest(spec_path)
    spec = json.loads(spec_path.read_text())
    registered = forward.registered_look(spec["study"], digest, end, path=looks, now_ms=now_ms)
    if spec.get("kind") != "panel_forward" or spec.get("passed") is not False:
        raise ValueError(f"{spec_path} is not the frozen panel spec")
    ends = sorted(r["end"] for r in forward.load_looks(looks) if (r["study"], r["spec_sha256"]) == (spec["study"],
                                                                                                    digest))
    if len(ends) != LOOKS_IN_PROTOCOL:
        raise ValueError(f"{len(ends)} looks are registered for the panel spec; the protocol names {LOOKS_IN_PROTOCOL}")
    number = ends.index(end) + 1
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: a look is evaluated once")
    current, frozen = code or ((lambda p: (dp.ROOT / p).read_bytes()), dp._git_show(spec["code_commit"]))
    changed = dp.code_changes(dp.FROZEN_CODE, current, frozen) + dp.check_models(spec, frozen_dir)
    if changed:
        raise ValueError(f"the frozen code or models changed: {', '.join(changed)}")
    end_ms = forward.day_ms(end)
    _whole(data, dp.COINS, forward.FORWARD_START, end_ms)
    run = dp._panel_run()
    new_tradfi = sorted(set(tradfi) - set(run.FROZEN_EXCLUDE))
    run.SNAP, run.LIMIT = data, end_ms
    run.EXCLUDE = frozenset(run.FROZEN_EXCLUDE) | frozenset(tradfi)
    times = np.arange(forward.FORWARD_START, end_ms - min(HORIZON.values()) * DAY, DAY, dtype=np.int64)
    panel, y, info = run.build_panel(times)
    if not (panel["symbol"].to_numpy() == y["symbol"].to_numpy()).all():
        raise ValueError("the panel's rows and labels are not aligned")
    per_day = panel.groupby(level=0).size().reindex(times).fillna(0).astype(int)
    if int(per_day.min()) < min_universe:
        raise ValueError(f"a decision day has {int(per_day.min())} contracts (< {min_universe}): the data is not whole")
    panel["log_rv_30"] = np.log(panel["rv_30"])
    x = panel[list(spec["features"])].astype(np.float32)
    paths = {name: frozen_dir / next(f for f in files if not f.endswith(".meta.json"))
             for name, files in spec["models"].items()}
    pred = {name: ps.predict_frozen(ps.load_frozen(path), x) for name, path in paths.items()}
    t = panel.index.to_numpy(dtype=np.int64)
    coins = np.isin(panel["symbol"].to_numpy(), dp.COINS)
    models = {}
    for name in sorted(pred):
        target, kind = name.split(" ")
        if kind == "naive":
            continue
        days = HORIZON[target]
        yv = y[target].to_numpy(float)
        naive = pred.get(f"{target} naive")
        ok = np.isfinite(yv) & (t < end_ms - days * DAY)

        def on(mask: np.ndarray) -> float | None:
            return score(target, yv[mask], pred[name][mask], None if naive is None else naive[mask], t[mask])
        measured = {"target": target, "rows": int(ok.sum()), "all": on(ok), "btc_eth": on(ok & coins),
                    "segments": [on(ok & (t >= a) & (t < b)) for a, b in segments([forward.day_ms(e) for e in
                                                                                    ends[:number]], days)]}
        if number == LOOKS_IN_PROTOCOL:
            measured["holds"] = verdict(target, measured)
        models[name] = measured
    universe = [{"days": int(((per_day.index >= a) & (per_day.index < b)).sum()),
                 "mean": float(per_day[(per_day.index >= a) & (per_day.index < b)].mean()),
                 "min": int(per_day[(per_day.index >= a) & (per_day.index < b)].min())}
                for a, b in segments([forward.day_ms(e) for e in ends[:number]], min(HORIZON.values()))]
    result = {"study": spec["study"], "end": end, "look": number, "of": LOOKS_IN_PROTOCOL,
              "registered": registered["registered"], "spec_sha256": digest, "data": str(data),
              "decision_days": [forward.FORWARD_START, int(times[-1]) if len(times) else None],
              "judged": number == LOOKS_IN_PROTOCOL, "models": models, "universe": universe,
              "tradfi_left_out_new": new_tradfi, "panel": info}
    result = json.loads(json.dumps(result, default=float))
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out / "README.md").write_text(_notes(result))
    return result


def _fmt(v: float | None) -> str:
    return "无" if v is None else f"{v:.3f}"


def _notes(r: dict[str, Any]) -> str:
    stage = "判定（第 6 次）" if r["judged"] else "只报告，不判定、不调任何东西"
    lines = [f"# 面板模型前向查看：第 {r['look']}/{r['of']} 次，到 {r['end']}（不含）", "",
             "数字是【实际运行结果】，来自同目录的 `result.json`。**模型成立也不是交易许可。**", "",
             f"- 这一次：{stage}。度量：趋势用 AUC，波动用 R2 减朴素预测的 R2，方向用每天截面 IC 的均值。",
             "- 各模型（合并 / 各段 / 只看 BTC 和 ETH）："]
    for name, m in sorted(r["models"].items()):
        held = "" if "holds" not in m or m["holds"] is None else ("，**成立**" if m["holds"] else "，**不成立**")
        lines.append(f"  - {name}：合并 {_fmt(m['all'])}；各段 {' / '.join(_fmt(s) for s in m['segments'])}；"
                     f"BTC/ETH {_fmt(m['btc_eth'])}；{m['rows']} 行{held}")
    sizes = "；".join(f"{u['mean']:.1f} / {u['min']}" for u in r["universe"])
    lines.append(f"- 每段的宇宙人数（日均 / 最少）：{sizes}")
    if r["tradfi_left_out_new"]:
        lines.append(f"- 新排除的 TradFi 合约：{', '.join(r['tradfi_left_out_new'])}")
    lines += ["- 半年的数据只够看排序能力还在不在，证明不了能赚钱。", ""]
    return "\n".join(lines)


def _tradfi_now() -> list[str]:
    from data.downloads import perps_feed
    return perps_feed.contracts(json.loads(perps_feed.http_get(perps_feed.EXCHANGE_INFO)))[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = look(end=args.end, data=args.data, out=args.out, tradfi=_tradfi_now())
    print(json.dumps({"look": result["look"], "models": {k: v["all"] for k, v in result["models"].items()}}))


if __name__ == "__main__":
    main()


__all__ = ["look", "score", "segments", "verdict"]
