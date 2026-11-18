"""Report-only table for the daily BTC/ETH report (owner 2026-11-18), written after the panel study's results; it
changes no verdict and no frozen model.

For BTCUSDT and ETHUSDT: the trend_14 logistic model's yearly walk-forward predictions on development data
(2022-2025, exactly as run.py evaluates them), split into thirds of that coin's own predictions; the share of days
whose 14-day trend label was 1 in each third, and on average. The frozen model is fit on more rows than any
walk-forward year, so applying these edges to its probabilities is an approximation, and the report says so.
Writes calibration.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402

OUT = HERE / "calibration.json"


def main() -> None:
    times = np.arange(R.EVAL_START + R.DAY, R.LIMIT + 1, R.DAY, dtype=np.int64)
    panel, y, _ = R.build_panel(times)
    panel["log_rv_30"] = np.log(panel["rv_30"])
    x = panel[list(R.ps.PANEL_FEATURES) + ["log_rv_30"]].astype(np.float32)
    t, sym = panel.index.to_numpy(dtype=np.int64), panel["symbol"].to_numpy()
    target = y["trend_14"].to_numpy(float)
    label_end = y["end_14"].fillna(pd.Series(t + 14 * R.DAY, index=y.index)).to_numpy(dtype=np.int64)
    ev = R.ps.evaluate(x, target, t, label_end, R.YEARS, kind="binary", model="logistic")
    keep = np.isfinite(target)
    pred, yk, sk = ev["predictions"], target[keep], sym[keep]
    out = {"model": "trend_14 logistic, yearly walk-forward, development data 2022-2025", "coins": {}}
    for coin in R.ps.ALWAYS:
        m = np.isfinite(pred) & (sk == coin)
        p, lab = pred[m], yk[m]
        edges = np.quantile(p, [1 / 3, 2 / 3])
        band = np.searchsorted(edges, p, side="right")
        out["coins"][coin] = {"edges": edges.tolist(), "rate": [float(lab[band == k].mean()) for k in range(3)],
                              "rows": [int((band == k).sum()) for k in range(3)], "base_rate": float(lab.mean()),
                              "auc": float(roc_auc_score(lab, p))}
        print(coin, json.dumps(out["coins"][coin]), flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
