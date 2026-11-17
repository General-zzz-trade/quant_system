"""Report-only items of the panel study, after its results (nothing here changes a verdict or a setting):
1. registered in README.md but missing from results.json: the same measures on the BTC and ETH rows;
2. diagnostics written after the results: the trend target's AUC within each day (which contracts, not when) against
   its pooled AUC, and the LightGBM models' most used features (split gain).
The models are refit exactly as run.py fits them; no null is rerun."""
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

OUT = HERE / "report.json"


def within_day_auc(t: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    scores = []
    for day in np.unique(t):
        m = t == day
        if 0 < y[m].mean() < 1:
            scores.append(roc_auc_score(y[m], p[m]))
    return float(np.mean(scores))


def main():
    times = np.arange(R.EVAL_START + R.DAY, R.LIMIT + 1, R.DAY, dtype=np.int64)
    panel, y, info = R.build_panel(times)
    panel["log_rv_30"] = np.log(panel["rv_30"])
    x = panel[list(R.ps.PANEL_FEATURES) + ["log_rv_30"]].astype(np.float32)
    t, sym = panel.index.to_numpy(dtype=np.int64), panel["symbol"].to_numpy()
    out = {"info": info, "btc_eth": {}, "within_day_auc": {}, "top_features": {}}
    for name in ("trend_14", "fwd_vol_7", "fwd_vol_14"):
        kind, end, _ = R.TARGETS[name]
        target = y[name].to_numpy(float)
        label_end = y[end].fillna(pd.Series(t + 14 * R.DAY, index=y.index)).to_numpy(dtype=np.int64)
        keep = np.isfinite(target)
        for model in R.MODELS[kind]:
            naive = "log_rv_30" if name.startswith("fwd_vol") else None
            ev = R.ps.evaluate(x, target, t, label_end, R.YEARS, kind=kind, model=model, naive=naive)
            pred = ev["predictions"]
            tk, yk, sk = t[keep], target[keep], sym[keep]
            test = np.isfinite(pred)
            entry = {}
            for coin in R.ps.ALWAYS:
                m = test & (sk == coin)
                if kind == "binary":
                    entry[coin] = {"auc": float(roc_auc_score(yk[m], pred[m])), "rows": int(m.sum()),
                                   "base_rate": float(yk[m].mean())}
                else:
                    entry[coin] = {"corr_pred_outcome": float(np.corrcoef(pred[m], yk[m])[0, 1]), "rows": int(m.sum())}
            out["btc_eth"][f"{name} {model}"] = entry
            if kind == "binary":
                out["within_day_auc"][model] = {"pooled": ev["auc"],
                                                "within_day_mean": within_day_auc(tk[test], yk[test], pred[test])}
            print(name, model, json.dumps(entry), flush=True)
    from lightgbm import LGBMClassifier
    target = y["trend_14"].to_numpy(float)
    keep = np.isfinite(target)
    params = {**R.ps.st.MODELS["lgbm"], "num_leaves": 31, "min_child_samples": 200, "n_estimators": 300,
              "importance_type": "gain"}
    xs, filled = x.iloc[np.flatnonzero(keep)], None
    filled = R.ps.st.fill(xs, xs)[0]
    m = LGBMClassifier(**params).fit(filled.to_numpy(np.float32), target[keep], sample_weight=R.ps._weights(t[keep]))
    gain = pd.Series(m.feature_importances_, index=x.columns).sort_values(ascending=False)
    out["top_features"]["trend_14 lgbm (fit on every development row, for reading only)"] = {
        k: round(float(v / gain.sum()), 4) for k, v in gain.head(15).items()}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("within_day_auc", "top_features")}, indent=1))


if __name__ == "__main__":
    main()
