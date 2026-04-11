#!/usr/bin/env python3
"""Feature ablation — rank the alpha contribution of each feature.

For each candidate feature (individually or by group):

  1. Load the production model's config + feature list
  2. Compute baseline IC on OOS test set (canonical ``compute_ic``)
  3. Drop that feature (zero it out — keeps column count stable)
  4. Recompute Ridge prediction on OOS and measure IC delta
  5. Rank features by ``|Δ IC|``

Features with ``Δ IC ≈ 0`` or *positive* (i.e. dropping them *improves*
IC) are candidates for removal on the next retrain — they're either
noise or redundant with another feature.  The expectation is 10-20% of
the 141-feature pool can be pruned without hurting live performance,
reducing training time and overfit surface.

Why Ridge only
--------------
LGBM ablation via retrain is too slow (~90s per feature × 141 features
= ~3.5h per symbol).  Ridge ablation is trivial: zero a column in X,
re-multiply W·X to get the new prediction, compute IC, done.  Since
Ridge and LGBM share the same feature selection pipeline and their
signals are blended IC-proportionally, a feature that Ridge can't use
*at all* is likely also a weak contributor to LGBM.

For a more expensive but more accurate ablation, use ``--retrain`` to
force a full LGBM retrain per ablation (slower by 50×).

Usage
-----
    # Rank all 141 features by Ridge |Δ IC| (fast, ~2 min)
    python3 scripts/feature_ablation.py --symbol BTCUSDT

    # Single feature
    python3 scripts/feature_ablation.py --symbol ETHUSDT --feature dvol_zscore

    # Ablate a whole group
    python3 scripts/feature_ablation.py --symbol BTCUSDT --group pcr

    # Output JSON for tooling
    python3 scripts/feature_ablation.py --symbol BTCUSDT --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/quant_system")

from shared.ic_metrics import compute_ic


def _load_model_and_data(symbol: str, model_dir: str) -> dict:
    """Load the production Ridge coefficients + OOS feature matrix.

    Uses the same pipeline as ``alpha.model_loader_prod`` for Ridge
    predictions, then builds the OOS chunk from the batch feature engine.
    """
    import pandas as pd
    from features.batch_feature_engine import compute_features_batch

    model_path = Path(f"models_v8/{model_dir}")
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = json.loads(cfg_path.read_text())
    ridge_features = cfg.get("ridge_features") or cfg.get("features", [])

    # Load Ridge weights + intercept from the JSON dump
    ridge_json_candidates = [
        model_path / "ridge_h24.json",
        model_path / "ridge_h12.json",
    ]
    ridge_w = None
    ridge_b = 0.0
    ridge_features_override: list[str] | None = None
    for rjp in ridge_json_candidates:
        if rjp.exists():
            rj = json.loads(rjp.read_text())
            # Rust-trained format: coefficients / intercept / features
            coef_key = "coefficients" if "coefficients" in rj else (
                "coef" if "coef" in rj else "weights"
            )
            ridge_w = np.array(rj.get(coef_key, []), dtype=np.float64)
            ridge_b = float(rj.get("intercept", 0.0))
            if "features" in rj and isinstance(rj["features"], list):
                ridge_features_override = rj["features"]
            break
    if ridge_features_override:
        ridge_features = ridge_features_override
    if ridge_w is None or len(ridge_w) == 0:
        raise RuntimeError(f"no ridge weights found under {model_path}")

    # Build OOS feature matrix
    data_path = Path(f"data_files/{symbol}_1h.csv")
    df = pd.read_csv(data_path).sort_values("open_time").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    feat_df = compute_features_batch(symbol, df)
    feature_names = [c for c in feat_df.columns
                     if c not in ("open_time", "close", "open", "high", "low", "volume")]

    # Align ridge feature ordering with the actual feature matrix
    col_idx = {n: i for i, n in enumerate(feature_names)}
    missing = [f for f in ridge_features if f not in col_idx]
    if missing:
        raise RuntimeError(f"Model references features not in batch: {missing[:5]}...")

    X_all = feat_df[ridge_features].values.astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0)

    # OOS = last 18 months (4320 bars × 12 ≈ 51840 at 1h).  Keep it simple:
    # use the final 30% of bars as OOS.
    n = len(X_all)
    oos_start = int(n * 0.70)
    horizon = int(cfg.get("horizon", 24))
    y = np.full(n, np.nan)
    for i in range(n - horizon):
        y[i] = (closes[i + horizon] - closes[i]) / closes[i]

    mask = ~np.isnan(y[oos_start:])
    X_oos = X_all[oos_start:][mask]
    y_oos = y[oos_start:][mask]

    return {
        "symbol": symbol,
        "model_dir": str(model_path),
        "ridge_features": ridge_features,
        "ridge_w": ridge_w,
        "ridge_b": ridge_b,
        "X_oos": X_oos,
        "y_oos": y_oos,
    }


def _ridge_pred(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def ablate(
    data: dict,
    target_features: list[str] | None = None,
    group_prefix: str | None = None,
) -> list[dict]:
    """Run per-feature ablation and return a ranked list."""
    feats = data["ridge_features"]
    w = data["ridge_w"]
    b = data["ridge_b"]
    X = data["X_oos"]
    y = data["y_oos"]

    baseline_pred = _ridge_pred(X, w, b)
    baseline_ic = compute_ic(baseline_pred, y)

    rows: list[dict] = []

    # Resolve which features to ablate
    if group_prefix:
        to_ablate = [f for f in feats if f.startswith(group_prefix)]
        if not to_ablate:
            return [{"error": f"no features with prefix '{group_prefix}'"}]
        # Group ablation: zero them all at once
        X_abl = X.copy()
        for f in to_ablate:
            idx = feats.index(f)
            X_abl[:, idx] = 0.0
        abl_pred = _ridge_pred(X_abl, w, b)
        abl_ic = compute_ic(abl_pred, y)
        rows.append({
            "feature": f"GROUP:{group_prefix}*",
            "n_features": len(to_ablate),
            "baseline_ic": round(baseline_ic, 4),
            "ablated_ic": round(abl_ic, 4),
            "delta_ic": round(abl_ic - baseline_ic, 4),
        })
        return rows

    if target_features:
        candidates = [f for f in target_features if f in feats]
    else:
        candidates = feats[:]

    for f in candidates:
        idx = feats.index(f)
        X_abl = X.copy()
        X_abl[:, idx] = 0.0
        abl_pred = _ridge_pred(X_abl, w, b)
        abl_ic = compute_ic(abl_pred, y)
        delta = abl_ic - baseline_ic
        rows.append({
            "feature": f,
            "coef": round(float(w[idx]), 5),
            "baseline_ic": round(baseline_ic, 4),
            "ablated_ic": round(abl_ic, 4),
            "delta_ic": round(delta, 4),
        })

    # Sort by delta_ic ascending (most negative = dropping this hurts most)
    rows.sort(key=lambda r: r["delta_ic"])
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    p.add_argument("--model-dir", default=None,
                   help="Model subdirectory (default: {symbol}_gate_v2)")
    p.add_argument("--feature", default=None,
                   help="Ablate a single feature")
    p.add_argument("--group", default=None,
                   help="Ablate all features with this prefix (eg. 'pcr', 'l2_tvl')")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    model_dir = args.model_dir or f"{args.symbol}_gate_v2"
    print(f"Loading {args.symbol} from {model_dir}...")
    t0 = time.time()
    data = _load_model_and_data(args.symbol, model_dir)
    print(f"  OOS bars: {len(data['X_oos'])}  "
          f"features: {len(data['ridge_features'])}  "
          f"load={time.time() - t0:.1f}s")

    target = [args.feature] if args.feature else None
    rows = ablate(data, target_features=target, group_prefix=args.group)

    if args.json:
        print(json.dumps({
            "symbol": args.symbol,
            "model_dir": model_dir,
            "results": rows,
        }, indent=2))
        return 0

    if rows and rows[0].get("error"):
        print(rows[0]["error"])
        return 1

    print(f"\n{'Feature':<35} {'coef':>9} {'base_ic':>9} {'abl_ic':>9} {'Δ IC':>9}")
    print("-" * 78)
    # Worst 15 (dropping hurts most) + best 15 (dropping helps most)
    k = min(15, len(rows) // 2)
    print("── hurts most when dropped (keep) ──")
    for r in rows[:k]:
        print(f"{r['feature']:<35} "
              f"{r.get('coef', 0):>+9.4f} "
              f"{r['baseline_ic']:>+9.4f} "
              f"{r['ablated_ic']:>+9.4f} "
              f"{r['delta_ic']:>+9.4f}")
    if len(rows) > 2 * k:
        print(f"...  ({len(rows) - 2 * k} middle features elided)")
    print("── helps most when dropped (consider removing) ──")
    for r in rows[-k:]:
        print(f"{r['feature']:<35} "
              f"{r.get('coef', 0):>+9.4f} "
              f"{r['baseline_ic']:>+9.4f} "
              f"{r['ablated_ic']:>+9.4f} "
              f"{r['delta_ic']:>+9.4f}")

    positive_deltas = [r for r in rows if r["delta_ic"] > 0]
    print(f"\n{len(positive_deltas)} features have positive Δ IC "
          f"(dropping improves IC — pruning candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
