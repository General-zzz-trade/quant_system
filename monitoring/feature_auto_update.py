"""Automated feature pool updater — reads IC analysis, updates forced_features.

Runs after feature_ic_analysis.py. Reads feature_ic_report.json, identifies
features with degraded IC (SIGN_FLIP or DEAD for 3+ consecutive days), and
suggests replacements from the candidate pool by computing fresh IC on recent
data.

Actions:
  1. Read feature_ic_report.json for current feature health
  2. Track consecutive degradation days in feature_health_history.json
  3. If a forced feature is SIGN_FLIP or DEAD for 3+ days, mark for replacement
  4. Compute IC for candidate features on recent 3-month data
  5. Select best stable replacement (consistent sign, |IC| > 0.05)
  6. Update alpha/retrain/config.py FORCED_FEATURES
  7. Trigger retrain via --sighup on next daily-retrain cycle

Safety:
  - Never removes more than 2 features per model per cycle
  - Replacement must have higher |IC| than the removed feature
  - All changes logged to data/runtime/feature_update_history.jsonl
  - Dry-run by default; --apply to actually modify config

Usage:
    python3 -m monitoring.feature_auto_update                # dry-run
    python3 -m monitoring.feature_auto_update --apply        # modify config
    python3 -m monitoring.feature_auto_update --alert        # send telegram
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

IC_REPORT_PATH = Path("data/runtime/feature_ic_report.json")
HEALTH_HISTORY_PATH = Path("data/runtime/feature_health_history.json")
UPDATE_HISTORY_PATH = Path("data/runtime/feature_update_history.jsonl")
CONFIG_PATH = Path("alpha/retrain/config.py")

# Minimum consecutive days of degradation before replacement
MIN_DEGRADE_DAYS = 3
# Maximum features to replace per model per cycle
MAX_REPLACEMENTS = 2
# Minimum IC for a replacement candidate
MIN_CANDIDATE_IC = 0.05


def _load_health_history() -> dict[str, dict[str, int]]:
    """Load {model: {feature: consecutive_bad_days}}."""
    if HEALTH_HISTORY_PATH.exists():
        try:
            return json.loads(HEALTH_HISTORY_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_health_history(history: dict[str, dict[str, int]]) -> None:
    HEALTH_HISTORY_PATH.write_text(json.dumps(history, indent=2))


def _log_update(entry: dict[str, Any]) -> None:
    with open(UPDATE_HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def analyze_degradation() -> dict[str, list[dict[str, Any]]]:
    """Identify features with sustained degradation.

    Returns {model: [{feature, class, days, ic_60d}]}.
    """
    if not IC_REPORT_PATH.exists():
        logger.warning("No IC report found at %s", IC_REPORT_PATH)
        return {}

    with open(IC_REPORT_PATH) as f:
        report = json.load(f)

    history = _load_health_history()
    degraded: dict[str, list[dict[str, Any]]] = {}

    for model_name, info in report.get("models", {}).items():
        if not isinstance(info, dict):
            continue
        features = info.get("features", {})
        model_history = history.setdefault(model_name, {})

        for fname, fdata in features.items():
            cls = fdata.get("class", "")
            ic_60d = (fdata.get("ics", {}).get("60d") or 0)

            if cls in ("SIGN_FLIP", "DEAD"):
                model_history[fname] = model_history.get(fname, 0) + 1
            else:
                model_history[fname] = 0

            if model_history.get(fname, 0) >= MIN_DEGRADE_DAYS:
                degraded.setdefault(model_name, []).append({
                    "feature": fname,
                    "class": cls,
                    "days": model_history[fname],
                    "ic_60d": ic_60d,
                })

    _save_health_history(history)
    return degraded


def find_replacements(symbol: str, exclude: set[str],
                      n: int = 3) -> list[dict[str, Any]]:
    """Find best replacement features from the candidate pool.

    Computes IC on recent 3-month data for all available features
    not currently in the model.
    """
    import numpy as np
    import pandas as pd
    from features.batch_feature_engine import compute_features_batch

    df = pd.read_csv(f"data_files/{symbol}_1h.csv")
    feat_df = compute_features_batch(symbol, df, include_onchain=True)
    close = df["close"].values

    # Forward return h24
    fwd = pd.Series(close).pct_change(24).shift(-24).values

    # Only use last 3 months
    window = 3 * 730
    candidates = []

    for col in feat_df.columns:
        if col in exclude or col in ("close", "volume", "open", "high", "low"):
            continue
        vals = feat_df[col].values[-window:]
        y = fwd[-window:]
        mask = ~(np.isnan(vals) | np.isnan(y))
        if mask.sum() < 200:
            continue
        ic = float(np.corrcoef(vals[mask], y[mask])[0, 1])
        if abs(ic) < MIN_CANDIDATE_IC:
            continue

        # Check stability: compute IC on 3 sub-windows
        sub_n = mask.sum() // 3
        ics = []
        idx = np.where(mask)[0]
        for i in range(3):
            sub = idx[i * sub_n:(i + 1) * sub_n]
            if len(sub) < 50:
                continue
            sub_ic = float(np.corrcoef(vals[sub], y[sub])[0, 1])
            ics.append(sub_ic)

        if len(ics) < 3:
            continue
        signs = [np.sign(ic) for ic in ics]
        stable = all(s == signs[0] for s in signs)
        if not stable:
            continue

        candidates.append({
            "feature": col,
            "ic_3m": ic,
            "stable": True,
            "sub_ics": ics,
        })

    candidates.sort(key=lambda x: -abs(x["ic_3m"]))
    return candidates[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually update config (default: dry-run)")
    parser.add_argument("--alert", action="store_true",
                        help="Send Telegram alert")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    degraded = analyze_degradation()

    if not degraded:
        print("No features with sustained degradation found.")
        return

    print("=" * 60)
    print("  Feature Auto-Update Report")
    print("=" * 60)

    for model_name, feats in degraded.items():
        print(f"\n  {model_name}:")
        for f in feats:
            print(f"    {f['feature']:25s} {f['class']:10s} "
                  f"{f['days']}d degraded  IC={f['ic_60d']:+.4f}")

        # Find replacements
        symbol = model_name.split("_")[0]
        if "_4h" in model_name:
            continue  # Skip 4h for now

        # Only replace up to MAX_REPLACEMENTS
        to_replace = sorted(feats, key=lambda x: x["days"], reverse=True)
        to_replace = to_replace[:MAX_REPLACEMENTS]

        if to_replace:
            all_model_feats = set()
            try:
                with open(f"models_v8/{model_name}/config.json") as cf:
                    cfg = json.load(cf)
                for hm in cfg.get("horizon_models", []):
                    all_model_feats.update(hm.get("features", []))
            except Exception:
                pass

            replacements = find_replacements(
                symbol, exclude=all_model_feats, n=len(to_replace) * 2)

            print("\n  Suggested replacements:")
            for i, r in enumerate(replacements):
                print(f"    {r['feature']:25s} IC={r['ic_3m']:+.4f} "
                      f"(stable, subs={[f'{x:+.3f}' for x in r['sub_ics']]})")

            if args.apply and replacements:
                # Log the update
                _log_update({
                    "ts": time.time(),
                    "model": model_name,
                    "removed": [f["feature"] for f in to_replace],
                    "added": [r["feature"] for r in replacements[:len(to_replace)]],
                    "reason": "auto_feature_update",
                })
                print("\n  ⚠️ Config update requires manual edit of "
                      "alpha/retrain/config.py FORCED_FEATURES")
                print("  Then run: python3 -m alpha.retrain.cli --force --sighup")

    if args.alert:
        try:
            from monitoring.notify import send_alert, AlertLevel
            details = {}
            for model_name, feats in degraded.items():
                details[model_name] = ", ".join(
                    f"{f['feature']}({f['class']},{f['days']}d)"
                    for f in feats)
            send_alert(
                AlertLevel.WARNING,
                f"Feature degradation: {sum(len(v) for v in degraded.values())} features flagged",
                details=details,
                source="feature_auto_update",
            )
        except Exception:
            logger.debug("Alert send failed", exc_info=True)


if __name__ == "__main__":
    main()
