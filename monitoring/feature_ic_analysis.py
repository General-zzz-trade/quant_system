"""Automated feature IC stability analysis — 30d/60d/90d/180d windows.

For each active model, computes per-feature IC in multiple lookback windows
and flags:
- SIGN_FLIP: feature IC changed sign across windows (unstable)
- DEAD: |IC| < 0.02 in all windows (no predictive power)
- STRONG: |IC| > 0.10 with consistent sign (reliable)

Output:
- JSON report at data/runtime/feature_ic_report.json
- Telegram alert if ≥3 features flagged DEAD or SIGN_FLIP in a production model

Usage:
    python3 -m monitoring.feature_ic_analysis
    python3 -m monitoring.feature_ic_analysis --alert       # send telegram summary
    python3 -m monitoring.feature_ic_analysis --model BTCUSDT_gate_v2
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Default production models to analyze
DEFAULT_MODELS = [
    ("BTCUSDT_gate_v2", "BTCUSDT", "1h"),
    ("ETHUSDT_gate_v2", "ETHUSDT", "1h"),
    ("BTCUSDT_4h", "BTCUSDT", "4h"),
]

WINDOWS_DAYS = [30, 60, 90, 180]

# Classification thresholds
IC_STRONG_THRESHOLD = 0.10
IC_DEAD_THRESHOLD = 0.02


def _load_features(symbol: str) -> pd.DataFrame:
    """Load and compute batch features for a symbol."""
    from features.batch_feature_engine import compute_features_batch
    df = pd.read_csv(f"data_files/{symbol}_1h.csv")
    return compute_features_batch(symbol, df, include_onchain=True, include_v11=True)


def _load_model_features(model_name: str) -> set[str]:
    """Extract feature set from a model's config."""
    cfg_path = Path(f"models_v8/{model_name}/config.json")
    if not cfg_path.exists():
        return set()
    cfg = json.loads(cfg_path.read_text())
    features: set[str] = set()
    for hm in cfg.get("horizon_models", []):
        if isinstance(hm, dict):
            features.update(hm.get("features", []))
    if not features and "features" in cfg:
        features.update(cfg["features"])
    return features


def _compute_feature_ic(
    feat_df: pd.DataFrame,
    feature: str,
    forward_bars: int,
    n_bars: int,
) -> float | None:
    """Compute spearman IC for a feature over recent n_bars."""
    if feature not in feat_df.columns:
        return None
    subset = feat_df.tail(n_bars + forward_bars).head(n_bars).copy()
    subset["fwd_ret"] = subset["close"].pct_change(forward_bars).shift(-forward_bars)
    valid = subset[[feature, "fwd_ret"]].dropna()
    if len(valid) < 50:
        return None
    try:
        return float(valid[feature].corr(valid["fwd_ret"], method="spearman"))
    except Exception:
        return None


def _classify(ics: list[float]) -> str:
    """Classify feature based on IC stability across windows."""
    if not ics:
        return "NO_DATA"
    # Check sign consistency
    signs = [1 if ic > 0 else (-1 if ic < 0 else 0) for ic in ics]
    nonzero = [s for s in signs if s != 0]
    consistent_sign = len(set(nonzero)) <= 1 if nonzero else False

    max_abs = max(abs(ic) for ic in ics)
    min_abs = min(abs(ic) for ic in ics)

    if max_abs < IC_DEAD_THRESHOLD:
        return "DEAD"
    if not consistent_sign and min_abs > 0.03:
        return "SIGN_FLIP"
    if max_abs >= IC_STRONG_THRESHOLD and consistent_sign:
        return "STRONG"
    if consistent_sign:
        return "STABLE"
    return "WEAK"


def analyze_model(model_name: str, symbol: str, interval: str) -> dict[str, Any]:
    """Analyze all features in a model across lookback windows."""
    logger.info("Analyzing %s (%s %s)...", model_name, symbol, interval)

    features = _load_model_features(model_name)
    if not features:
        return {"error": f"no features in {model_name}"}

    bars_per_day = 24 if interval == "1h" else (6 if interval == "4h" else 24)
    # Use 12-bar forward return (most models use h12)
    forward_bars = 12 if interval == "1h" else 12

    feat_df = _load_features(symbol)

    results: dict[str, Any] = {
        "model": model_name,
        "symbol": symbol,
        "interval": interval,
        "n_features": len(features),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "features": {},
        "summary": {"STRONG": 0, "STABLE": 0, "WEAK": 0, "SIGN_FLIP": 0, "DEAD": 0, "NO_DATA": 0},
    }

    for feat in sorted(features):
        ics: dict[str, float | None] = {}
        ic_values: list[float] = []
        for days in WINDOWS_DAYS:
            n_bars = days * bars_per_day
            ic = _compute_feature_ic(feat_df, feat, forward_bars, n_bars)
            ics[f"{days}d"] = round(ic, 4) if ic is not None else None
            if ic is not None:
                ic_values.append(ic)

        classification = _classify(ic_values)
        results["features"][feat] = {"ics": ics, "class": classification}
        results["summary"][classification] += 1

    return results


def build_report(models: list[tuple[str, str, str]] | None = None) -> dict[str, Any]:
    """Build full IC analysis report for all models."""
    if models is None:
        models = DEFAULT_MODELS

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }
    for model_name, symbol, interval in models:
        try:
            report["models"][model_name] = analyze_model(model_name, symbol, interval)
        except Exception as e:
            logger.exception("Failed to analyze %s", model_name)
            report["models"][model_name] = {"error": str(e)}

    return report


def send_alert_summary(report: dict[str, Any]) -> None:
    """Send Telegram alert summarizing any models with flagged features."""
    from monitoring.notify import send_alert, AlertLevel

    flagged_models: list[str] = []
    details: dict[str, str] = {}

    for model_name, result in report["models"].items():
        if "error" in result:
            continue
        summary = result.get("summary", {})
        bad = summary.get("SIGN_FLIP", 0) + summary.get("DEAD", 0)
        total = result.get("n_features", 0)
        strong = summary.get("STRONG", 0)
        if bad >= 3:
            flagged_models.append(model_name)
        details[model_name] = f"{total} feats, {strong} strong, {bad} flagged (SIGN_FLIP+DEAD)"

    if flagged_models:
        level = AlertLevel.WARNING
        title = f"Feature IC analysis: {len(flagged_models)} models need attention"
        details["flagged"] = ", ".join(flagged_models)
    else:
        level = AlertLevel.INFO
        title = "Feature IC analysis: all models healthy"

    send_alert(level, title, details=details, source="feature_ic_analysis")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Only analyze this specific model")
    parser.add_argument("--alert", action="store_true", help="Send telegram alert")
    parser.add_argument("--output", default="data/runtime/feature_ic_report.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.model:
        # Filter to specific model
        models = [(m, s, i) for m, s, i in DEFAULT_MODELS if m == args.model]
    else:
        models = DEFAULT_MODELS

    report = build_report(models)

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report saved to %s", output_path)

    # Print summary
    for model_name, result in report["models"].items():
        if "error" in result:
            print(f"\n{model_name}: ERROR - {result['error']}")
            continue
        print(f"\n=== {model_name} ({result['n_features']} features) ===")
        summary = result.get("summary", {})
        for cls in ["STRONG", "STABLE", "WEAK", "SIGN_FLIP", "DEAD", "NO_DATA"]:
            count = summary.get(cls, 0)
            if count > 0:
                print(f"  {cls:<10} {count}")
        # Show flagged features
        flagged = [f for f, d in result["features"].items() if d["class"] in ("SIGN_FLIP", "DEAD")]
        if flagged:
            print(f"  Flagged: {flagged}")

    if args.alert:
        send_alert_summary(report)


if __name__ == "__main__":
    main()
