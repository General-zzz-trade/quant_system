"""Small, transparent Ridge reference models with auditable training boundaries.

These are new research models, not a reconstruction of the unversioned live
model files. Features use the existing incremental production computer, so the
live runner can load the same artifact and compute the same inputs.

A model ``spec`` is predeclared: feature set, label horizon (bars) and Ridge
alpha. Labels are forward log returns over ``horizon`` bars and every label
must end strictly before the training cutoff.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from data.quality.causal_bars import aggregate_closed, fingerprint
from features.enriched_computer import EnrichedFeatureComputer

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "ohlcv": ("ret_1", "ret_3", "ret_6", "ret_12", "vol_20"),
    # Binance-native derivatives data only; every input is also available live.
    "derivatives": ("ret_1", "ret_3", "ret_6", "ret_12", "vol_20",
                    "funding_rate", "funding_zscore_24", "oi_change_pct", "oi_change_24",
                    "taker_buy_ratio_ma10", "taker_imbalance", "ls_ratio_zscore_24", "cvd_20"),
}
FEATURES = FEATURE_SETS["ohlcv"]  # backward-compatible default
DEFAULT_SPEC = {"feature_set": "ohlcv", "horizon": 1, "alpha": 1.0}


def normalize_spec(spec: dict | None) -> dict:
    spec = {**DEFAULT_SPEC, **(spec or {})}
    if spec["feature_set"] not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {spec['feature_set']!r}")
    horizon, alpha = int(spec["horizon"]), float(spec["alpha"])
    if horizon < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("horizon must be >= 1 and ridge alpha positive")
    return {"feature_set": spec["feature_set"], "horizon": horizon, "alpha": alpha}


def compute_bar(computer, bar, symbol="BTCUSDT", context=None):
    """One closed bar -> incremental features. ``context`` supplies causal
    derivatives inputs (funding, OI, ratios) as of the bar's close."""
    ts = datetime.fromtimestamp(bar.available_at / 1000, timezone.utc)
    extras = getattr(bar, "extras", None) or {}
    derived = context.at(bar.available_at) if context is not None else {}
    return computer.on_bar(symbol, open_=float(bar.open), high=float(bar.high), low=float(bar.low),
                           close=float(bar.close), volume=float(bar.volume), hour=ts.hour, dow=ts.weekday(),
                           trades=float(extras.get("trades", 0.0)),
                           taker_buy_volume=float(extras.get("taker_buy_volume", 0.0)),
                           quote_volume=float(extras.get("quote_volume", 0.0)),
                           taker_buy_quote_volume=float(extras.get("taker_buy_quote_volume", 0.0)),
                           **derived)


def feature_rows(bars, symbol="BTCUSDT", context=None):
    computer = EnrichedFeatureComputer()
    return [compute_bar(computer, bar, symbol, context) for bar in bars]


def vector(features, names=FEATURES):
    values = [features.get(k) for k in names]
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return None
    return np.asarray(values, dtype=float)


def design(bars, cutoff, *, symbol="BTCUSDT", context=None, spec=None, rows=None):
    """Feature matrix and forward-return labels; every label ends before ``cutoff``."""
    spec = normalize_spec(spec)
    names, h = FEATURE_SETS[spec["feature_set"]], spec["horizon"]
    rows = feature_rows(bars, symbol, context) if rows is None else rows
    samples = []
    for i in range(50, len(bars) - h):
        if bars[i + h].available_at >= cutoff:
            break
        x = vector(rows[i], names)
        if x is not None:
            samples.append((x, math.log(float(bars[i + h].close / bars[i].close)), bars[i + h].available_at))
    return samples


def fit_samples(samples, spec, *, training_sha256):
    spec = normalize_spec(spec)
    names = FEATURE_SETS[spec["feature_set"]]
    if len(samples) < 40:
        raise ValueError("need at least 40 eligible training labels after 50 warmup bars per timeframe")
    x = np.stack([s[0] for s in samples])
    y = np.asarray([s[1] for s in samples])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    normalized = (x - mean) / scale
    intercept = float(y.mean())
    weights = np.linalg.solve(normalized.T @ normalized + spec["alpha"] * np.eye(len(names)),
                              normalized.T @ (y - intercept))
    return {"features": list(names), "mean": mean.tolist(), "scale": scale.tolist(),
            "weights": weights.tolist(), "intercept": intercept, "alpha": spec["alpha"],
            "horizon": spec["horizon"], "feature_set": spec["feature_set"],
            "samples": len(samples), "label_end_ms": max(s[2] for s in samples),
            "training_sha256": training_sha256}


def fit_model(bars, cutoff, alpha=1.0, *, symbol="BTCUSDT", context=None, spec=None):
    spec = normalize_spec({**(spec or {}), **({"alpha": alpha} if spec is None else {})})
    samples = design(bars, cutoff, symbol=symbol, context=context, spec=spec)
    return fit_samples(samples, spec, training_sha256=fingerprint(
        [b.record() for b in bars if b.available_at < cutoff]))


def train_bundle(bars, cutoff, *, symbol="BTCUSDT", context=None, spec=None):
    # Cut raw input before feature computation: a trainer cannot inspect held-out bars.
    spec = normalize_spec(spec)
    history = [b for b in bars if b.available_at < cutoff]
    models = {"1h": fit_model(history, cutoff, symbol=symbol, context=context, spec=spec),
              "4h": fit_model(aggregate_closed(history), cutoff, symbol=symbol, context=context, spec=spec)}
    bundle = {"schema": 1, "kind": "incremental_ohlcv_ridge_reference", "cutoff_ms": cutoff,
              "models": models, "hyperparameter_candidates": 1, "online_updates": False}
    if symbol != "BTCUSDT" or spec != DEFAULT_SPEC:
        bundle.update(schema=2, kind="incremental_ridge_reference", symbol=symbol, spec=spec)
    bundle["sha256"] = fingerprint(bundle)
    return bundle


def validate_bundle(bundle, evaluation_start, *, symbol=None):
    raw = {k: v for k, v in bundle.items() if k != "sha256"}
    if bundle.get("schema") not in (1, 2) or fingerprint(raw) != bundle.get("sha256"):
        raise ValueError("model artifact hash/schema mismatch")
    spec = normalize_spec(bundle.get("spec")) if bundle["schema"] == 2 else DEFAULT_SPEC
    if symbol is not None and bundle.get("symbol", "BTCUSDT") != symbol:
        raise ValueError("model was trained for a different symbol")
    if bundle["cutoff_ms"] > evaluation_start:
        raise ValueError("model trained after evaluation began")
    names = list(FEATURE_SETS[spec["feature_set"]])
    for interval in ("1h", "4h"):
        model = bundle["models"][interval]
        if model["label_end_ms"] >= evaluation_start or model["label_end_ms"] >= bundle["cutoff_ms"]:
            raise ValueError("training labels overlap evaluation")
        if model["features"] != names:
            raise ValueError("feature schema mismatch")
        for name in ("mean", "scale", "weights"):
            if len(model[name]) != len(names) or not all(math.isfinite(v) for v in model[name]):
                raise ValueError("invalid model coefficients")
        if min(model["scale"]) <= 0 or not math.isfinite(model["intercept"]):
            raise ValueError("invalid model normalization")


class FrozenPredictor:
    def __init__(self, artifact):
        self.artifact = artifact
        self._names = tuple(artifact.get("features", FEATURES))
        self._mean = np.asarray(artifact["mean"], dtype=float)
        self._scale = np.asarray(artifact["scale"], dtype=float)
        self._weights = np.asarray(artifact["weights"], dtype=float)
        self._config = {}

    def predict(self, features):
        x = vector(features, self._names)
        if x is None:
            return None
        return float(((x - self._mean) / self._scale) @ self._weights + self.artifact["intercept"])

    def update_online_ridge(self, realized_return):
        pass  # frozen artifact; no learning from the final holdout
