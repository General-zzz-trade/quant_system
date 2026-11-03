"""Small, transparent Ridge reference models with auditable training boundaries.

These are new research models, not a reconstruction of the unversioned live
model files. Features use the existing incremental production computer.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from data.quality.causal_bars import aggregate_closed, fingerprint
from features.enriched_computer import EnrichedFeatureComputer

FEATURES = ("ret_1", "ret_3", "ret_6", "ret_12", "vol_20")


def compute_bar(computer, bar):
    ts = datetime.fromtimestamp(bar.available_at / 1000, timezone.utc)
    return computer.on_bar("BTCUSDT", open_=float(bar.open), high=float(bar.high), low=float(bar.low),
                           close=float(bar.close), volume=float(bar.volume), hour=ts.hour, dow=ts.weekday())


def feature_rows(bars):
    computer = EnrichedFeatureComputer()
    return [compute_bar(computer, bar) for bar in bars]


def vector(features):
    values = [features.get(k) for k in FEATURES]
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return None
    return np.asarray(values, dtype=float)


def fit_model(bars, cutoff, alpha=1.0):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge regularization must be positive")
    rows = feature_rows(bars)
    samples = [(vector(rows[i]), math.log(float(bars[i + 1].close / bars[i].close)), bars[i + 1].available_at)
               for i in range(50, len(bars) - 1) if bars[i + 1].available_at < cutoff]
    samples = [s for s in samples if s[0] is not None]
    if len(samples) < 40:
        raise ValueError("need at least 40 eligible training labels after 50 warmup bars per timeframe")
    x = np.stack([s[0] for s in samples])
    y = np.asarray([s[1] for s in samples])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    normalized = (x - mean) / scale
    intercept = float(y.mean())
    weights = np.linalg.solve(normalized.T @ normalized + alpha * np.eye(len(FEATURES)),
                              normalized.T @ (y - intercept))
    return {"features": list(FEATURES), "mean": mean.tolist(), "scale": scale.tolist(),
            "weights": weights.tolist(), "intercept": intercept, "alpha": alpha,
            "samples": len(samples), "label_end_ms": max(s[2] for s in samples),
            "training_sha256": fingerprint([b.record() for b in bars if b.available_at < cutoff])}


def train_bundle(bars, cutoff):
    # Cut raw input before feature computation: a trainer cannot inspect held-out bars.
    history = [b for b in bars if b.available_at < cutoff]
    models = {"1h": fit_model(history, cutoff), "4h": fit_model(aggregate_closed(history), cutoff)}
    bundle = {"schema": 1, "kind": "incremental_ohlcv_ridge_reference", "cutoff_ms": cutoff,
              "models": models, "hyperparameter_candidates": 1, "online_updates": False}
    bundle["sha256"] = fingerprint(bundle)
    return bundle


def validate_bundle(bundle, evaluation_start):
    raw = {k: v for k, v in bundle.items() if k != "sha256"}
    if bundle.get("schema") != 1 or fingerprint(raw) != bundle.get("sha256"):
        raise ValueError("model artifact hash/schema mismatch")
    if bundle["cutoff_ms"] > evaluation_start:
        raise ValueError("model trained after evaluation began")
    for interval in ("1h", "4h"):
        model = bundle["models"][interval]
        if model["label_end_ms"] >= evaluation_start or model["label_end_ms"] >= bundle["cutoff_ms"]:
            raise ValueError("training labels overlap evaluation")
        if model["features"] != list(FEATURES):
            raise ValueError("feature schema mismatch")
        for name in ("mean", "scale", "weights"):
            if len(model[name]) != len(FEATURES) or not all(math.isfinite(v) for v in model[name]):
                raise ValueError("invalid model coefficients")
        if min(model["scale"]) <= 0 or not math.isfinite(model["intercept"]):
            raise ValueError("invalid model normalization")


class FrozenPredictor:
    def __init__(self, artifact):
        self.artifact = artifact
        self._config = {}

    def predict(self, features):
        x = vector(features)
        if x is None:
            return None
        model = self.artifact
        return float(((x - model["mean"]) / model["scale"]) @ model["weights"] + model["intercept"])

    def update_online_ridge(self, realized_return):
        pass  # frozen artifact; no learning from the final holdout
