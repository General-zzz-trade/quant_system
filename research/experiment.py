#!/usr/bin/env python3
"""Research experiment framework — run, track, and compare experiments.

Usage:
    python3 -m research.experiment run --config experiments/my_experiment.yaml
    python3 -m research.experiment compare --dir experiments/results/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """A minimal experiment definition."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    dataset: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def with_params(self, **kwargs: Any) -> "Experiment":
        p = dict(self.params)
        p.update(kwargs)
        return Experiment(name=self.name, params=p, dataset=dict(self.dataset), created_at=self.created_at)


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_id: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    model_path: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None
    duration_sec: float = 0.0
    timestamp: str = ""


@dataclass
class ExperimentRunner:
    """Runs experiments with config snapshot and metrics tracking."""

    out_dir: Path = Path("experiments/results")

    def run(self, config: Dict[str, Any]) -> ExperimentResult:
        """Run a single experiment from config dict."""
        import numpy as np
        import pandas as pd

        from alpha.models.lgbm_alpha import LGBMAlphaModel
        from alpha.training.trainer import ModelTrainer
        from scripts.archive.train_lgbm import FEATURE_NAMES, compute_features_from_ohlcv, compute_target

        experiment_id = _make_id(config)
        run_dir = self.out_dir / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        t0 = time.monotonic()

        df = pd.read_csv(config.get("data_path", "data/btcusdt.csv"))

        features_cfg = config.get("features", {})
        feat_df = compute_features_from_ohlcv(
            df,
            fast_ma=features_cfg.get("fast_ma", 10),
            slow_ma=features_cfg.get("slow_ma", 30),
            vol_window=features_cfg.get("vol_window", 20),
        )

        training_cfg = config.get("training", {})
        target = compute_target(df["close"], horizon=training_cfg.get("horizon", 5))

        X = feat_df[list(FEATURE_NAMES)]
        y = target
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask].values, y[mask].values

        model = LGBMAlphaModel(name="lgbm_alpha", feature_names=FEATURE_NAMES)
        trainer = ModelTrainer(model=model, out_dir=run_dir)
        results = trainer.walk_forward_train(
            X, y, n_splits=training_cfg.get("n_splits", 5), expanding=True,
        )

        duration = time.monotonic() - t0

        metrics = {
            "avg_val_mse": float(np.mean([r.metrics["val_mse"] for r in results])),
            "avg_direction_accuracy": float(np.mean([r.metrics["direction_accuracy"] for r in results])),
            "n_folds": len(results),
            "n_samples": len(X),
        }

        feature_importance = None
        if hasattr(model, "_model") and model._model is not None:
            try:
                imp = model._model.feature_importances_
                feature_importance = dict(zip(FEATURE_NAMES, [float(x) for x in imp]))
            except Exception:
                pass

        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            metrics=metrics,
            model_path=str(run_dir / "lgbm_alpha_final.pkl"),
            feature_importance=feature_importance,
            duration_sec=duration,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        with open(run_dir / "result.json", "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        logger.info(
            "Experiment %s: direction_acc=%.4f duration=%.1fs",
            experiment_id, metrics["avg_direction_accuracy"], duration,
        )
        return result


def compare_experiments(results_dir: Path) -> None:
    """Print comparison table of all experiments in a results directory."""
    results = []
    for result_path in sorted(results_dir.glob("*/result.json")):
        with open(result_path) as f:
            results.append(json.load(f))

    if not results:
        print("No experiment results found.")
        return

    print(f"{'ID':<20} {'Dir Acc':>10} {'Val MSE':>12} {'Samples':>8} {'Duration':>10}")
    print("-" * 64)
    for r in sorted(results, key=lambda x: -x["metrics"]["avg_direction_accuracy"]):
        print(
            f"{r['experiment_id'][:20]:<20} "
            f"{r['metrics']['avg_direction_accuracy']:>10.4f} "
            f"{r['metrics']['avg_val_mse']:>12.6f} "
            f"{r['metrics']['n_samples']:>8} "
            f"{r['duration_sec']:>9.1f}s"
        )


def _make_id(config: Dict[str, Any]) -> str:
    cfg_str = json.dumps(config, sort_keys=True, default=str)
    h = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{h}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Research experiment framework")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, default=Path("experiments/results"))

    compare_parser = sub.add_parser("compare", help="Compare experiment results")
    compare_parser.add_argument("--dir", type=Path, default=Path("experiments/results"))

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "run":
        import yaml  # type: ignore[import-untyped]
        with open(args.config) as f:
            config = yaml.safe_load(f)
        runner = ExperimentRunner(out_dir=args.out)
        result = runner.run(config)
        print(f"Experiment {result.experiment_id}: direction_accuracy={result.metrics['avg_direction_accuracy']:.4f}")
    elif args.command == "compare":
        compare_experiments(args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
