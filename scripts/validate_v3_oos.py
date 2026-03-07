"""Train regularized models on 80% data, run full backtest on 20% holdout.

Tests multiple configs (regularization levels, forward targets) and thresholds.

Usage:
    python3 -m scripts.validate_v3_oos
"""
from __future__ import annotations

import json
import pickle
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TRAIN_FRAC = 0.80

# Features computable from OHLCV only (actual names from EnrichedFeatureComputer)
OHLCV_FEATURES = [
    "rsi_14", "rsi_6", "macd_line", "macd_signal", "macd_hist",
    "atr_norm_14", "bb_pctb_20", "bb_width_20",
    "close_vs_ma20", "close_vs_ma50", "ma_cross_5_20", "ma_cross_10_30",
    "mean_reversion_20",
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "vol_5", "vol_20", "vol_ma_ratio_5_20", "vol_ratio_20", "vol_regime",
    "price_acceleration",
    "body_ratio", "upper_shadow", "lower_shadow",
    "trade_intensity", "taker_imbalance", "taker_buy_ratio", "taker_buy_ratio_ma10",
    "avg_trade_size", "avg_trade_size_ratio", "volume_per_trade", "trade_count_regime",
]

CONFIGS = {
    "mod_reg_1h": {
        "params": {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.02,
            "num_leaves": 20, "min_child_samples": 50,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "objective": "regression", "verbosity": -1,
        },
        "fwd_bars": 1,
    },
    "mod_reg_4h": {
        "params": {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.02,
            "num_leaves": 20, "min_child_samples": 50,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "objective": "regression", "verbosity": -1,
        },
        "fwd_bars": 4,
    },
    "light_reg_1h": {
        "params": {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "num_leaves": 31, "min_child_samples": 30,
            "reg_alpha": 0.0, "reg_lambda": 0.1,
            "objective": "regression", "verbosity": -1,
        },
        "fwd_bars": 1,
    },
}

THRESHOLDS = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02]


def _sf(v, default: float = 0.0) -> float:
    try:
        return float(v) if v != "" else default
    except (ValueError, TypeError):
        return default


def compute_features(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for all rows using EnrichedFeatureComputer."""
    from features.enriched_computer import EnrichedFeatureComputer
    fc = EnrichedFeatureComputer()
    features_list = []
    for _, row in df.iterrows():
        feats = fc.on_bar(
            symbol,
            close=float(row["close"]),
            high=float(row["high"]),
            low=float(row["low"]),
            open_=float(row["open"]),
            volume=float(row["volume"]),
            quote_volume=float(row.get("quote_volume", 0)),
            taker_buy_volume=float(row.get("taker_buy_volume", 0)),
            trades=float(row.get("trades", 0)),
        )
        features_list.append(feats)
    return pd.DataFrame(features_list)


def train_config(
    symbol: str, config_name: str, config: dict,
    feat_df: pd.DataFrame, close_series: pd.Series,
    out_base: Path,
) -> tuple[Path, list[str]]:
    """Train a single config. Returns (model_path, feature_names)."""
    import lightgbm as lgb

    fwd_bars = config["fwd_bars"]
    fwd = close_series.pct_change(fwd_bars).shift(-fwd_bars).values
    feat_df_copy = feat_df.copy()
    feat_df_copy["target"] = fwd[:len(feat_df_copy)]

    use_features = [f for f in OHLCV_FEATURES if f in feat_df_copy.columns]
    X = feat_df_copy[use_features].values.astype(np.float64)
    y = feat_df_copy["target"].values.astype(np.float64)

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]

    val_split = int(len(X) * 0.9)
    model = lgb.LGBMRegressor(**config["params"])
    model.fit(X[:val_split], y[:val_split], eval_set=[(X[val_split:], y[val_split:])])

    model_dir = out_base / "models" / symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{config_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": tuple(use_features)}, f)

    print(f"  [{config_name}] Trained on {len(X)} samples, {len(use_features)} feats, fwd={fwd_bars}h")
    return model_path, use_features


def score_distribution(model_path: Path, symbol: str, test_df: pd.DataFrame, train_df: pd.DataFrame) -> np.ndarray:
    """Compute raw model scores on test data (with warmup from training tail)."""
    from features.enriched_computer import EnrichedFeatureComputer
    from infra.model_signing import load_verified_pickle

    data = load_verified_pickle(model_path)
    model = data["model"]
    feature_names = data["features"]

    fc = EnrichedFeatureComputer()
    warmup = min(200, len(train_df))
    for _, row in train_df.iloc[-warmup:].iterrows():
        fc.on_bar(
            symbol,
            close=float(row["close"]), high=float(row["high"]),
            low=float(row["low"]), open_=float(row["open"]),
            volume=float(row["volume"]),
            quote_volume=float(row.get("quote_volume", 0)),
            taker_buy_volume=float(row.get("taker_buy_volume", 0)),
            trades=float(row.get("trades", 0)),
        )

    test_feats = []
    for _, row in test_df.iterrows():
        feats = fc.on_bar(
            symbol,
            close=float(row["close"]), high=float(row["high"]),
            low=float(row["low"]), open_=float(row["open"]),
            volume=float(row["volume"]),
            quote_volume=float(row.get("quote_volume", 0)),
            taker_buy_volume=float(row.get("taker_buy_volume", 0)),
            trades=float(row.get("trades", 0)),
        )
        test_feats.append(feats)

    test_feat_df = pd.DataFrame(test_feats)
    X = test_feat_df[[f for f in feature_names if f in test_feat_df.columns]].fillna(0).values
    return model.predict(X)


def run_backtests(
    symbol: str, model_path: Path, test_csv: Path, out_base: Path, config_name: str,
) -> dict:
    """Run full backtest for multiple thresholds."""
    from alpha.models.lgbm_alpha import LGBMAlphaModel
    from decision.ml_decision import make_ml_decision
    from features.enriched_computer import EnrichedFeatureComputer
    from runner.backtest_runner import run_backtest

    alpha = LGBMAlphaModel(name=config_name)
    alpha.load(model_path)

    results = {}
    for th in THRESHOLDS:
        dm = make_ml_decision(
            symbol=symbol, risk_pct=0.30, threshold=th,
            threshold_short=999.0,  # long-only
        )
        fc_bt = EnrichedFeatureComputer()
        bt_dir = out_base / symbol / config_name / f"th{th}"

        try:
            equity, fills = run_backtest(
                csv_path=test_csv,
                symbol=symbol,
                starting_balance=Decimal("10000"),
                fee_bps=Decimal("4"),
                slippage_bps=Decimal("2"),
                out_dir=bt_dir,
                decision_modules=[dm],
                feature_computer=fc_bt,
                alpha_models=[alpha],
            )

            summary_path = bt_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                summary["threshold"] = th
                results[th] = summary
            else:
                results[th] = {"threshold": th, "return": 0, "trades": 0}
        except Exception as e:
            print(f"    th={th}: ERROR {e}")
            results[th] = {"threshold": th, "error": str(e)}

    return results


def main() -> None:
    out_base = Path("output/v3_validation")
    all_results: dict = {}

    for sym in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"  {sym}")
        print(f"{'='*60}")

        csv_path = Path(f"data_files/{sym}_1h.csv")
        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        split = int(len(df) * TRAIN_FRAC)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        print(f"  {len(train_df)} train, {len(test_df)} test bars")

        # Compute features once for training
        print("  Computing training features...")
        feat_df = compute_features(sym, train_df)

        # Save test CSV once
        test_csv = out_base / f"{sym}_test.csv"
        out_base.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(test_csv, index=False)

        all_results[sym] = {}

        for cfg_name, cfg in CONFIGS.items():
            print(f"\n  --- {cfg_name} ---")

            # Train
            model_path, use_feats = train_config(
                sym, cfg_name, cfg, feat_df, train_df["close"], out_base,
            )

            # Score distribution
            scores = score_distribution(model_path, sym, test_df, train_df)
            print(f"  Scores: mean={scores.mean():.6f}, std={scores.std():.6f}, "
                  f"[{scores.min():.6f}, {scores.max():.6f}]")
            for th in THRESHOLDS:
                n_long = (scores > th).sum()
                n_short = (scores < -th).sum()
                if n_long > 0 or n_short > 0:
                    print(f"    th={th}: {n_long} long, {n_short} short")

            # Backtest
            results = run_backtests(sym, model_path, test_csv, out_base, cfg_name)
            all_results[sym][cfg_name] = results

            # Print results
            for th in sorted(results.keys()):
                r = results[th]
                trades = int(_sf(r.get("trades", 0)))
                if trades > 0:
                    ret = _sf(r.get("return", 0)) * 100
                    sharpe = _sf(r.get("sharpe_ratio", 0))
                    maxdd = _sf(r.get("max_drawdown", 0)) * 100
                    wr = _sf(r.get("win_rate", 0)) * 100
                    print(f"    th={th}: ret={ret:+.2f}%, sharpe={sharpe:.3f}, "
                          f"trades={trades}, maxDD={maxdd:.1f}%, WR={wr:.0f}%")

    # Save
    with open(out_base / "results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"{'Symbol':<10} {'Config':<16} {'Th':>7} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>7}")
    print(f"{'-'*90}")
    for sym in all_results:
        for cfg_name in all_results[sym]:
            for th in sorted(all_results[sym][cfg_name].keys()):
                r = all_results[sym][cfg_name][th]
                trades = int(_sf(r.get("trades", 0)))
                if trades > 0:
                    ret = _sf(r.get("return", 0)) * 100
                    sharpe = _sf(r.get("sharpe_ratio", 0))
                    maxdd = _sf(r.get("max_drawdown", 0)) * 100
                    print(f"{sym:<10} {cfg_name:<16} {th:>7.4f} {ret:>+9.2f}% {sharpe:>8.3f} {maxdd:>7.1f}% {trades:>7}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
