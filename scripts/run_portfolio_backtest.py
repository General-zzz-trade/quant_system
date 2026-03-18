#!/usr/bin/env python3
"""Portfolio backtest for current alpha configuration.

Runs walk-forward backtest for each symbol with live config parameters,
then combines into portfolio-level metrics.

Uses pickle for loading trusted local model files produced by our own
training pipeline (same pattern as model_loader.py).
"""
from __future__ import annotations

import json
import pickle  # noqa: S403 — trusted local model files only
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config: auto-loads from model config.json ──
LEVERAGE = 5.0


def _build_symbols() -> dict:
    """Build SYMBOLS dict from model config.json files."""
    _SYMBOL_DIRS = {
        "BTCUSDT": "models_v8/BTCUSDT_gate_v2",
        "ETHUSDT": "models_v8/ETHUSDT_gate_v2",
        "SUIUSDT": "models_v8/SUIUSDT",
        "AXSUSDT": "models_v8/AXSUSDT",
    }
    symbols = {}
    for sym, model_dir in _SYMBOL_DIRS.items():
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        symbols[sym] = {
            "csv": f"data_files/{sym}_1h.csv",
            "model_dir": model_dir,
            "deadzone": cfg.get("deadzone", 0.5),
            "min_hold": cfg.get("min_hold", 18),
            "max_hold": cfg.get("max_hold", 60),
            "long_only": cfg.get("long_only", False),
            "cost_bps": 6.0,
            "leverage": LEVERAGE,
            "weight": 0.25,
        }
    return symbols


SYMBOLS = _build_symbols()

# Walk-forward parameters
WF_TRAIN_BARS = 4320   # 6 months
WF_TEST_BARS = 720     # 1 month
WF_STEP_BARS = 720     # slide 1 month
ZSCORE_WINDOW = 720
ZSCORE_WARMUP = 180


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    if "open_time" in cols:
        df = df.sort_values("open_time").reset_index(drop=True)
    elif cols[0].isdigit() or str(df.iloc[0, 0]).replace(".", "").isdigit():
        df.columns = ["open_time", "open", "high", "low", "close", "volume"] + cols[6:]
        df = df.sort_values("open_time").reset_index(drop=True)
    for c in ["close", "open", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_features(df: pd.DataFrame, eth_closes: np.ndarray | None = None) -> pd.DataFrame:
    """Compute features using RustFeatureEngine (same as live).

    For BTC V14 dominance features, pass eth_closes aligned to df index.
    """
    from _quant_hotpath import RustFeatureEngine
    eng = RustFeatureEngine()
    features_list = []
    for i, (_, row) in enumerate(df.iterrows()):
        close = float(row["close"])
        eng.push_bar(
            close=close,
            volume=float(row.get("volume", 0)),
            high=float(row.get("high", close)),
            low=float(row.get("low", close)),
            open=float(row.get("open", close)),
        )
        feats = eng.get_features()
        # V14: add BTC/ETH dominance features manually if ETH data available
        if eth_closes is not None and i < len(eth_closes):
            eth_c = eth_closes[i]
            if close > 0 and eth_c > 0:
                ratio = close / eth_c
                # Rolling deviations and returns on ratio
                if i >= 20:
                    buf = [df["close"].iloc[j] / eth_closes[j]
                           for j in range(max(0, i - 49), i + 1)
                           if eth_closes[j] > 0]
                    if len(buf) >= 20:
                        ma20 = np.mean(buf[-20:])
                        feats["btc_dom_dev_20"] = (ratio - ma20) / ma20 if ma20 > 0 else 0
                    if len(buf) >= 50:
                        ma50 = np.mean(buf[-50:])
                        feats["btc_dom_dev_50"] = (ratio - ma50) / ma50 if ma50 > 0 else 0
                if i >= 24:
                    r24 = df["close"].iloc[i - 24] / eth_closes[i - 24] if eth_closes[i - 24] > 0 else ratio
                    feats["btc_dom_ret_24"] = (ratio - r24) / r24 if r24 > 0 else 0
                if i >= 72:
                    r72 = df["close"].iloc[i - 72] / eth_closes[i - 72] if eth_closes[i - 72] > 0 else ratio
                    feats["btc_dom_ret_72"] = (ratio - r72) / r72 if r72 > 0 else 0
        features_list.append(feats)
    return pd.DataFrame(features_list, index=df.index)


def rolling_zscore(preds: np.ndarray, window: int = 720, warmup: int = 180) -> np.ndarray:
    z = np.full_like(preds, np.nan)
    for i in range(len(preds)):
        start = max(0, i - window + 1)
        chunk = preds[start:i + 1]
        if len(chunk) < warmup:
            continue
        mu = np.mean(chunk)
        std = np.std(chunk)
        z[i] = (preds[i] - mu) / std if std > 1e-10 else 0.0
    return z


def pred_to_signal(preds: np.ndarray, deadzone: float, min_hold: int,
                   max_hold: int, long_only: bool) -> np.ndarray:
    """Z-score -> discretize -> min-hold -> max-hold constraint pipeline."""
    z = rolling_zscore(preds, ZSCORE_WINDOW, ZSCORE_WARMUP)
    z = np.clip(z, -5.0, 5.0)
    if long_only:
        z = np.maximum(z, 0.0)

    raw = np.where(z > deadzone, 1.0, np.where(z < -deadzone, -1.0, 0.0))

    signal = np.zeros_like(raw)
    hold_count = 0
    current = 0.0
    for i in range(len(raw)):
        if np.isnan(raw[i]):
            signal[i] = 0.0
            continue
        if current != 0 and hold_count < min_hold:
            signal[i] = current
            hold_count += 1
        elif current != 0 and hold_count >= max_hold:
            signal[i] = 0.0
            current = 0.0
            hold_count = 0
        elif raw[i] != 0 and raw[i] != current:
            signal[i] = raw[i]
            current = raw[i]
            hold_count = 1
        elif raw[i] == current and current != 0:
            signal[i] = current
            hold_count += 1
        else:
            signal[i] = 0.0
            current = 0.0
            hold_count = 0
    return signal


def backtest_symbol(symbol: str, cfg: dict) -> dict:
    """Walk-forward backtest for one symbol."""
    print(f"\n{'='*70}")
    print(f"  {symbol}: dz={cfg['deadzone']} mh={cfg['min_hold']} "
          f"max_h={cfg['max_hold']} long_only={cfg['long_only']} lev={cfg.get('leverage', 1.0)}x")
    print(f"{'='*70}")

    df = load_csv(cfg["csv"])
    print(f"  Data: {len(df)} bars")

    model_dir = Path(cfg["model_dir"])
    with open(model_dir / "config.json") as f:
        model_cfg = json.load(f)

    hm = model_cfg["horizon_models"][0]
    features = hm["features"]
    ridge_features = hm.get("ridge_features", features)

    # Load models (trusted local artifacts from our training pipeline)
    with open(model_dir / hm["lgbm"], "rb") as f:
        raw = pickle.load(f)  # noqa: S301 — trusted local model
    lgbm = raw["model"] if isinstance(raw, dict) else raw

    ridge = None
    ridge_name = hm.get("ridge", "")
    if ridge_name and (model_dir / ridge_name).exists():
        with open(model_dir / ridge_name, "rb") as f:
            raw = pickle.load(f)  # noqa: S301 — trusted local model
        ridge = raw["model"] if isinstance(raw, dict) else raw
        # Ridge may have its own feature list stored in pickle
        if isinstance(raw, dict) and "features" in raw:
            ridge_features = raw["features"]

    ridge_w = model_cfg.get("ridge_weight", 0.6)
    lgbm_w = model_cfg.get("lgbm_weight", 0.4)

    # For BTC V14: load ETH closes for dominance features
    eth_closes = None
    if symbol == "BTCUSDT" and Path("data_files/ETHUSDT_1h.csv").exists():
        eth_df = load_csv("data_files/ETHUSDT_1h.csv")
        # Align by open_time if available, else by position
        if "open_time" in df.columns and "open_time" in eth_df.columns:
            merged = df[["open_time"]].merge(
                eth_df[["open_time", "close"]].rename(columns={"close": "eth_close"}),
                on="open_time", how="left",
            )
            eth_closes = merged["eth_close"].ffill().values
        else:
            # Positional alignment (truncate to shorter)
            min_len = min(len(df), len(eth_df))
            eth_closes = eth_df["close"].values[-min_len:]
            if len(df) > min_len:
                eth_closes = np.concatenate([np.full(len(df) - min_len, np.nan), eth_closes])

    print(f"  Computing features ({len(features)})...")
    feat_df = compute_features(df, eth_closes=eth_closes)

    # Walk-forward folds
    n = len(df)
    folds = []
    start = 0
    while start + WF_TRAIN_BARS + WF_TEST_BARS <= n:
        folds.append((start, start + WF_TRAIN_BARS, start + WF_TRAIN_BARS + WF_TEST_BARS))
        start += WF_STEP_BARS
    if not folds:
        split = int(n * 0.8)
        folds = [(0, split, n)]

    print(f"  Walk-forward: {len(folds)} folds")

    all_test_rets = []
    fold_results = []

    for fold_i, (tr_start, tr_end, te_end) in enumerate(folds):
        X_test = np.nan_to_num(
            feat_df.iloc[tr_end:te_end].reindex(columns=features, fill_value=0.0).values,
            nan=0.0,
        )
        closes_test = df["close"].iloc[tr_end:te_end].values

        lgbm_pred = lgbm.predict(X_test)
        if ridge is not None:
            # Ensure all ridge features exist; fill missing/None/NaN with 0
            rf_df = feat_df.iloc[tr_end:te_end].reindex(columns=ridge_features, fill_value=0.0)
            X_ridge = rf_df.fillna(0.0).values.astype(np.float64)
            pred = ridge_w * ridge.predict(X_ridge) + lgbm_w * lgbm_pred
        else:
            pred = lgbm_pred

        signal = pred_to_signal(pred, cfg["deadzone"], cfg["min_hold"],
                                cfg["max_hold"], cfg["long_only"])

        rets = np.diff(closes_test) / closes_test[:-1]
        sig_t = signal[:-1]
        leverage = cfg.get("leverage", 1.0)
        cost = cfg["cost_bps"] / 10000
        strat_ret = sig_t * rets * leverage
        changes = np.concatenate([[False], np.diff(sig_t) != 0])
        strat_ret[changes] -= cost * leverage  # costs also scale with leverage

        # Liquidation simulation: if single-bar loss exceeds margin (1/leverage),
        # cap loss at -100% for that bar (total equity wipeout on that position).
        if leverage > 1:
            liq_threshold = -1.0 / leverage  # e.g., -10% at 10x
            for li in range(len(strat_ret)):
                if strat_ret[li] < liq_threshold * leverage:
                    strat_ret[li] = -0.99  # near-total loss on position

        cum_ret = np.prod(1 + strat_ret) - 1
        n_trades = int(np.sum(changes))
        sharpe = (np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(8760)
                  if np.std(strat_ret) > 0 else 0.0)

        fold_results.append({
            "fold": fold_i + 1, "return": cum_ret, "sharpe": sharpe,
            "trades": n_trades,
            "long_pct": np.mean(sig_t > 0) * 100,
            "short_pct": np.mean(sig_t < 0) * 100,
            "flat_pct": np.mean(sig_t == 0) * 100,
        })
        all_test_rets.extend(strat_ret.tolist())

    positive = sum(1 for f in fold_results if f["return"] > 0)
    avg_sharpe = np.mean([f["sharpe"] for f in fold_results])
    total_ret = np.prod([1 + f["return"] for f in fold_results]) - 1

    print(f"\n  {'Fold':>5} {'Return':>8} {'Sharpe':>7} {'Trades':>7} "
          f"{'Long%':>6} {'Short%':>7} {'Flat%':>6}")
    for f in fold_results:
        m = "+" if f["return"] > 0 else "-"
        print(f"  {f['fold']:>5} {f['return']:>+7.1%} {f['sharpe']:>7.2f} "
              f"{f['trades']:>7} {f['long_pct']:>5.0f}% {f['short_pct']:>6.0f}% "
              f"{f['flat_pct']:>5.0f}% {m}")

    print(f"\n  {positive}/{len(fold_results)} positive | "
          f"Return: {total_ret:+.1%} | Sharpe: {avg_sharpe:.2f}")

    return {
        "symbol": symbol, "folds": len(fold_results), "positive": positive,
        "total_return": total_ret, "avg_sharpe": avg_sharpe,
        "fold_results": fold_results, "returns": all_test_rets,
    }


def main():
    print("=" * 70)
    print(f"  PORTFOLIO WALK-FORWARD BACKTEST (leverage={LEVERAGE}x)")
    print(f"  Symbols: {list(SYMBOLS.keys())}")
    print("=" * 70)

    results = {}
    for symbol, cfg in SYMBOLS.items():
        results[symbol] = backtest_symbol(symbol, cfg)

    # Portfolio summary
    print(f"\n{'='*70}")
    print("  PORTFOLIO SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Symbol':<12} {'Folds':>6} {'Pass':>7} {'Return':>9} {'Sharpe':>8}")
    print(f"  {'-'*44}")
    for sym, r in results.items():
        if not r:
            continue
        print(f"  {sym:<12} {r['folds']:>6} {r['positive']:>3}/{r['folds']:<3} "
              f"{r['total_return']:>+8.1%} {r['avg_sharpe']:>8.2f}")

    # Combined portfolio — Sharpe-weighted allocation
    _SHARPE_WEIGHTS = {
        "SUIUSDT": 0.40, "ETHUSDT": 0.28, "AXSUSDT": 0.19, "BTCUSDT": 0.11,
    }

    for label, weights in [("Sharpe-weighted", _SHARPE_WEIGHTS),
                           ("Sharpe-wt + DD control", _SHARPE_WEIGHTS)]:
        max_len = max(len(r["returns"]) for r in results.values() if r)
        port_rets = np.zeros(max_len)
        count = np.zeros(max_len)
        for sym, r in results.items():
            if not r or not r["returns"]:
                continue
            rets = np.array(r["returns"])
            n = len(rets)
            offset = max_len - n
            w = weights.get(sym, 0.25) if weights else 0.25
            port_rets[offset:offset + n] += rets * w
            count[offset:offset + n] += w

        mask = count > 0
        port_rets[mask] /= count[mask]
        port_rets = port_rets[mask]

        if len(port_rets) > 0:
            # Apply portfolio-level DD control for DD-controlled variant
            if "DD control" in label:
                cum_eq = np.cumprod(1 + port_rets)
                peak = np.maximum.accumulate(cum_eq)
                dd_pct = (peak - cum_eq) / peak
                # Optimized for 5x: DD>5% → 0.3x, DD>10% → stop (0x)
                dd_scale = np.where(dd_pct > 0.10, 0.0,
                           np.where(dd_pct > 0.05, 0.30, 1.0))
                port_rets = port_rets * dd_scale

            cum = np.cumprod(1 + port_rets)
            port_cum = cum[-1] - 1
            port_sharpe = np.mean(port_rets) / np.std(port_rets) * np.sqrt(8760)
            port_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
            win_rate = np.mean(port_rets > 0) * 100

            print(f"\n  PORTFOLIO ({label}):")
            print(f"    Total return:  {port_cum:+.1%}")
            print(f"    Sharpe:        {port_sharpe:.2f}")
            print(f"    Max drawdown:  {port_dd:.1%}")
            print(f"    Win rate:      {win_rate:.1f}%")
            print(f"    Bars:          {len(port_rets)}")


if __name__ == "__main__":
    main()
