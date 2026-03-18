#!/usr/bin/env python3
"""Sweep configurations to find optimal portfolio parameters.

Tests multiple deadzone/long_only combinations per symbol.
Pickle is used for trusted local model files from our own training pipeline
(same pattern as model_loader.py and run_portfolio_backtest.py).
"""
from __future__ import annotations

import json
import pickle  # noqa: S403 — trusted local model files only, same as model_loader.py
from pathlib import Path

import numpy as np
import pandas as pd

WF_TRAIN_BARS = 4320
WF_TEST_BARS = 720
WF_STEP_BARS = 720
ZSCORE_WINDOW = 720
ZSCORE_WARMUP = 180


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)
    for c in ["close", "open", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_features(df, eth_closes=None):
    from _quant_hotpath import RustFeatureEngine
    eng = RustFeatureEngine()
    features_list = []
    for i, (_, row) in enumerate(df.iterrows()):
        close = float(row["close"])
        eng.push_bar(close=close, volume=float(row.get("volume", 0)),
                     high=float(row.get("high", close)),
                     low=float(row.get("low", close)),
                     open=float(row.get("open", close)))
        feats = eng.get_features()
        if eth_closes is not None and i < len(eth_closes) and eth_closes[i] > 0:
            ratio = close / eth_closes[i]
            if i >= 20:
                buf = [df["close"].iloc[j] / eth_closes[j]
                       for j in range(max(0, i - 49), i + 1) if eth_closes[j] > 0]
                if len(buf) >= 20:
                    feats["btc_dom_dev_20"] = (ratio - np.mean(buf[-20:])) / np.mean(buf[-20:])
                if len(buf) >= 50:
                    feats["btc_dom_dev_50"] = (ratio - np.mean(buf[-50:])) / np.mean(buf[-50:])
            if i >= 24 and eth_closes[i - 24] > 0:
                r24 = df["close"].iloc[i - 24] / eth_closes[i - 24]
                feats["btc_dom_ret_24"] = (ratio - r24) / r24
            if i >= 72 and eth_closes[i - 72] > 0:
                r72 = df["close"].iloc[i - 72] / eth_closes[i - 72]
                feats["btc_dom_ret_72"] = (ratio - r72) / r72
        features_list.append(feats)
    return pd.DataFrame(features_list, index=df.index)


def pred_to_signal(preds, deadzone, min_hold, max_hold, long_only):
    z = np.full_like(preds, np.nan)
    for i in range(len(preds)):
        chunk = preds[max(0, i - ZSCORE_WINDOW + 1):i + 1]
        if len(chunk) < ZSCORE_WARMUP:
            continue
        std = np.std(chunk)
        z[i] = (preds[i] - np.mean(chunk)) / std if std > 1e-10 else 0.0
    z = np.clip(z, -5.0, 5.0)
    if long_only:
        z = np.maximum(z, 0.0)
    raw = np.where(z > deadzone, 1.0, np.where(z < -deadzone, -1.0, 0.0))
    signal = np.zeros_like(raw)
    hold_count = 0
    current = 0.0
    for i in range(len(raw)):
        if np.isnan(raw[i]):
            continue
        if current != 0 and hold_count < min_hold:
            signal[i] = current
            hold_count += 1
        elif current != 0 and hold_count >= max_hold:
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
            current = 0.0
            hold_count = 0
    return signal


def run_wf(feat_df, df, lgbm, ridge, features, ridge_features, rw, lw,
           deadzone, min_hold, max_hold, long_only, cost_bps=6.0):
    n = len(df)
    folds = []
    s = 0
    while s + WF_TRAIN_BARS + WF_TEST_BARS <= n:
        folds.append((s, s + WF_TRAIN_BARS, s + WF_TRAIN_BARS + WF_TEST_BARS))
        s += WF_STEP_BARS
    if not folds:
        sp = int(n * 0.8)
        folds = [(0, sp, n)]
    fold_rets = []
    all_rets = []
    for _, tr_e, te_e in folds:
        X = np.nan_to_num(feat_df.iloc[tr_e:te_e].reindex(columns=features, fill_value=0.0).values, nan=0.0)
        closes = df["close"].iloc[tr_e:te_e].values
        pred = lgbm.predict(X)
        if ridge is not None:
            rf = feat_df.iloc[tr_e:te_e].reindex(columns=ridge_features, fill_value=0.0)
            Xr = rf.fillna(0.0).values.astype(np.float64)
            pred = rw * ridge.predict(Xr) + lw * pred
        sig = pred_to_signal(pred, deadzone, min_hold, max_hold, long_only)
        rets = np.diff(closes) / closes[:-1]
        sig_t = sig[:-1]
        sr = sig_t * rets
        chg = np.concatenate([[False], np.diff(sig_t) != 0])
        sr[chg] -= cost_bps / 10000
        fold_rets.append(np.prod(1 + sr) - 1)
        all_rets.extend(sr.tolist())
    ar = np.array(all_rets)
    pos = sum(1 for r in fold_rets if r > 0)
    total = np.prod([1 + r for r in fold_rets]) - 1
    sharpe = np.mean(ar) / np.std(ar) * np.sqrt(8760) if np.std(ar) > 0 else 0.0
    dd = np.min(np.cumprod(1 + ar) / np.maximum.accumulate(np.cumprod(1 + ar)) - 1) if len(ar) > 0 else 0
    return {"folds": len(fold_rets), "positive": pos, "total_return": total,
            "sharpe": sharpe, "max_dd": dd, "returns": all_rets}


def main():
    SWEEP = {
        "BTCUSDT": {
            "csv": "data_files/BTCUSDT_1h.csv", "model_dir": "models_v8/BTCUSDT_gate_v2",
            "min_hold": 48, "max_hold": 288,
            "configs": [
                {"dz": 0.8, "lo": False, "label": "baseline(dz=0.8)"},
                {"dz": 0.8, "lo": True, "label": "LO(dz=0.8)"},
                {"dz": 1.2, "lo": False, "label": "dz=1.2"},
                {"dz": 1.2, "lo": True, "label": "LO+dz=1.2"},
                {"dz": 1.5, "lo": True, "label": "LO+dz=1.5"},
            ],
        },
        "ETHUSDT": {
            "csv": "data_files/ETHUSDT_1h.csv", "model_dir": "models_v8/ETHUSDT_gate_v2",
            "min_hold": 18, "max_hold": 60,
            "configs": [
                {"dz": 0.3, "lo": False, "label": "baseline(dz=0.3)"},
                {"dz": 0.4, "lo": False, "label": "dz=0.4"},
                {"dz": 0.5, "lo": False, "label": "dz=0.5"},
                {"dz": 0.3, "lo": True, "label": "LO(dz=0.3)"},
                {"dz": 0.4, "lo": True, "label": "LO+dz=0.4"},
            ],
        },
        "SUIUSDT": {
            "csv": "data_files/SUIUSDT_1h.csv", "model_dir": "models_v8/SUIUSDT",
            "min_hold": 18, "max_hold": 60,
            "configs": [
                {"dz": 0.7, "lo": True, "label": "current(LO+0.7)"},
                {"dz": 0.5, "lo": True, "label": "LO+dz=0.5"},
                {"dz": 0.9, "lo": True, "label": "LO+dz=0.9"},
                {"dz": 0.4, "lo": False, "label": "old(dz=0.4)"},
            ],
        },
        "AXSUSDT": {
            "csv": "data_files/AXSUSDT_1h.csv", "model_dir": "models_v8/AXSUSDT",
            "min_hold": 18, "max_hold": 60,
            "configs": [
                {"dz": 0.6, "lo": True, "label": "current(LO+0.6)"},
                {"dz": 0.8, "lo": True, "label": "LO+dz=0.8"},
                {"dz": 1.0, "lo": True, "label": "LO+dz=1.0"},
                {"dz": 0.25, "lo": False, "label": "old(dz=0.25)"},
            ],
        },
    }

    all_results = {}
    for symbol, spec in SWEEP.items():
        print(f"\n{'='*70}")
        print(f"  SWEEP: {symbol}")
        print(f"{'='*70}")
        df = load_csv(spec["csv"])
        md = Path(spec["model_dir"])
        with open(md / "config.json") as f:
            mcfg = json.load(f)
        hm = mcfg["horizon_models"][0]
        features = hm["features"]
        ridge_features = hm.get("ridge_features", features)
        with open(md / hm["lgbm"], "rb") as f:  # noqa: S301
            raw = pickle.load(f)
        lgbm = raw["model"] if isinstance(raw, dict) else raw
        ridge = None
        rn = hm.get("ridge", "")
        if rn and (md / rn).exists():
            with open(md / rn, "rb") as f:  # noqa: S301
                raw = pickle.load(f)
            ridge = raw["model"] if isinstance(raw, dict) else raw
            if isinstance(raw, dict) and "features" in raw:
                ridge_features = raw["features"]
        rw = mcfg.get("ridge_weight", 0.6)
        lw = mcfg.get("lgbm_weight", 0.4)
        eth_closes = None
        if symbol == "BTCUSDT":
            eth_df = load_csv("data_files/ETHUSDT_1h.csv")
            m = df[["open_time"]].merge(
                eth_df[["open_time", "close"]].rename(columns={"close": "ec"}),
                on="open_time", how="left")
            eth_closes = m["ec"].ffill().values
        print("  Computing features...")
        feat_df = compute_features(df, eth_closes)
        sym_results = []
        for cfg in spec["configs"]:
            r = run_wf(feat_df, df, lgbm, ridge, features, ridge_features, rw, lw,
                       cfg["dz"], spec["min_hold"], spec["max_hold"], cfg["lo"])
            r["label"] = cfg["label"]
            sym_results.append(r)
        all_results[symbol] = sym_results
        print(f"\n  {'Config':<20} {'Pass':>7} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8}")
        print(f"  {'-'*54}")
        for r in sym_results:
            print(f"  {r['label']:<20} {r['positive']:>3}/{r['folds']:<3} "
                  f"{r['total_return']:>+8.1%} {r['sharpe']:>8.2f} {r['max_dd']:>7.1%}")

    # Find best per symbol (highest Sharpe with >45% positive folds)
    print(f"\n{'='*70}")
    print("  BEST CONFIG PER SYMBOL")
    print(f"{'='*70}")
    best = {}
    for sym, results in all_results.items():
        good = [r for r in results if r["positive"] / max(r["folds"], 1) > 0.45]
        if good:
            b = max(good, key=lambda r: r["sharpe"])
        else:
            b = max(results, key=lambda r: r["sharpe"])
        best[sym] = b
        print(f"  {sym:<12} {b['label']:<20} Sharpe={b['sharpe']:.2f} "
              f"Return={b['total_return']:+.1%} Pass={b['positive']}/{b['folds']} DD={b['max_dd']:.1%}")

    # Optimal portfolio
    print(f"\n{'='*70}")
    print("  OPTIMAL PORTFOLIO (equal weight, best configs)")
    print(f"{'='*70}")
    active = [s for s, b in best.items() if b["returns"]]
    if not active:
        return
    max_len = max(len(best[s]["returns"]) for s in active)
    port = np.zeros(max_len)
    cnt = np.zeros(max_len)
    w = 1.0 / len(active)
    for sym in active:
        rets = np.array(best[sym]["returns"])
        off = max_len - len(rets)
        port[off:off + len(rets)] += rets * w
        cnt[off:off + len(rets)] += w
    mask = cnt > 0
    port[mask] /= cnt[mask]
    port = port[mask]
    cum = np.cumprod(1 + port)
    print(f"  Return:  {cum[-1] - 1:+.1%}")
    print(f"  Sharpe:  {np.mean(port) / np.std(port) * np.sqrt(8760):.2f}")
    print(f"  MaxDD:   {np.min(cum / np.maximum.accumulate(cum) - 1):.1%}")
    print(f"  WinRate: {np.mean(port > 0) * 100:.1f}%")


if __name__ == "__main__":
    main()
