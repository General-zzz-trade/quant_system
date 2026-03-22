#!/usr/bin/env python3
"""Full backtest: 4h alpha with actual trained models, 10x leverage, $500 start.

Uses pickle for loading sklearn/lightgbm trained model files (standard ML format).

Tests:
1. BTC/ETH 4h standalone
2. BTC/ETH 1h standalone (baseline)
3. 1h+4h combined portfolio
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import pickle  # noqa: S403 — required for sklearn/lightgbm model files
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from scripts.train_multi_horizon import rolling_zscore

DATA_DIR = Path("data_files")
MODEL_DIR = Path("models_v8")

FEE = 0.0004
SLIP = 0.0003
COST = FEE + SLIP


def load_and_resample(symbol: str, rule: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_1h.csv")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    if rule == "1h":
        return df
    df_r = df.set_index("datetime")
    agg = {"open_time": "first", "open": "first", "high": "max",
           "low": "min", "close": "last", "volume": "sum"}
    for c in ["quote_volume", "taker_buy_volume", "taker_buy_quote_volume", "trades"]:
        if c in df.columns:
            agg[c] = "sum"
    return df_r.resample(rule).agg(agg).dropna(subset=["close"]).reset_index()


def compute_all_features(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    feat_df = compute_features_batch(symbol, df)
    cm_path = DATA_DIR / "cross_market_daily.csv"
    if cm_path.exists():
        cm = pd.read_csv(cm_path, parse_dates=["date"])
        cm["date"] = cm["date"].dt.date
        bar_dates = pd.to_datetime(df["open_time"], unit="ms").dt.date
        cm_idx = cm.set_index("date")
        # T-1 shift: shift index forward by 1 day so bar on date D uses data from D-1
        cm_idx.index = [d + pd.Timedelta(days=1) for d in cm_idx.index]
        for col in cm_idx.columns:
            mapped = bar_dates.map(lambda d, c=col: cm_idx[c].get(d, np.nan))
            feat_df[col] = mapped.ffill().values
    return feat_df


def load_model_predict(symbol: str, interval: str, feat_df: pd.DataFrame):
    suffix = {"4h": "_4h", "1h": "_gate_v2", "15m": "_15m", "1d": "_1d"}[interval]
    mdir = MODEL_DIR / f"{symbol}{suffix}"
    cfg = json.load(open(mdir / "config.json"))
    features = cfg.get("features") or cfg["horizon_models"][0]["features"]
    for f in features:
        if f not in feat_df.columns:
            feat_df[f] = 0.0
    cfg["_features"] = features
    X = feat_df[features].values.astype(np.float64)

    # Find LGBM model file (naming varies: lgb_model.pkl, lgbm_v8.pkl, lgbm_h12.pkl)
    lgb_candidates = [
        mdir / "lgb_model.pkl",
        mdir / "lgbm_v8.pkl",
    ]
    # Also try horizon-specific files
    for hm in cfg.get("horizon_models", []):
        lgb_candidates.append(mdir / hm.get("lgbm", ""))

    lgb_path = None
    for p in lgb_candidates:
        if p.exists():
            lgb_path = p
            break

    if lgb_path is None:
        raise FileNotFoundError(f"No LGBM model found in {mdir}")

    with open(lgb_path, "rb") as fh:
        mdl = pickle.load(fh)
    # Model may be wrapped in dict: {"model": <Booster>, "features": [...]}
    if isinstance(mdl, dict) and "model" in mdl:
        mdl = mdl["model"]
    return mdl.predict(X), cfg


def gen_signal(pred, close, cfg):
    dz = cfg.get("deadzone", 1.0)
    mh = cfg.get("min_hold", 6)
    maxh = cfg.get("max_hold", 36)
    lo = cfg.get("long_only", False)
    mg = cfg.get("monthly_gate", False)
    zw = cfg.get("zscore_window", 180)
    zwm = cfg.get("zscore_warmup", 45)

    z = rolling_zscore(pred, window=zw, warmup=zwm)
    sig = np.zeros(len(z))
    sig[z > dz] = 1
    sig[z < -dz] = -1
    if lo:
        sig[sig < 0] = 0
    if mg:
        sw = 120 if len(close) < 5000 else 480
        sma = pd.Series(close).rolling(sw, min_periods=sw // 2).mean().values
        for i in range(len(sig)):
            if not np.isnan(sma[i]) and close[i] < sma[i]:
                sig[i] = min(sig[i], 0)

    cur, hold = 0, 0
    for i in range(len(sig)):
        s = sig[i]
        if s != cur and s != 0:
            cur, hold = s, 1
        elif cur != 0 and hold < mh:
            sig[i] = cur
            hold += 1
        elif s != cur:
            cur, hold = s, (1 if s != 0 else 0)

    cur, hold = 0, 0
    for i in range(len(sig)):
        if sig[i] != 0:
            if sig[i] == cur:
                hold += 1
                if hold > maxh:
                    sig[i], cur, hold = 0, 0, 0
            else:
                cur, hold = sig[i], 1
        else:
            cur, hold = 0, 0
    return sig


def bt(close, sig, lev, cap, init=500.0, label=""):
    eq = init
    pk = eq
    mdd = 0.0
    trades = []
    pos = 0
    ep = 0.0
    eqc = [eq]

    for i in range(1, len(close)):
        s = int(sig[i])
        c = close[i]
        if s != pos:
            if pos != 0 and ep > 0:
                raw = pos * (c / ep - 1)
                net = raw - 2 * COST
                pnl = eq * cap * lev * net
                eq += pnl
                trades.append(net)
            if s != 0:
                pos, ep = s, c
            else:
                pos, ep = 0, 0.0
        pk = max(pk, eq)
        dd = (eq - pk) / pk * 100 if pk > 0 else 0
        mdd = min(mdd, dd)
        eqc.append(eq)

    if not trades:
        return {"label": label, "sharpe": 0, "cagr": 0, "ret": 0, "mdd": 0,
                "n": 0, "wr": 0, "eq": init}

    bpd = 6 if "4h" in label else 24
    nd = len(close) / bpd
    ny = nd / 365

    eq_arr = np.array(eqc)
    daily = eq_arr[::bpd]
    dr = np.diff(daily) / daily[:-1]
    dr = dr[np.isfinite(dr)]
    sharpe = np.mean(dr) / np.std(dr) * np.sqrt(365) if np.std(dr) > 0 else 0
    wr = np.mean(np.array(trades) > 0) * 100
    cagr = (eq / init) ** (1 / max(ny, 0.1)) - 1

    return {"label": label, "sharpe": round(sharpe, 2), "cagr": round(cagr * 100, 1),
            "ret": round((eq / init - 1) * 100, 1), "mdd": round(mdd, 1),
            "n": len(trades), "wr": round(wr, 1), "eq": round(eq, 0)}


def wf(close, sig, lev, cap, bpd, fold_d=90, min_d=365, label=""):
    mb = min_d * bpd
    fb = fold_d * bpd
    nf = (len(close) - mb) // fb
    if nf < 3:
        return {"pass": "N/A", "ms": 0}
    sharpes = []
    for f in range(nf):
        s = mb + f * fb
        e = min(s + fb, len(close))
        r = bt(close[s:e], sig[s:e], lev, cap, 500, label)
        if r["n"] > 0:
            sharpes.append(r["sharpe"])
    np_ = sum(1 for s in sharpes if s > 0)
    return {"pass": f"{np_}/{len(sharpes)}", "ms": round(np.mean(sharpes), 2) if sharpes else 0}


def combined_bt(strats, lev, init, label):
    max_d = max(len(c) // bpd for c, s, cap, bpd in strats)
    eq = init
    pk = eq
    mdd = 0.0
    nt = 0
    drets = []

    for day in range(1, max_d):
        dpnl = 0.0
        for close, sig, cap, bpd in strats:
            start = day * bpd
            for i in range(start, min(start + bpd, len(close))):
                if i < 1:
                    continue
                if sig[i - 1] != 0:
                    r = sig[i - 1] * (close[i] / close[i - 1] - 1)
                    dpnl += eq * cap * lev * r
                if sig[i] != sig[i - 1] and abs(sig[i] - sig[i - 1]) > 0:
                    dpnl -= eq * cap * lev * COST
                    nt += 1
        eq += dpnl
        drets.append(dpnl / max(eq - dpnl, 1) if eq - dpnl > 0 else 0)
        pk = max(pk, eq)
        dd = (eq - pk) / pk * 100 if pk > 0 else 0
        mdd = min(mdd, dd)

    dr = np.array(drets)
    sharpe = np.mean(dr) / np.std(dr) * np.sqrt(365) if np.std(dr) > 0 else 0
    ny = max_d / 365
    cagr = (eq / init) ** (1 / max(ny, 0.1)) - 1
    wr = np.mean(dr[dr != 0] > 0) * 100 if np.any(dr != 0) else 0

    return {"label": label, "sharpe": round(sharpe, 2), "cagr": round(cagr * 100, 1),
            "ret": round((eq / init - 1) * 100, 1), "mdd": round(mdd, 1),
            "n": nt, "wr": round(wr, 1), "eq": round(eq, 0)}


def main():
    lev = 10.0
    results_all = {}

    for sym in ["BTCUSDT", "ETHUSDT"]:
        print(f"\n{'='*70}")
        print(f"  {sym} — Full Backtest ($500 start, {lev}x leverage)")
        print(f"{'='*70}")

        # 4h
        df4 = load_and_resample(sym, "4h")
        f4 = compute_all_features(sym, df4)
        p4, c4 = load_model_predict(sym, "4h", f4)
        cl4 = df4["close"].values
        s4 = gen_signal(p4, cl4, c4)
        cap4 = 0.08 if "BTC" in sym else 0.06

        ic4, _ = stats.spearmanr(
            p4[~np.isnan(p4)][:-6],
            np.array([cl4[i+6]/cl4[i]-1 for i in range(len(cl4)-6)])[~np.isnan(p4[:-6])]
        )
        print(f"  4h model IC: {ic4:.4f}, bars: {len(df4)}, signals: {int(np.sum(np.abs(np.diff(s4))>0))}")

        r4 = bt(cl4, s4, lev, cap4, label=f"{sym} 4h")
        w4 = wf(cl4, s4, lev, cap4, 6, label=f"{sym} 4h")

        # 1h
        df1 = load_and_resample(sym, "1h")
        f1 = compute_all_features(sym, df1)
        p1, c1 = load_model_predict(sym, "1h", f1)
        cl1 = df1["close"].values
        s1 = gen_signal(p1, cl1, c1)
        cap1 = 0.15 if "BTC" in sym else 0.10

        r1 = bt(cl1, s1, lev, cap1, label=f"{sym} 1h")
        w1 = wf(cl1, s1, lev, cap1, 24, label=f"{sym} 1h")

        # Combined
        rc = combined_bt(
            [(cl1, s1, cap1, 24), (cl4, s4, cap4, 6)],
            lev, 500.0, f"{sym} 1h+4h combined",
        )

        print(f"\n  {'Strategy':<25} {'Sharpe':>7} {'CAGR%':>7} {'Return%':>10} {'MaxDD%':>8} "
              f"{'Trades':>7} {'WR%':>6} {'Final$':>9} {'WF Pass':>8}")
        print(f"  {'-'*90}")
        for r, w in [(r1, w1), (r4, w4), (rc, None)]:
            ws = w["pass"] if w else "N/A"
            print(f"  {r['label']:<25} {r['sharpe']:7.2f} {r['cagr']:7.1f} {r['ret']:10.1f} "
                  f"{r['mdd']:8.1f} {r['n']:7d} {r['wr']:6.1f} {r['eq']:9.0f} {ws:>8}")

        results_all[sym] = {"1h": r1, "4h": r4, "combo": rc}

    # Grand summary
    print(f"\n\n{'='*70}")
    print(f"  GRAND SUMMARY (BTC + ETH Portfolio)")
    print(f"{'='*70}")

    # Full portfolio: BTC 1h + BTC 4h + ETH 1h + ETH 4h
    all_strats = []
    for sym in ["BTCUSDT", "ETHUSDT"]:
        df4 = load_and_resample(sym, "4h")
        f4 = compute_all_features(sym, df4)
        p4, c4 = load_model_predict(sym, "4h", f4)
        s4 = gen_signal(p4, df4["close"].values, c4)
        cap4 = 0.08 if "BTC" in sym else 0.06
        all_strats.append((df4["close"].values, s4, cap4, 6))

        df1 = load_and_resample(sym, "1h")
        f1 = compute_all_features(sym, df1)
        p1, c1 = load_model_predict(sym, "1h", f1)
        s1 = gen_signal(p1, df1["close"].values, c1)
        cap1 = 0.15 if "BTC" in sym else 0.10
        all_strats.append((df1["close"].values, s1, cap1, 24))

    full = combined_bt(all_strats, lev, 500.0, "Full Portfolio (BTC+ETH 1h+4h)")
    print(f"\n  {full['label']}")
    print(f"    Sharpe:      {full['sharpe']}")
    print(f"    CAGR:        {full['cagr']}%")
    print(f"    Total Return: {full['ret']}%")
    print(f"    Max Drawdown: {full['mdd']}%")
    print(f"    Trades:      {full['n']}")
    print(f"    Win Rate:    {full['wr']}%")
    print(f"    $500 → ${full['eq']}")


if __name__ == "__main__":
    main()
