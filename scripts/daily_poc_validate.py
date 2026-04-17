#!/usr/bin/env python3
"""Phase 1 stability validation for 1D POC.

Extends scripts/daily_poc.py:
  - Monthly PnL breakdown (find any bad stretches)
  - Bootstrap p-value for Sharpe > 0 (5000 resamples)
  - Feature importance (last-fold LGBM gain)
  - Consecutive-loss analysis (max drawdown in trade count)
  - Rolling 90-day Sharpe (is it stable over time or just early luck?)

Run: python3 -m scripts.daily_poc_validate
"""
from __future__ import annotations
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from scripts.daily_poc import (  # noqa: E402
    COST_MAKER_RT,
    COST_TAKER_RT,
    MIN_HOLD,
    MAX_HOLD,
    compute_1d_features,
    resample_1h_to_1d,
    run_backtest,
    walk_forward_predict,
    Trade,
)

DZ_BEST = {"BTCUSDT": 1.75, "ETHUSDT": 1.75}  # pick high-Sharpe dz for each


def monthly_breakdown(trades: list[Trade], dates: np.ndarray, label: str) -> dict:
    """Aggregate trade PnL by calendar month."""
    monthly = {}
    for t in trades:
        exit_date = dates[min(t.exit_bar, len(dates) - 1)]
        month = pd.Timestamp(exit_date).strftime("%Y-%m")
        if month not in monthly:
            monthly[month] = {"net": 0.0, "trades": 0, "wins": 0, "longs": 0, "shorts": 0}
        monthly[month]["net"] += t.net_pnl
        monthly[month]["trades"] += 1
        if t.net_pnl > 0:
            monthly[month]["wins"] += 1
        if t.direction > 0:
            monthly[month]["longs"] += 1
        else:
            monthly[month]["shorts"] += 1

    print(f"\n  [{label}] Monthly breakdown:")
    print(f"  {'Month':>8} {'Trades':>7} {'L/S':>6} {'WR':>6} {'Net$':>9} {'Cum$':>9}")
    print("  " + "-" * 52)
    cum = 0.0
    pos_months = 0
    neg_months = 0
    zero_months = 0
    worst = ("", 0.0)
    best = ("", 0.0)
    for month in sorted(monthly.keys()):
        m = monthly[month]
        cum += m["net"]
        wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        if m["net"] > 0:
            pos_months += 1
        elif m["net"] < 0:
            neg_months += 1
        if m["trades"] == 0:
            zero_months += 1
        if m["net"] < worst[1]:
            worst = (month, m["net"])
        if m["net"] > best[1]:
            best = (month, m["net"])
        print(f"  {month:>8} {m['trades']:>7} "
              f"{m['longs']:>2}/{m['shorts']:<2}  "
              f"{wr:>5.0f}% {m['net']:>+8.1f} {cum:>+8.1f}")
    total_months = len(monthly)
    print(f"  {'─'*52}")
    print(f"  Months: {total_months} total  "
          f"({pos_months} win / {neg_months} loss / {zero_months} no-trade)")
    print(f"  Best : {best[0]} ${best[1]:+.1f}")
    print(f"  Worst: {worst[0]} ${worst[1]:+.1f}")
    return {
        "months": total_months,
        "positive": pos_months,
        "negative": neg_months,
        "worst_month": worst,
    }


def bootstrap_sharpe(net_pnls: np.ndarray, n_boot: int = 5000, seed: int = 42) -> dict:
    """Bootstrap Sharpe with resampling. Returns distribution + p(Sharpe>0)."""
    rng = np.random.default_rng(seed)
    n = len(net_pnls)
    if n < 2:
        return {"p_positive": 0.0, "ci95": (0, 0)}

    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resample = net_pnls[idx]
        mu = resample.mean()
        sd = resample.std(ddof=1)
        sharpes[b] = mu / max(sd, 1e-10)

    # Annualize using avg observed hold = ~2 days → 182 trades/year would be aggressive;
    # we use raw (per-trade) sharpe for the p-value check.
    p_pos = float((sharpes > 0).mean())
    ci95 = (float(np.quantile(sharpes, 0.025)),
            float(np.quantile(sharpes, 0.975)))
    return {"p_positive": p_pos, "ci95": ci95, "median": float(np.median(sharpes))}


def consecutive_loss_stats(trades: list[Trade]) -> dict:
    """Max consecutive losers, max dd by trade count."""
    if not trades:
        return {}
    nets = [t.net_pnl for t in trades]
    max_consec_loss = 0
    cur_streak = 0
    for n in nets:
        if n < 0:
            cur_streak += 1
            max_consec_loss = max(max_consec_loss, cur_streak)
        else:
            cur_streak = 0

    # Equity curve from trades + drawdown points
    eq = 10000.0
    peak = eq
    dd_start = None
    max_dd = 0.0
    max_dd_len = 0
    dd_len = 0
    for t in trades:
        eq += t.net_pnl
        if eq >= peak:
            peak = eq
            dd_len = 0
            dd_start = None
        else:
            dd = (peak - eq) / peak
            if dd_start is None:
                dd_start = t.entry_bar
            dd_len += 1
            if dd > max_dd:
                max_dd = dd
                max_dd_len = dd_len

    return {
        "max_consec_loss": max_consec_loss,
        "max_dd_pct": max_dd * 100,
        "max_dd_len": max_dd_len,  # in trade count
    }


def feature_importance(X, y, feature_names, top_k: int = 15):
    """Train LGBM on last fold and report feature gain."""
    import lightgbm as lgb
    valid = ~np.isnan(y)
    X_v = X[valid]
    y_v = y[valid]
    if len(X_v) < 100:
        return []
    dtrain = lgb.Dataset(X_v, label=y_v)
    model = lgb.train(
        {"objective": "regression", "metric": "rmse",
         "num_leaves": 31, "learning_rate": 0.03, "verbosity": -1,
         "feature_fraction": 0.85, "bagging_fraction": 0.85,
         "bagging_freq": 3, "min_data_in_leaf": 20},
        dtrain, num_boost_round=200,
    )
    gain = model.feature_importance(importance_type="gain")
    imp = sorted(zip(feature_names, gain), key=lambda x: -x[1])
    return imp[:top_k]


def rolling_sharpe(trades: list[Trade], dates: np.ndarray, window_days: int = 180) -> list:
    """Compute rolling Sharpe in 6-month windows (by trade count)."""
    if len(trades) < 10:
        return []
    rolls = []
    # Sort by entry_bar
    sorted_t = sorted(trades, key=lambda t: t.entry_bar)
    # Use per-trade sharpe (not annualized) over trailing 12-trade windows
    nets = np.array([t.net_pnl for t in sorted_t])
    W = max(10, min(20, len(sorted_t) // 3))
    for i in range(W, len(sorted_t)):
        window = nets[i - W:i]
        if window.std(ddof=1) > 0:
            sh = window.mean() / window.std(ddof=1)
            exit_bar = sorted_t[i].exit_bar
            exit_date = dates[min(exit_bar, len(dates) - 1)]
            rolls.append((pd.Timestamp(exit_date).strftime("%Y-%m"), sh))
    return rolls


def validate_symbol(symbol: str, macro: pd.DataFrame):
    print("=" * 72)
    print(f"Phase 1 validation: {symbol} 1D")
    print("=" * 72)

    df_1h = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    df_1d = resample_1h_to_1d(df_1h)
    macro_start = macro["date"].min()
    df_1d = df_1d[df_1d["date"] >= macro_start].reset_index(drop=True)

    feat_df, feature_names = compute_1d_features(df_1d, macro)
    closes = df_1d["close"].values.astype(np.float64)

    log_rets = np.diff(np.log(closes), prepend=np.nan)
    target = np.roll(log_rets, -1)
    target[-1] = np.nan

    X_all = feat_df[feature_names].values.astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Features: {len(feature_names)}, Bars: {len(df_1d)}")

    preds, n_folds = walk_forward_predict(X_all, target, df_1d["date"].values, 6, 3)
    valid = ~np.isnan(preds)

    oos_preds = preds[valid]
    oos_closes = closes[valid]
    oos_dates = df_1d["date"].values[valid]

    dz = DZ_BEST[symbol]
    trades_maker = run_backtest(oos_preds, oos_closes, dz, COST_MAKER_RT,
                                 long_only=False, min_hold=MIN_HOLD, max_hold=MAX_HOLD)
    trades_taker = run_backtest(oos_preds, oos_closes, dz, COST_TAKER_RT,
                                 long_only=False, min_hold=MIN_HOLD, max_hold=MAX_HOLD)

    print(f"  Best dz: {dz}")
    print(f"  Trades:  maker={len(trades_maker)}  taker={len(trades_taker)}")

    # Monthly breakdown (maker cost = best-case)
    monthly_breakdown(trades_maker, oos_dates, "maker")
    monthly_breakdown(trades_taker, oos_dates, "taker")

    # Bootstrap Sharpe
    nets_m = np.array([t.net_pnl for t in trades_maker])
    nets_t = np.array([t.net_pnl for t in trades_taker])
    bs_m = bootstrap_sharpe(nets_m)
    bs_t = bootstrap_sharpe(nets_t)

    print("\n  Bootstrap Sharpe (per-trade, 5000 resamples):")
    print(f"    maker: P(Sharpe>0) = {bs_m['p_positive']*100:.1f}%  "
          f"95% CI = [{bs_m['ci95'][0]:+.3f}, {bs_m['ci95'][1]:+.3f}]  "
          f"median = {bs_m['median']:+.3f}")
    print(f"    taker: P(Sharpe>0) = {bs_t['p_positive']*100:.1f}%  "
          f"95% CI = [{bs_t['ci95'][0]:+.3f}, {bs_t['ci95'][1]:+.3f}]  "
          f"median = {bs_t['median']:+.3f}")

    # Consecutive loss
    cls_m = consecutive_loss_stats(trades_maker)
    print("\n  Risk metrics (maker):")
    print(f"    Max consecutive losses: {cls_m.get('max_consec_loss', 0)}")
    print(f"    Max DD: {cls_m.get('max_dd_pct', 0):.2f}%  "
          f"(over {cls_m.get('max_dd_len', 0)} trades)")

    # Feature importance
    print("\n  Feature importance (top 15, trained on full sample):")
    imp = feature_importance(X_all, target, feature_names, top_k=15)
    for i, (name, gain) in enumerate(imp, 1):
        print(f"    {i:>2}. {name:<30s} gain={gain:>10.1f}")

    # Rolling Sharpe
    rolls = rolling_sharpe(trades_maker, oos_dates)
    if rolls:
        print("\n  Rolling Sharpe (12-trade window, last 10 snapshots):")
        for month, sh in rolls[-10:]:
            bar = "█" * max(0, min(int((sh + 1) * 10), 20))
            print(f"    {month}  {sh:>+6.3f}  {bar}")


def main():
    macro = pd.read_csv("/quant_system/data_files/cross_market_daily.csv")
    macro["date"] = pd.to_datetime(macro["date"]).dt.date

    for sym in ("BTCUSDT", "ETHUSDT"):
        validate_symbol(sym, macro)
        print()


if __name__ == "__main__":
    main()
