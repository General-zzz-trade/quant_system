"""1D paper-trading runner — observes BTC/ETH daily models without live orders.

Runs once a day after UTC 00:00 (the previous day's close), pulls the latest
1D bar + macro features, queries the trained 1D ensemble, and simulates
trades according to the same dz/min_hold/max_hold rules as the live runner.
Does NOT place real orders — purely observational.

Outputs:
  data/runtime/daily_paper_audit.jsonl   per-day signal + position state
  data/runtime/daily_paper_state.json    persistent state
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

logger = logging.getLogger(__name__)

AUDIT_PATH = Path("data/runtime/daily_paper_audit.jsonl")
STATE_PATH = Path("data/runtime/daily_paper_state.json")
NOTIONAL = 500.0
COST_BPS = 14


def _read_bundle(path):
    # Trusted local artefact from our own retrain pipeline.
    import pickle as _pkl  # noqa: S403

    with open(path, "rb") as f:
        return _pkl.load(f)  # noqa: S301


def _load_model_bundle(model_dir):
    lgbm = _read_bundle(f"{model_dir}/lgbm_v8.pkl")
    xgb_m = _read_bundle(f"{model_dir}/xgb_v8.pkl")
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    return lgbm, xgb_m, cfg


def _fetch_recent_1d_bars(symbol: str, n_days: int = 200) -> pd.DataFrame:
    from scripts.daily_poc import resample_1h_to_1d

    df_1h = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    df_1d = resample_1h_to_1d(df_1h)
    return df_1d.tail(n_days).reset_index(drop=True)


def _build_features(df_1d, macro):
    from scripts.daily_poc import compute_1d_features

    feat_df, feature_names = compute_1d_features(df_1d, macro)
    X = feat_df[feature_names].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_names


def _predict(lgbm_bundle, xgb_bundle, X_row):
    import xgboost as xgb

    p_l = float(lgbm_bundle["model"].predict(X_row.reshape(1, -1))[0])
    p_x = float(xgb_bundle["model"].predict(xgb.DMatrix(X_row.reshape(1, -1)))[0])
    return 0.5 * p_l + 0.5 * p_x


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def _append_audit(record: dict) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    record["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with AUDIT_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _decide_action(z, dz, current_signal):
    if current_signal != 0:
        if current_signal * z < -0.3:
            return 0, "z_reversal"
        if abs(z) < 0.2:
            return 0, "z_decay"
        return current_signal, "hold"
    if z > dz:
        return 1, "long_entry"
    if z < -dz:
        return -1, "short_entry"
    return 0, "no_signal"


def run_symbol(symbol: str, model_dir: str, macro: pd.DataFrame, dry_run: bool = False):
    lgbm, xgb_m, cfg = _load_model_bundle(model_dir)
    feature_names_cfg = cfg["features"]
    dz = cfg["deadzone"]
    min_hold = cfg["min_hold"]
    max_hold = cfg["max_hold"]
    pred_mean = cfg["zscore_pred_mean"]
    pred_std = max(cfg["zscore_pred_std"], 1e-10)

    df_1d = _fetch_recent_1d_bars(symbol, n_days=200)
    X_all, feature_names = _build_features(df_1d, macro)

    latest_idx = len(df_1d) - 1
    latest_date = str(df_1d["date"].iloc[latest_idx])
    latest_close = float(df_1d["close"].iloc[latest_idx])

    X_row = np.zeros(len(feature_names_cfg), dtype=np.float64)
    for j, name in enumerate(feature_names_cfg):
        if name in feature_names:
            X_row[j] = X_all[latest_idx, feature_names.index(name)]

    pred = _predict(lgbm, xgb_m, X_row)
    z = (pred - pred_mean) / pred_std

    state = _load_state()
    sym_state = state.get(symbol, {
        "signal": 0, "entry_price": 0.0, "entry_date": None,
        "bars_held": 0, "cum_pnl": 0.0, "trades": 0, "wins": 0,
        "last_bar_date": None,
    })

    if sym_state.get("last_bar_date") == latest_date:
        msg = f"{symbol}: already processed {latest_date}"
        logger.info(msg)
        return {"symbol": symbol, "skipped": True, "msg": msg}

    current_signal = sym_state["signal"]
    new_signal, reason = _decide_action(z, dz, current_signal)

    audit_record = {
        "type": "paper_signal",
        "symbol": symbol,
        "model_dir": model_dir,
        "date": latest_date,
        "close": latest_close,
        "prediction": pred,
        "z_score": z,
        "deadzone": dz,
        "current_signal": current_signal,
        "new_signal": new_signal,
        "reason": reason,
    }

    if current_signal == 0 and new_signal != 0:
        sym_state["signal"] = new_signal
        sym_state["entry_price"] = latest_close
        sym_state["entry_date"] = latest_date
        sym_state["bars_held"] = 0
        audit_record["action"] = "OPEN"
    elif current_signal != 0:
        sym_state["bars_held"] += 1
        force_close = False
        close_reason = ""
        if sym_state["bars_held"] >= max_hold:
            force_close = True
            close_reason = "max_hold"
        elif sym_state["bars_held"] >= min_hold:
            if reason in ("z_reversal", "z_decay"):
                force_close = True
                close_reason = reason
        if force_close:
            ep = sym_state["entry_price"]
            pnl_pct = current_signal * (latest_close - ep) / ep
            gross = pnl_pct * NOTIONAL
            cost = COST_BPS / 10000 * NOTIONAL
            net = gross - cost
            sym_state["cum_pnl"] += net
            sym_state["trades"] += 1
            if net > 0:
                sym_state["wins"] += 1
            audit_record.update({
                "action": "CLOSE",
                "close_reason": close_reason,
                "entry_price": ep,
                "exit_price": latest_close,
                "bars_held": sym_state["bars_held"],
                "gross_pnl": gross,
                "net_pnl": net,
                "cum_pnl": sym_state["cum_pnl"],
                "trades": sym_state["trades"],
                "win_rate": sym_state["wins"] / sym_state["trades"] * 100,
            })
            sym_state["signal"] = 0
            sym_state["entry_price"] = 0.0
            sym_state["entry_date"] = None
            sym_state["bars_held"] = 0
        else:
            audit_record["action"] = "HOLD"
            audit_record["bars_held"] = sym_state["bars_held"]
    else:
        audit_record["action"] = "FLAT"

    sym_state["last_bar_date"] = latest_date
    state[symbol] = sym_state

    if not dry_run:
        _save_state(state)
        _append_audit(audit_record)

    return {"symbol": symbol, "audit": audit_record, "state": sym_state}


def main():
    parser = argparse.ArgumentParser(description="1D paper-trading runner")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write state or audit, just print")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    macro = pd.read_csv("/quant_system/data_files/cross_market_daily.csv")
    macro["date"] = pd.to_datetime(macro["date"]).dt.date

    print("=" * 72)
    print(f"1D paper runner @ {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print("=" * 72)

    for sym in args.symbols:
        model_dir = f"/quant_system/models_v8/{sym}_1d"
        if not Path(model_dir).exists():
            print(f"  {sym}: model dir not found, skipping")
            continue
        try:
            r = run_symbol(sym, model_dir, macro, dry_run=args.dry_run)
            if r.get("skipped"):
                print(f"  {r['msg']}")
                continue
            a = r["audit"]
            s = r["state"]
            print(f"  {sym}  date={a['date']}  close=${a['close']:,.2f}  "
                  f"pred={a['prediction']:+.5f}  z={a['z_score']:+.2f}  "
                  f"dz={a['deadzone']}  action={a.get('action')}  "
                  f"reason={a.get('reason')}")
            if a.get("action") == "OPEN":
                side = "LONG" if a["new_signal"] > 0 else "SHORT"
                print(f"    → OPENED {side} @ ${a['close']:,.2f}")
            elif a.get("action") == "CLOSE":
                print(f"    → CLOSED ({a['close_reason']}): "
                      f"net_pnl=${a['net_pnl']:+.2f} held={a['bars_held']}d  "
                      f"cum=${a['cum_pnl']:+.2f}  WR={a['win_rate']:.0f}%")
            elif a.get("action") == "HOLD":
                side = "long" if s["signal"] > 0 else "short"
                print(f"    → HOLD ({a['bars_held']}d)  "
                      f"entry=${s['entry_price']:,.2f} side={side}  "
                      f"cum=${s['cum_pnl']:+.2f}")
        except Exception:
            logger.exception("paper runner failed for %s", sym)


if __name__ == "__main__":
    main()
