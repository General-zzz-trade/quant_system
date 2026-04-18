"""1D LIVE runner — places real OKX orders for BTC/ETH daily models.

Companion to runner.daily_paper_runner. Same prediction & decision logic,
but actually puts orders on the wire. Designed to run once a day after
UTC 00:00 (the same window as the paper runner).

Safety design (intentionally conservative for first-week deployment):
  1. **--dry-run by default** — no orders placed unless --live is passed.
  2. **Equity-based hard cap** — never more than EQUITY_PCT_CAP of OKX
     equity per single 1D position (default 25%).
  3. **Notional floor** — skip if computed coin qty would be < $50
     notional (waste of fees on a small account).
  4. **Live IC kill switch** — defers to data/runtime/symbol_pause_state.json;
     if a 1D runner is paused (BTCUSDT_1d / ETHUSDT_1d), skip.
  5. **Single-position invariant** — refuses to OPEN if exchange already
     has any 1D-tagged position for that symbol (close it first manually).
  6. **Audit join** — writes to data/runtime/decision_audit_okx.jsonl
     using the SAME format as the existing 1h/4h runners, with
     `runner_key="BTCUSDT_1d"` etc., so all monitoring tools (live IC
     kill switch, daily PnL, decision audit) pick it up automatically.

How to flip live on Day 4:
  1. Edit /etc/systemd/system/daily-live-runner.service (sudo) and remove
     the `--dry-run` flag from ExecStart.
  2. sudo systemctl daemon-reload && sudo systemctl enable --now daily-live-runner.timer

Until then, this script is harmless even if cron-fires it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from runner.daily_paper_runner import (  # noqa: E402
    _build_features,
    _decide_action,
    _fetch_recent_1d_bars,
    _load_model_bundle,
    _predict,
)
from monitoring.live_ic_killswitch import is_runner_paused  # noqa: E402

logger = logging.getLogger(__name__)

# Audit lands in the existing OKX live audit file so monitoring picks it up.
AUDIT_PATH = Path("data/runtime/decision_audit_okx.jsonl")
LIVE_STATE_PATH = Path("data/runtime/daily_live_state.json")

# Conservative sizing: 25% of equity in notional terms per position.
# At $377 equity that's a max ~$94 notional position — small enough to
# hand-baby for the first 2 weeks while we build confidence.
EQUITY_PCT_CAP = float(os.environ.get("DAILY_LIVE_EQUITY_PCT", "0.25"))
MIN_NOTIONAL_USD = float(os.environ.get("DAILY_LIVE_MIN_NOTIONAL", "50"))


def _load_live_state() -> dict:
    if LIVE_STATE_PATH.exists():
        try:
            return json.loads(LIVE_STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_live_state(state: dict) -> None:
    LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def _audit_write(record: dict) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    record["ts"] = time.time()
    record["venue"] = "okx"
    with AUDIT_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _make_okx_adapter():
    from dotenv import load_dotenv

    load_dotenv("/quant_system/.env")
    from execution.adapters.okx.adapter import OkxAdapter
    from execution.adapters.okx.config import OkxConfig

    cfg = OkxConfig(
        api_key=os.environ.get("OKX_API_KEY", ""),
        api_secret=os.environ.get("OKX_API_SECRET", ""),
        passphrase=os.environ.get("OKX_API_PASSPHRASE", ""),
        base_url=os.environ.get("OKX_BASE_URL", "https://www.okx.com"),
    )
    adapter = OkxAdapter(cfg)
    adapter.connect()  # populates instrument metadata cache
    return adapter


def _get_equity_usdt(adapter) -> float:
    snap = adapter.get_balances()
    for b in snap.balances:
        if b.asset == "USDT":
            return float(b.free + b.locked)
    return 0.0


def _existing_position_qty(adapter, internal_symbol: str) -> float:
    """Returns signed coin qty (positive long, negative short)."""
    for p in adapter.get_positions():
        if p.symbol == internal_symbol and not p.is_flat:
            return float(p.qty)
    return 0.0


def _compute_size_qty(equity: float, price: float) -> float:
    """Notional in USD = equity * EQUITY_PCT_CAP. Coin qty = notional / price."""
    notional = equity * EQUITY_PCT_CAP
    if notional < MIN_NOTIONAL_USD:
        return 0.0
    return notional / max(price, 1e-9)


def _execute(symbol: str, side: str, qty_coin: float, adapter, *, dry_run: bool) -> dict:
    """Place a market order via OkxAdapter. side='buy'|'sell', qty in coin units."""
    if dry_run:
        return {"status": "dry_run", "would_send": {"symbol": symbol, "side": side, "qty": qty_coin}}
    try:
        return adapter.send_market_order(symbol, side, qty_coin)
    except Exception as e:
        logger.exception("send_market_order failed")
        return {"status": "error", "msg": str(e)}


def run_symbol(symbol: str, model_dir: str, macro: pd.DataFrame, *, live: bool):
    runner_key = f"{symbol}_1d"

    if is_runner_paused(runner_key):
        msg = f"{runner_key}: paused by live IC kill switch — skipping"
        logger.warning(msg)
        return {"symbol": symbol, "skipped": True, "msg": msg}

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

    state = _load_live_state()
    sym_state = state.get(runner_key, {
        "signal": 0, "entry_price": 0.0, "entry_date": None,
        "bars_held": 0, "last_bar_date": None,
        "last_order": None,
    })

    if sym_state.get("last_bar_date") == latest_date:
        msg = f"{runner_key}: already processed {latest_date}"
        logger.info(msg)
        return {"symbol": symbol, "skipped": True, "msg": msg}

    current_signal = sym_state["signal"]
    new_signal, reason = _decide_action(z, dz, current_signal)

    adapter = _make_okx_adapter()
    equity = _get_equity_usdt(adapter)
    existing_qty = _existing_position_qty(adapter, symbol)

    audit = {
        "type": "daily_live_signal",
        "symbol": symbol,
        "runner_key": runner_key,
        "date": latest_date,
        "close": latest_close,
        "prediction": pred,
        "z_score": z,
        "deadzone": dz,
        "current_signal": current_signal,
        "new_signal": new_signal,
        "reason": reason,
        "equity_usdt": equity,
        "existing_qty": existing_qty,
        "live_mode": live,
    }

    action_taken = "FLAT"
    order_resp = None

    # Open new position
    if current_signal == 0 and new_signal != 0:
        # Refuse to open if exchange already has a position for this symbol
        # (could be from 1h/4h runners; safer to manually disambiguate)
        if abs(existing_qty) > 1e-8:
            audit["skip_reason"] = "exchange already holds position"
            audit["action"] = "SKIPPED_EXISTING_POS"
            _audit_write(audit)
            sym_state["last_bar_date"] = latest_date
            state[runner_key] = sym_state
            _save_live_state(state)
            return {"symbol": symbol, "audit": audit, "state": sym_state}

        qty_coin = _compute_size_qty(equity, latest_close)
        if qty_coin <= 0:
            audit["skip_reason"] = f"qty {qty_coin} below MIN_NOTIONAL_USD={MIN_NOTIONAL_USD}"
            audit["action"] = "SKIPPED_TOO_SMALL"
            _audit_write(audit)
            sym_state["last_bar_date"] = latest_date
            state[runner_key] = sym_state
            _save_live_state(state)
            return {"symbol": symbol, "audit": audit, "state": sym_state}

        side = "buy" if new_signal == 1 else "sell"
        order_resp = _execute(symbol, side, qty_coin, adapter, dry_run=not live)
        sym_state["signal"] = new_signal
        sym_state["entry_price"] = latest_close
        sym_state["entry_date"] = latest_date
        sym_state["bars_held"] = 0
        sym_state["last_order"] = {"side": side, "qty": qty_coin, "resp": order_resp}
        action_taken = "OPEN"
        audit["intended_qty"] = qty_coin
        audit["order_response"] = order_resp

    # Hold or close
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
            # Close: opposite side of existing exchange position
            if abs(existing_qty) < 1e-8:
                audit["close_skip"] = "exchange already flat"
                action_taken = "ALREADY_FLAT"
            else:
                close_side = "sell" if existing_qty > 0 else "buy"
                close_qty = abs(existing_qty)
                order_resp = _execute(symbol, close_side, close_qty, adapter, dry_run=not live)
                action_taken = "CLOSE"
                audit["close_reason"] = close_reason
                audit["intended_qty"] = close_qty
                audit["order_response"] = order_resp
            sym_state["signal"] = 0
            sym_state["entry_price"] = 0.0
            sym_state["entry_date"] = None
            sym_state["bars_held"] = 0
        else:
            action_taken = "HOLD"

    audit["action"] = action_taken
    sym_state["last_bar_date"] = latest_date
    state[runner_key] = sym_state

    _audit_write(audit)
    _save_live_state(state)

    return {"symbol": symbol, "audit": audit, "state": sym_state}


def main():
    parser = argparse.ArgumentParser(description="1D LIVE runner — places real OKX orders")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"],
                        help="Symbols to trade. Default: BTCUSDT only "
                             "(ETH disabled until holdout IC recovers)")
    parser.add_argument("--live", action="store_true",
                        help="REQUIRED to actually place orders. Without "
                             "this flag the runner does everything except "
                             "send the order (dry-run).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    macro = pd.read_csv("/quant_system/data_files/cross_market_daily.csv")
    macro["date"] = pd.to_datetime(macro["date"]).dt.date

    print("=" * 72)
    mode = "LIVE" if args.live else "DRY-RUN"
    print(f"1D {mode} runner @ {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"  EQUITY_PCT_CAP={EQUITY_PCT_CAP*100:.0f}%  "
          f"MIN_NOTIONAL_USD=${MIN_NOTIONAL_USD:.0f}")
    print("=" * 72)

    for sym in args.symbols:
        model_dir = f"/quant_system/models_v8/{sym}_1d"
        if not Path(model_dir).exists():
            print(f"  {sym}: no model — skipping")
            continue
        try:
            r = run_symbol(sym, model_dir, macro, live=args.live)
            if r.get("skipped"):
                print(f"  {r['msg']}")
                continue
            a = r["audit"]
            s = r["state"]
            print(f"  {sym}_1d  date={a['date']}  close=${a['close']:,.2f}  "
                  f"z={a['z_score']:+.2f}  dz={a['deadzone']}  "
                  f"action={a.get('action')}  reason={a.get('reason')}")
            print(f"    equity=${a['equity_usdt']:,.2f}  "
                  f"existing_qty={a['existing_qty']:+.4f}  mode={mode}")
            if a.get("action") == "OPEN":
                side = "LONG" if a["new_signal"] > 0 else "SHORT"
                print(f"    → OPENED {side} qty={a['intended_qty']:.4f} "
                      f"resp={a.get('order_response')}")
            elif a.get("action") == "CLOSE":
                print(f"    → CLOSED qty={a['intended_qty']:.4f} "
                      f"reason={a['close_reason']} "
                      f"resp={a.get('order_response')}")
            elif a.get("action") == "HOLD":
                print(f"    → HOLD ({s['bars_held']}d) "
                      f"side={'long' if s['signal']>0 else 'short'} "
                      f"entry=${s['entry_price']:,.2f}")
            elif a.get("action", "").startswith("SKIPPED"):
                print(f"    → SKIPPED: {a.get('skip_reason')}")
        except Exception:
            logger.exception("daily live runner failed for %s", sym)


if __name__ == "__main__":
    main()
