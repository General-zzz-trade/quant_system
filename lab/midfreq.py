"""Mid-frequency feasibility: panel, causal features, candidate scores and a long/short simulation.

Protocol: research/protocols/2026-11-14-midfreq-feasibility.md. Data: a snapshot written by
data/downloads/binance_archive.py (5-minute bars of the 15 most liquid USDT perpetuals of 2022-Q4).

Conventions: bar ``i`` opens at ``times[i]`` and is known at ``times[i] + BAR``. A decision made when bar
``i`` closes uses bars ``<= i`` only and is filled at the open of bar ``i + 1``. Missing bars are NaN and a
symbol with a NaN anywhere in a feature window has no score at that decision.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BAR = 300_000
DAY = 86_400_000
BARS_PER_DAY = 288
LEGS = 3                                    # long the top 3, short the bottom 3
FALLBACK_RULES = {"step": 0.0, "min_notional": 5.0}
RIDGE_FEATURES = ("ret_15m", "ret_1h", "ret_4h", "ret_24h", "flow_15m", "flow_1h", "flow_4h", "volume_ratio",
                  "vol_ratio", "range_position")


@dataclass
class Panel:
    symbols: list[str]
    times: np.ndarray                       # open time of each 5-minute bar, contiguous
    open: np.ndarray                        # [bars, symbols]
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote: np.ndarray                       # quote volume
    taker: np.ndarray                       # taker-buy quote volume
    funding: list[tuple[np.ndarray, np.ndarray]]   # per symbol: settlement times, rates
    rules: list[dict]                       # per symbol: step, min_notional


def load_panel(data_dir: Path, *, end_ms: int | None = None) -> Panel:
    """Every symbol of the snapshot on one contiguous 5-minute grid; rows at or after ``end_ms`` are dropped."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    symbols = [u["symbol"] for u in manifest["universe"]]
    listed = json.loads((data_dir / "exchange_info.json").read_text())
    frames, funding = [], []
    for s in symbols:
        with gzip.open(data_dir / f"{s}_5m.csv.gz", "rt") as stream:
            frame = pd.read_csv(stream)
        with gzip.open(data_dir / f"{s}_funding.csv.gz", "rt") as stream:
            fund = pd.read_csv(stream)
        if end_ms is not None:
            frame, fund = frame[frame["open_time"] + BAR <= end_ms], fund[fund["funding_time"] < end_ms]
        frames.append(frame.set_index("open_time"))
        funding.append((fund["funding_time"].to_numpy(dtype=np.int64), fund["rate"].to_numpy(float)))
    first = min(int(f.index[0]) for f in frames)
    last = max(int(f.index[-1]) for f in frames)
    times = np.arange(first, last + BAR, BAR, dtype=np.int64)

    def grid(column: str) -> np.ndarray:
        return np.column_stack([f[column].reindex(times).to_numpy(float) for f in frames])

    rules = [{"step": float(listed[s]["step"]), "min_notional": float(listed[s]["min_notional"])}
             if s in listed else dict(FALLBACK_RULES) for s in symbols]
    return Panel(symbols, times, grid("open"), grid("high"), grid("low"), grid("close"), grid("quote_volume"),
                 grid("taker_buy_quote_volume"), funding, rules)


# -- features (each row uses bars up to and including that row) ----------------------------------------
def features(panel: Panel) -> dict[str, np.ndarray]:
    close = pd.DataFrame(panel.close)
    logp = np.log(close)
    r = logp.diff()
    sigma = r.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY).std()          # per 5-minute bar, last 24 hours
    quote, taker = pd.DataFrame(panel.quote), pd.DataFrame(panel.taker)
    out: dict[str, np.ndarray] = {"sigma": sigma.to_numpy()}
    for name, bars in (("15m", 3), ("1h", 12), ("4h", 48), ("24h", BARS_PER_DAY)):
        out[f"raw_ret_{name}"] = (logp - logp.shift(bars)).to_numpy()
        out[f"ret_{name}"] = out[f"raw_ret_{name}"] / (out["sigma"] * math.sqrt(bars))
    for name, bars in (("15m", 3), ("1h", 12), ("4h", 48)):
        q = quote.rolling(bars, min_periods=bars).sum()
        t = taker.rolling(bars, min_periods=bars).sum()
        out[f"flow_{name}"] = ((2 * t - q) / q.where(q > 0)).to_numpy()
    q_1h = quote.rolling(12, min_periods=12).sum()
    q_24h = quote.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY).sum() / 24.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out["volume_ratio"] = np.log((q_1h / q_24h.where(q_24h > 0)).where(lambda x: x > 0)).to_numpy()
    out["vol_ratio"] = (r.rolling(12, min_periods=12).std() / sigma.where(sigma > 0)).to_numpy()
    top = pd.DataFrame(panel.high).rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY).max()
    bottom = pd.DataFrame(panel.low).rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY).min()
    span = (top - bottom).where(lambda x: x > 0)
    out["range_position"] = ((close - bottom) / span).to_numpy()
    return out


def decision_rows(panel: Panel, horizon_bars: int, *, start: int, end: int) -> np.ndarray:
    """Bars whose close lands on the horizon clock inside [start, end) and whose fill bar and exit exist."""
    closes = panel.times + BAR
    rows = np.flatnonzero((closes % (horizon_bars * BAR) == 0) & (closes >= start) & (closes < end))
    return rows[rows + 1 + horizon_bars < len(panel.times)]


def forward_return(panel: Panel, rows: np.ndarray, horizon_bars: int) -> np.ndarray:
    """Return from the fill (open of the next bar) to the open ``horizon_bars`` later."""
    return panel.open[rows + 1 + horizon_bars] / panel.open[rows + 1] - 1.0


def demean(x: np.ndarray) -> np.ndarray:
    """Subtract the cross-sectional mean of each row over the symbols that have a value."""
    finite = np.isfinite(x)
    count = finite.sum(axis=1, keepdims=True)
    total = np.where(finite, x, 0.0).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return x - np.where(count > 0, total / np.maximum(count, 1), np.nan)


def ridge_inputs(panel: Panel, feats: dict, rows: np.ndarray, horizon_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """(x [decisions, symbols, features], y [decisions, symbols]); both cross-sectionally demeaned."""
    x = np.stack([demean(feats[name][rows]) for name in RIDGE_FEATURES], axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = forward_return(panel, rows, horizon_bars) / (feats["sigma"][rows] * math.sqrt(horizon_bars))
    return x, np.clip(demean(y), -5, 5)


def ridge_scores(panel: Panel, feats: dict, rows: np.ndarray, horizon_bars: int, *, first_test: int, block: int,
                 limit: int, min_train: int = 1000) -> tuple[np.ndarray, list[dict]]:
    """Purged expanding walk-forward predictions for ``rows``; a block trains on labels that ended before it.

    Training uses every decision from the first bar of the panel (the warm-up months before the first
    test block included), so the first block already has a model.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    wanted = rows
    rows = decision_rows(panel, horizon_bars, start=int(panel.times[0]), end=limit)
    x, y = ridge_inputs(panel, feats, rows, horizon_bars)
    decided = panel.times[rows] + BAR
    label_end = panel.times[rows + 1 + horizon_bars]
    usable = np.isfinite(x).all(axis=2) & np.isfinite(y)
    has_x = np.isfinite(x).all(axis=2)
    scores, blocks = np.full(y.shape, np.nan), []
    a = first_test
    while a < limit:
        b = min(a + block, limit)
        train, test = label_end < a, (decided >= a) & (decided < b)
        x_train, y_train = x[train][usable[train]], y[train][usable[train]]
        if len(x_train) >= min_train and test.any():
            mu, sd = x_train.mean(axis=0), x_train.std(axis=0)
            sd[sd == 0] = 1.0
            model = Ridge(alpha=float(len(x_train))).fit(np.clip((x_train - mu) / sd, -5, 5), y_train)
            block_x = x[test]
            predicted = np.full(block_x.shape[:2], np.nan)
            ok = has_x[test]
            predicted[ok] = model.predict(np.clip((block_x[ok] - mu) / sd, -5, 5))
            scores[test] = predicted
            scored = usable[test]
            ic = float(spearmanr(predicted[scored], y[test][scored]).statistic) if scored.sum() >= 30 else float("nan")
            blocks.append({"test_start": int(a), "train_rows": int(len(x_train)), "ic": ic})
        a = b
    return scores[np.searchsorted(rows, wanted)], blocks


def rule_scores(feats: dict, rows: np.ndarray, rule: str) -> np.ndarray:
    """Higher score = long. Reversal is long the lowest past return; momentum and flow are long the highest."""
    if rule == "xs_rev_15m":
        return -feats["raw_ret_15m"][rows]
    if rule == "xs_rev_1h":
        return -feats["raw_ret_1h"][rows]
    if rule == "xs_mom_4h":
        return feats["raw_ret_24h"][rows]
    if rule == "flow_1h":
        return feats["flow_1h"][rows]
    raise ValueError(rule)


# -- simulation ---------------------------------------------------------------------------------------
@dataclass
class Run:
    equity: pd.Series                       # daily, first point = starting capital
    turnover: float                         # traded notional / average equity, per year
    skipped_legs: int
    opened_legs: int
    wiped: bool
    average_legs: float


def _lots(notional: float, price: float, rule: dict) -> float:
    qty = notional / price
    if rule["step"] > 0:
        qty = math.floor(qty / rule["step"] + 1e-9) * rule["step"]
    return qty if qty > 0 and qty * price >= rule["min_notional"] else 0.0


def simulate(panel: Panel, rows: np.ndarray, scores: np.ndarray, *, capital: float, fee: float,
             slippage: float) -> Run:
    """Long the top LEGS and short the bottom LEGS of ``scores`` at each decision; fills at the next open."""
    n = len(panel.symbols)
    qty, cash = np.zeros(n), capital
    days, values = [int((panel.times[rows[0]] + BAR) // DAY) - 1], [capital]
    traded = skipped = opened = held_sum = 0
    last_fill, wiped = int(panel.times[rows[0] + 1]), False
    last_price = np.full(n, np.nan)

    def trade(k: int, delta: float, price: float) -> None:
        nonlocal cash, traded
        fill = price * (1 + slippage if delta > 0 else 1 - slippage)
        cash -= delta * fill + abs(delta) * fill * fee
        traded += abs(delta) * fill
        qty[k] += delta

    for row, score in zip(rows.tolist(), scores):
        fill_time = int(panel.times[row + 1])
        price = panel.open[row + 1]
        for k in np.flatnonzero(qty):
            times, rates = panel.funding[k]
            lo, hi = np.searchsorted(times, [last_fill, fill_time], side="right")
            for at, rate in zip(times[lo:hi].tolist(), rates[lo:hi].tolist()):
                mark = panel.open[min(int((at - panel.times[0]) // BAR), len(panel.times) - 1), k]
                cash -= qty[k] * (mark if math.isfinite(mark) else last_price[k]) * rate
            if not math.isfinite(price[k]):                       # the symbol stopped trading: close at the last price
                trade(k, -qty[k], last_price[k])
        last_price = np.where(np.isfinite(price), price, last_price)
        equity = cash + float(np.nansum(qty * np.where(np.isfinite(price), price, 0.0)))
        if equity <= 0:
            wiped = True
        if not wiped:
            ok = np.flatnonzero(np.isfinite(score) & np.isfinite(price))
            want = np.zeros(n)
            if len(ok) >= 2 * LEGS:
                order = ok[np.argsort(score[ok], kind="stable")]
                want[order[-LEGS:]], want[order[:LEGS]] = 1.0, -1.0
            for k in np.flatnonzero((np.sign(qty) != want) & (qty != 0) & np.isfinite(price)):
                trade(k, -qty[k], price[k])
            for k in np.flatnonzero((want != 0) & (qty == 0)):
                size = _lots(equity / (2 * LEGS), price[k], panel.rules[k])
                if size == 0:
                    skipped += 1
                    continue
                trade(k, want[k] * size, price[k])
                opened += 1
        held_sum += int(np.count_nonzero(qty))
        value = max(cash + float(np.nansum(qty * np.where(np.isfinite(price), price, 0.0))), 0.0) if not wiped else 0.0
        day = fill_time // DAY
        if day == days[-1]:
            values[-1] = value
        else:
            days.append(int(day))
            values.append(value)
        last_fill = fill_time
    equity_curve = pd.Series(values, index=days, dtype=float)
    years = max(len(equity_curve) - 1, 1) / 365.0
    return Run(equity_curve, traded / max(float(equity_curve.mean()), 1e-9) / years, skipped, opened, wiped,
               held_sum / max(len(rows), 1))


__all__ = ["BAR", "DAY", "LEGS", "RIDGE_FEATURES", "Panel", "Run", "load_panel", "features", "decision_rows",
           "forward_return", "ridge_scores", "rule_scores", "simulate", "demean"]
