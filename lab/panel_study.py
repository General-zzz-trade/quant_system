"""The whole-market panel study (owner 2026-11-17: "做第二个路线", "我要求不只做BTC和ETH，全市场的永续合约USDT都可以做",
"执行"): one model over the most liquid crypto USDT perpetuals of each day instead of one per coin, for more samples.

- ``liquid_universe``: at each decision time the ``top`` contracts by median daily quote volume over the 30 days
  before it (complete days known by then), among those with at least 60 days of bars; BTC and ETH always.
- ``symbol_features``: ``SYMBOL_FEATURES`` of one contract, only from its own Binance bars and funding (so a live
  runtime would need Binance alone): returns scaled by volatility, the volatility and flow features of
  lab.feature_pool, its realized-volatility family, funding, and the 25 daily technical indicators.
- ``cross_section``: per day each contract's rank among that day's contracts for a few features, and the market's
  state (breadth, dispersion, BTC's own trend and volatility), the same for every contract that day.
- ``evaluate``: pooled walk-forward by year with the label window purged, every day weighing as much as any other
  (the weights of a day's rows sum to 1); fixed models; AUC, R2 gain over the naive volatility forecast, and for
  the direction target the daily cross-sectional rank correlation (its mean, a t statistic on days 14 apart, and
  each year's mean).
- ``shuffled``: the nulls. Direction: the target permuted across contracts within each day (the cross-sectional
  question); trend and volatility: each contract's target series shuffled in 30-day blocks (the time question).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from lab import feature_pool as fp
from lab import feature_study as st

H, DAY = fp.H, fp.DAY
ALWAYS = ("BTCUSDT", "ETHUSDT")
RANKED = ("ret_7_z", "ret_30_z", "funding_3d", "rv_30", "volume_z_30")
SCALED_RETURNS = (1, 3, 7, 14, 30, 60, 90)
PRICE = ("rv_7", "rv_30", "rv_ratio_7_30", "rv_pct_365", "range_ratio_14_90", "er_14", "er_30", "dist_ma_20",
         "dist_ma_50", "dist_ma_200", "skew_30", "vov_60", "hourly_rv_ratio", "taker_1d", "taker_7d", "volume_z_30")
REALIZED = ("rv_intraday_1d", "har_rv_7", "har_rv_30", "parkinson_7", "garman_klass_7", "ewma_vol", "jump_share_7",
            "downside_share_7", "vol_of_rv_30")
FUNDING = ("funding_3d", "funding_z_90", "funding_chg_7d")
SYMBOL_FEATURES = (tuple(f"ret_{k}_z" for k in SCALED_RETURNS) + PRICE + REALIZED + FUNDING
                   + tuple(f"ta_{n}_1d" for n in fp.TECHNICAL))
MARKET = ("breadth_ma30", "dispersion_7", "funding_pos_share", "alt_minus_btc_30", "btc_ret_7_z", "btc_ret_30_z",
          "btc_rv_30")
PANEL_FEATURES = SYMBOL_FEATURES + tuple(f"rank_{n}" for n in RANKED) + MARKET


def daily_volume(hourly: pd.DataFrame) -> pd.Series:
    """Quote volume of each complete UTC day, indexed by the day's end."""
    day = hourly["open_time"] // DAY
    g = hourly.groupby(day)["quote_volume"]
    full = g.count() == 24
    v = g.sum()[full]
    return pd.Series(v.to_numpy(float), index=(v.index.to_numpy(dtype=np.int64) + 1) * DAY)


def liquid_universe(volumes: dict[str, pd.Series], first_bar: dict[str, int], times: np.ndarray, *, top: int,
                    always: Sequence[str] = ALWAYS) -> pd.DataFrame:
    """Membership (True/False) of each contract at each decision time (index): the ``top`` by the median daily quote
    volume of the 30 days before the time, among contracts with 25 such days and a first bar 60 days before; plus
    ``always`` whenever they have data."""
    medians = {}
    for symbol, v in volumes.items():
        med = v.rolling(30, min_periods=25).median()
        m = fp._last(med, times, DAY)
        m[times - first_bar[symbol] < 60 * DAY] = np.nan
        medians[symbol] = m
    table = pd.DataFrame(medians, index=pd.Index(times))
    rank = table.rank(axis=1, ascending=False, method="first")
    member = (rank <= top) & table.notna()
    for symbol in always:
        if symbol in member:
            member[symbol] = member[symbol] | table[symbol].notna()
    return member


def _funding(funding: pd.Series, d_index: np.ndarray, times: np.ndarray) -> dict:
    f3 = funding.rolling(9, min_periods=9).mean() if len(funding) else funding
    f3_daily = pd.Series(fp._last(f3, d_index, 9 * H), index=d_index)
    out = {"funding_3d": fp._last(f3, times, 9 * H), "funding_z_90": fp._last(fp._z(f3_daily, 90), times, 2 * DAY)}
    out["funding_chg_7d"] = out["funding_3d"] - fp._last(f3, times - 7 * DAY, 9 * H)
    return out


def symbol_features(hourly: pd.DataFrame, funding: pd.Series, times: np.ndarray) -> pd.DataFrame:
    """``SYMBOL_FEATURES`` of one contract at each decision time (index), from values known by then."""
    times = np.asarray(times, dtype=np.int64)
    d = fp.daily(hourly)
    price = fp._price(d, hourly, times)
    out = {f"ret_{k}_z": price[f"ret_{k}"] / (price["rv_30"] * np.sqrt(k)) for k in SCALED_RETURNS}
    out.update({k: price[k] for k in PRICE})
    out.update(fp._realized(hourly, times))
    out.update(_funding(funding, d.index.to_numpy(np.int64), times))
    ta = fp.technical(fp.bars(hourly, 24))
    out.update({f"ta_{n}_1d": fp._last(ta[n], times, 2 * DAY) for n in fp.TECHNICAL})
    return pd.DataFrame({k: np.asarray(out[k], dtype=float) for k in SYMBOL_FEATURES}, index=pd.Index(times))


def cross_section(panel: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """``panel`` (index: decision time, a ``symbol`` column, ``SYMBOL_FEATURES``) with each day's ranks of ``RANKED``
    (0..1 among that day's contracts) and ``market`` (index: decision time) joined on the time."""
    out = panel.copy()
    for name in RANKED:
        out[f"rank_{name}"] = out.groupby(level=0)[name].rank(pct=True)
    joined = market.reindex(out.index)
    for name in MARKET:
        out[name] = joined[name].to_numpy()
    return out


def market_state(closes: pd.DataFrame, funding_3d: pd.DataFrame, btc: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    """``MARKET`` at each decision time: lab.feature_pool.market_breadth over every contract, and BTC's trend
    (returns scaled by its volatility) and volatility from ``btc`` (``symbol_features`` of BTCUSDT)."""
    breadth = fp.market_breadth(closes, funding_3d)
    out = {k: fp._last(breadth[k], times, 2 * DAY) for k in fp.BREADTH}
    b = btc.reindex(pd.Index(times))
    out.update({"btc_ret_7_z": b["ret_7_z"].to_numpy(), "btc_ret_30_z": b["ret_30_z"].to_numpy(),
                "btc_rv_30": b["rv_30"].to_numpy()})
    return pd.DataFrame(out, index=pd.Index(times))[list(MARKET)]


def load_symbol(snapshot: Path, symbol: str, limit: int) -> tuple[pd.DataFrame, pd.Series]:
    """One contract's hourly bars (with ``available_at``) and funding, rows known by ``limit``."""
    h = pd.read_csv(Path(snapshot) / f"{symbol}_1h.csv.gz")
    h["available_at"] = h["open_time"] + H
    h = h[h["available_at"] <= limit].reset_index(drop=True)
    path = Path(snapshot) / f"{symbol}_funding.csv.gz"
    f = pd.read_csv(path) if path.exists() else pd.DataFrame({"funding_time": [], "rate": []})
    f = f[f["funding_time"] <= limit]
    funding = pd.Series(f["rate"].to_numpy(float), index=f["funding_time"].to_numpy(dtype=np.int64)).sort_index()
    return h, funding


# -- targets and evaluation ----------------------------------------------------------------------------------------
def symbol_targets(closes: pd.Series, times: np.ndarray) -> pd.DataFrame:
    """Per decision time: ``trend_14``, ``up_14`` (lab.feature_study), ``fwd_vol_7``, ``fwd_vol_14`` and the
    direction target ``ret_14_scaled``, the next 14 days' log return over the last 30 days' spread x sqrt(14);
    and each label's end. Labels look ahead by design."""
    t14, t7 = st.targets(closes, 14), st.targets(closes, 7)
    full = t14.index
    c = closes.reindex(full).astype(float)
    logc = np.log(c)
    sigma = logc.diff().rolling(30, min_periods=30).std()
    scaled = (logc.shift(-14) - logc) / (sigma * np.sqrt(14))
    out = pd.DataFrame({"trend_14": t14["trend"], "up_14": t14["up"], "fwd_vol_7": t7["fwd_vol"],
                        "fwd_vol_14": t14["fwd_vol"], "ret_14_scaled": scaled.reindex(full),
                        "end_7": t7["label_end"], "end_14": t14["label_end"]}, index=full)
    return out.reindex(pd.Index(times))


def _weights(times: np.ndarray) -> np.ndarray:
    counts = pd.Series(times).map(pd.Series(times).value_counts()).to_numpy(float)
    return 1.0 / counts


def _fit_predict(x_train: pd.DataFrame, y_train: np.ndarray, w: np.ndarray, x_test: pd.DataFrame, kind: str,
                 model: str) -> np.ndarray:
    x_train, x_test = st.fill(x_train, x_test)
    if model == "lgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        params = {**st.MODELS["lgbm"], "num_leaves": 31, "min_child_samples": 200, "n_estimators": 300}
        m = (LGBMClassifier if kind == "binary" else LGBMRegressor)(**params)
        m.fit(x_train.to_numpy(np.float32), y_train, sample_weight=w)
    else:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        inner = LogisticRegression(**st.MODELS["logistic"]) if model == "logistic" else Ridge(**st.MODELS["ridge"])
        m = make_pipeline(StandardScaler(), inner)
        m.fit(x_train.to_numpy(np.float32), y_train, **{f"{m.steps[-1][0]}__sample_weight": w})
    x = x_test.to_numpy(np.float32)
    return m.predict_proba(x)[:, 1] if kind == "binary" else m.predict(x)


def daily_rank_ic(times: np.ndarray, pred: np.ndarray, y: np.ndarray, *, min_names: int = 10) -> pd.Series:
    """Each day's Spearman correlation between predictions and outcomes across contracts (days with fewer than
    ``min_names`` contracts left out)."""
    frame = pd.DataFrame({"t": times, "p": pred, "y": y})
    out = {}
    for t, g in frame.groupby("t"):
        if len(g) >= min_names:
            out[t] = g["p"].rank().corr(g["y"].rank())
    return pd.Series(out, dtype=float).sort_index()


def ic_summary(ic: pd.Series, *, step: int = 14) -> dict:
    """Mean daily IC, a t statistic on days ``step`` apart (non-overlapping 14-day labels), and each year's mean."""
    ic = ic.dropna()
    spaced = ic.iloc[::step]
    t = float(spaced.mean() / spaced.std() * np.sqrt(len(spaced))) if len(spaced) > 2 and spaced.std() > 0 else 0.0
    years = pd.to_datetime(ic.index.to_numpy(dtype=np.int64), unit="ms").year
    return {"mean_ic": float(ic.mean()), "t_spaced": t, "n_spaced": int(len(spaced)),
            "by_year": {int(y): float(ic[years == y].mean()) for y in sorted(set(years))}}


def evaluate(x: pd.DataFrame, y: np.ndarray, times: np.ndarray, label_end: np.ndarray, years: Sequence[int], *,
             kind: str, model: str, naive: str | None = None) -> dict:
    """Pooled walk-forward (see the module docstring); rows are positions, ``times`` their decision times."""
    from sklearn.metrics import roc_auc_score
    keep = np.isfinite(y)
    x, y, times, label_end = x.iloc[np.flatnonzero(keep)], y[keep], times[keep], label_end[keep]
    pred = np.full(len(y), np.nan)
    naive_pred = np.full(len(y), np.nan)
    for year in years[1:]:
        start, stop = st.year_start(year), st.year_start(year + 1)
        tr, te = np.flatnonzero(label_end < start), np.flatnonzero((times >= start) & (times < stop))
        if not len(tr) or not len(te):
            continue
        w = _weights(times[tr])
        pred[te] = _fit_predict(x.iloc[tr], y[tr], w, x.iloc[te], kind, model)
        if naive is not None:
            naive_pred[te] = _fit_predict(x[[naive]].iloc[tr], y[tr], w, x[[naive]].iloc[te], kind, "ridge")
    test = np.isfinite(pred)
    out: dict = {"n_test": int(test.sum()), "predictions": pred}
    tt, yt, pt = times[test], y[test], pred[test]
    yrs = pd.to_datetime(tt, unit="ms").year
    if kind == "binary":
        out["auc"] = float(roc_auc_score(yt, pt))
        out["by_year"] = {int(v): float(roc_auc_score(yt[yrs == v], pt[yrs == v]))
                          for v in sorted(set(yrs)) if 0 < yt[yrs == v].mean() < 1}
        return out

    def r2(a: np.ndarray, b: np.ndarray) -> float:
        return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))
    if naive is not None:
        qt = naive_pred[test]
        out["r2"], out["r2_naive"] = r2(yt, pt), r2(yt, qt)
        out["r2_gain"] = out["r2"] - out["r2_naive"]
        out["by_year"] = {int(v): r2(yt[yrs == v], pt[yrs == v]) - r2(yt[yrs == v], qt[yrs == v])
                          for v in sorted(set(yrs))}
    out["ic"] = ic_summary(daily_rank_ic(tt, pt, yt))
    return out


def shuffled(y: np.ndarray, times: np.ndarray, symbols: np.ndarray, *, how: str, rng: np.random.Generator,
             block: int = 30) -> np.ndarray:
    """A null target: ``how="within_day"`` permutes the target across contracts inside each day; ``"blocks"``
    shuffles each contract's own target series in blocks of ``block`` days (in time order)."""
    out = y.copy()
    frame = pd.DataFrame({"t": times, "s": symbols, "i": np.arange(len(y))})
    if how == "within_day":
        for _, g in frame.groupby("t"):
            idx = g["i"].to_numpy()
            out[idx] = y[rng.permutation(idx)]
        return out
    for _, g in frame.sort_values("t").groupby("s"):
        idx = g["i"].to_numpy()
        blocks = [idx[k:k + block] for k in range(0, len(idx), block)]
        order = rng.permutation(len(blocks))
        out[idx] = y[np.concatenate([blocks[k] for k in order])]
    return out
