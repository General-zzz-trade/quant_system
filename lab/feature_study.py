"""The feature-pool study's machinery (owner 2026-11-17: "来使用加入特征池的方式去做，参考更多的数据去做", "执行").

- ``targets``: what is predicted, fixed before any result. ``trend``: over the next ``horizon`` days the coin moves at
  least ``TREND_K`` times its usual move (the last 30 daily returns' spread x sqrt(horizon)) and in one direction
  (efficiency ratio of those days at least ``TREND_ER``). ``fwd_vol``: the log spread of the next ``horizon`` daily
  returns. ``up``: the coin is higher ``horizon`` days on. Labels look ahead by design; features never do.
- ``year_folds``: each test year trains on every earlier row whose label ended before the year began (the label
  window is purged), so nothing of the test year reaches the fit.
- ``evaluate``: fixed models (``MODELS``, no tuning), missing values filled from the training rows only, scored on
  the stitched test years: AUC for a yes/no target; for ``fwd_vol`` the R2 gained over the naive forecast (a linear
  fit on the naive feature alone, the last 30 days' spread).
- ``null_scores``: the same with the target shuffled in blocks (the time structure inside a block kept): what this
  many features find by chance.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DAY = 86_400_000
TREND_K, TREND_ER = 2.0, 0.4
MODELS = {
    "logistic": {"penalty": "l1", "C": 0.1, "solver": "liblinear", "max_iter": 1000},
    "lgbm": {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 50,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": 0, "verbose": -1},
    "ridge": {"alpha": 10.0},
}


def year_start(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def targets(closes: pd.Series, horizon: int) -> pd.DataFrame:
    """Per decision time (index: the day's end, when the close is known) the three targets and the label's end."""
    full = pd.RangeIndex(int(closes.index.min()), int(closes.index.max()) + DAY, DAY)
    c = closes.reindex(full).astype(float)
    logc = np.log(c)
    r = logc.diff()
    sigma = r.rolling(30, min_periods=30).std()
    future = logc.shift(-horizon)
    move = (future - logc).abs()
    path = c.diff().abs().rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    er = ((c.shift(-horizon) - c).abs() / path).where(path > 0)
    trend = ((move >= TREND_K * sigma * np.sqrt(horizon)) & (er.fillna(0.0) >= TREND_ER)).astype(float)
    known = future.notna() & sigma.notna()
    fwd = pd.Series(np.nan, index=c.index)
    for k in range(len(c) - horizon):                        # the spread of returns k+1 .. k+horizon
        window = r.iloc[k + 1:k + 1 + horizon]
        if window.notna().sum() == horizon and window.std() > 0:
            fwd.iloc[k] = np.log(window.std())
    out = pd.DataFrame({"trend": trend.where(known), "fwd_vol": fwd,
                        "up": (future > logc).astype(float).where(future.notna()),
                        "label_end": c.index.to_numpy(dtype=np.int64) + horizon * DAY}, index=c.index)
    return out.loc[closes.index]


def year_folds(index: pd.Index, label_end: pd.Series, years: Sequence[int]) -> list[dict]:
    """For each year after the first: positions of the training rows (labels ended before the year) and of the
    test rows (decision times inside the year)."""
    t = np.asarray(index, dtype=np.int64)
    ends = np.asarray(label_end.reindex(index), dtype=np.int64)
    folds = []
    for year in years[1:]:
        start, stop = year_start(year), year_start(year + 1)
        folds.append({"year": year, "train": np.flatnonzero(ends < start),
                      "test": np.flatnonzero((t >= start) & (t < stop))})
    return folds


def fill(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Missing values filled with the training rows' medians (0 where a column has none); an infinite value is
    missing."""
    train, test = train.replace([np.inf, -np.inf], np.nan), test.replace([np.inf, -np.inf], np.nan)
    medians = train.median().fillna(0.0)
    return train.fillna(medians), test.fillna(medians)


def _model(kind: str, model: str):
    if model == "lgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        return (LGBMClassifier if kind == "binary" else LGBMRegressor)(**MODELS["lgbm"])
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    inner = LogisticRegression(**MODELS["logistic"]) if model == "logistic" else Ridge(**MODELS["ridge"])
    return make_pipeline(StandardScaler(), inner)


def _fit_predict(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, kind: str, model: str) -> np.ndarray:
    x_train, x_test = fill(x_train, x_test)
    fitted = _model(kind, model).fit(x_train.to_numpy(float), y_train.to_numpy(float))
    if kind == "binary":
        return fitted.predict_proba(x_test.to_numpy(float))[:, 1]
    return fitted.predict(x_test.to_numpy(float))


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def evaluate(x: pd.DataFrame, y: pd.Series, label_end: pd.Series, years: Sequence[int], *, kind: str, model: str,
             naive: str | None = None) -> dict:
    """Out-of-sample predictions over the test years and their skill (see the module docstring)."""
    from sklearn.metrics import roc_auc_score
    keep = y.notna()
    x, y, label_end = x.loc[keep], y.loc[keep], label_end.loc[keep]
    preds, naive_preds, by_year = [], [], {}
    for fold in year_folds(x.index, label_end, years):
        if not len(fold["test"]) or not len(fold["train"]):
            continue
        tr, te = fold["train"], fold["test"]
        p = _fit_predict(x.iloc[tr], y.iloc[tr], x.iloc[te], kind, model)
        part = pd.Series(p, index=x.index[te])
        preds.append(part)
        yt = y.iloc[te].to_numpy(float)
        if kind == "binary":
            by_year[fold["year"]] = float(roc_auc_score(yt, p)) if 0 < yt.mean() < 1 else None
        else:
            q = _fit_predict(x[[naive]].iloc[tr], y.iloc[tr], x[[naive]].iloc[te], kind, "ridge")
            naive_preds.append(pd.Series(q, index=x.index[te]))
            by_year[fold["year"]] = _r2(yt, p) - _r2(yt, q)
    pred = pd.concat(preds)
    truth = y.loc[pred.index].to_numpy(float)
    out = {"predictions": pred, "by_year": by_year, "n_test": int(len(pred))}
    if kind == "binary":
        out["auc"] = float(roc_auc_score(truth, pred.to_numpy()))
        out["base_rate"] = float(truth.mean())
    else:
        q = pd.concat(naive_preds).loc[pred.index].to_numpy()
        out["r2"], out["r2_naive"] = _r2(truth, pred.to_numpy()), _r2(truth, q)
        out["r2_gain"] = out["r2"] - out["r2_naive"]
    return out


def _shuffled(y: pd.Series, block: int, rng: np.random.Generator) -> pd.Series:
    values = y.to_numpy(copy=True)
    blocks = [values[k:k + block] for k in range(0, len(values), block)]
    order = rng.permutation(len(blocks))
    return pd.Series(np.concatenate([blocks[k] for k in order]), index=y.index)


def null_scores(x: pd.DataFrame, y: pd.Series, label_end: pd.Series, years: Sequence[int], *, kind: str, model: str,
                runs: int, block: int, seed: int, naive: str | None = None) -> list[float]:
    """The skill score (AUC, or R2 gain) of ``runs`` evaluations with the target shuffled in blocks of ``block``."""
    rng = np.random.default_rng(seed)
    keep = y.notna()
    out = []
    for _ in range(runs):
        shuffled = _shuffled(y.loc[keep], block, rng)
        result = evaluate(x.loc[keep], shuffled, label_end.loc[keep], years, kind=kind, model=model, naive=naive)
        out.append(result["auc"] if kind == "binary" else result["r2_gain"])
    return out
