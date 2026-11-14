"""Append-only registry of every configuration evaluated on real data.

Selection inflates the best Sharpe; the deflated Sharpe test needs to know
how many configurations were tried in a family and how dispersed their
Sharpes were. Every study appends its trials here (``record``) in the same
commit as its results, and a gate reads the whole family (``family_stats``).

A row:
    family     search family; trials in one family are compared together
    study      study id (e.g. "pair_2026-11-14")
    trial      id unique within the study
    config     what was evaluated
    window     evaluation window (ms)
    sharpe     annualized daily Sharpe, or null when the study did not record it
    count      trials the row stands for (> 1 only for rows without per-trial Sharpes)
    note       free text
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from lab.metrics import deflated_sharpe_probability_k

REGISTRY = Path("research/trials.jsonl")
REQUIRED = ("family", "study", "trial", "config", "window", "sharpe", "count")


def load(path: Path = REGISTRY, family: str | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if family is None or r["family"] == family]


def record(rows: list[dict], *, path: Path = REGISTRY) -> None:
    """Append rows; refuses incomplete rows and (study, trial) pairs already recorded."""
    path = Path(path)
    seen = {(r["study"], r["trial"]) for r in load(path)}
    lines = []
    for row in rows:
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"trial row missing {missing}")
        if not isinstance(row["count"], int) or row["count"] < 1:
            raise ValueError("count must be a positive integer")
        sharpe = row["sharpe"]
        if sharpe is not None and not (isinstance(sharpe, (int, float)) and math.isfinite(sharpe)):
            raise ValueError("sharpe must be finite or null")
        key = (row["study"], row["trial"])
        if key in seen:
            raise ValueError(f"trial {key} already recorded; the registry is append-only")
        seen.add(key)
        lines.append(json.dumps(row, sort_keys=True, default=str))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        for line in lines:
            stream.write(line + "\n")


def family_stats(family: str, *, path: Path = REGISTRY, extra: list[dict] = ()) -> dict:
    """Trial count and Sharpe dispersion of a family (registry rows plus ``extra`` not yet recorded)."""
    rows = load(path, family) + [r for r in extra if r["family"] == family]
    sharpes = [float(r["sharpe"]) for r in rows if r["sharpe"] is not None]
    return {"family": family, "n_trials": int(sum(r["count"] for r in rows)), "n_with_sharpe": len(sharpes),
            "sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) >= 2 else float("nan"),
            "best_sharpe": max(sharpes) if sharpes else float("nan")}


def deflated(daily_returns: pd.Series, family: str, *, path: Path = REGISTRY, extra: list[dict] = ()) -> dict:
    """Deflated Sharpe probability of ``daily_returns`` against every trial in ``family``."""
    stats = family_stats(family, path=path, extra=extra)
    std = stats["sharpe_std"] if stats["n_trials"] >= 2 else 0.0
    return {**stats, "probability": deflated_sharpe_probability_k(daily_returns, std, stats["n_trials"])}


__all__ = ["REGISTRY", "load", "record", "family_stats", "deflated"]
