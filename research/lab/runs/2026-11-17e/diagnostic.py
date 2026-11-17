"""After the 2026-11-17e results (not registered before them; report only, no new configuration, no selection): the
walk-forward's chosen configurations of each idea as a gross daily long/short book - no fees, no slippage, no
funding, no stop - to see whether the signal or the trading lost the money.

Each day d+1 every symbol of the fold's list holds the target decided at the close of day d (the weight of the day's
first hour), at 2x of a tenth of the capital: the day's return is the sum of weight x 2 x the symbol's close-to-close
return / 10.

    python research/lab/runs/2026-11-17e/diagnostic.py      # -> diagnostic.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))
import judge as run  # noqa: E402
from lab import um_lab as lab  # noqa: E402

DAY = lab.DAY


def gross(name: str, row: dict, j) -> pd.Series:
    parts = []
    for k, label in zip(j.folds, row["chosen_per_fold"]):
        start, end = j.periods[k]
        symbols = list(j.lists[k])
        strategy = j.ideas[name][0](label["params"])
        targets = j._targets(strategy, symbols)
        days = np.arange(start // DAY, end // DAY)
        book = pd.Series(0.0, index=days)
        for s in symbols:
            close = lab.frame(s, run.SNAP)["close"]
            daily = close.groupby(close.index // DAY).last()
            ret = (daily / daily.shift(1) - 1).reindex(days).fillna(0.0)
            first_hour = targets[s].groupby(targets[s].index // DAY).first().reindex(days).fillna(0.0)
            book += first_hour * 2 * ret / run.SLOTS
        parts.append(book)
    return pd.concat(parts)


def main() -> None:
    screened = json.loads((HERE / "screen.json").read_text())
    j = run.make()
    out = {}
    for name, row in screened["ideas"].items():
        s = lab.summary(gross(name, row, j))
        out[name] = {k: s.get(k) for k in ("total_return", "sharpe", "max_drawdown", "days")}
    (HERE / "diagnostic.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
