"""Walk-Forward Validation — extracted from backtest_runner.py."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from runner.backtest.csv_io import iter_ohlcv_csv
from runner.backtest.metrics import (
    _build_trades_from_fills,
    _build_summary,
    _json_safe,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One window of a walk-forward test."""
    window_idx: int
    train_bars: int
    test_bars: int
    test_summary: Dict[str, Any]


def run_walk_forward(
    *,
    csv_path: Path,
    symbol: str,
    starting_balance: Decimal,
    ma_window: int,
    order_qty: Decimal,
    fee_bps: Decimal,
    slippage_bps: Decimal = Decimal("0"),
    train_size: int = 500,
    test_size: int = 100,
    out_dir: Optional[Path] = None,
) -> List[WalkForwardWindow]:
    from runner.backtest_runner import run_backtest

    all_bars = list(iter_ohlcv_csv(csv_path))
    if len(all_bars) < train_size + test_size:
        raise ValueError(
            f"Not enough bars for walk-forward: need {train_size + test_size}, got {len(all_bars)}"
        )

    results: List[WalkForwardWindow] = []
    window_idx = 0
    start = 0

    while start + train_size + test_size <= len(all_bars):
        test_start = start + train_size
        test_end = test_start + test_size
        test_window_bars = all_bars[start:test_end]
        test_only_bars = all_bars[test_start:test_end]

        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as tmp:
            writer = csv.writer(tmp)
            writer.writerow(["ts", "open", "high", "low", "close", "volume"])
            for bar in test_window_bars:
                writer.writerow([
                    bar.ts.isoformat(),
                    str(bar.o),
                    str(bar.h),
                    str(bar.l),
                    str(bar.c),
                    str(bar.v) if bar.v is not None else "0",
                ])
            tmp_path = tmp.name

        try:
            window_out = (out_dir / f"window_{window_idx:03d}") if out_dir else None
            equity, fills = run_backtest(
                csv_path=Path(tmp_path),
                symbol=symbol,
                starting_balance=starting_balance,
                ma_window=ma_window,
                order_qty=order_qty,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                out_dir=window_out,
            )

            test_equity = equity[train_size:] if len(equity) > train_size else equity
            test_fills_raw = fills

            trades = _build_trades_from_fills(test_fills_raw)
            summary = _build_summary(
                equity=test_equity,
                trades=trades,
                csv_path=csv_path,
                symbol=symbol,
            )
            summary["window_idx"] = window_idx
            summary["train_bars"] = train_size
            summary["test_bars"] = len(test_only_bars)
            summary["test_start_ts"] = test_only_bars[0].ts.isoformat() if test_only_bars else ""
            summary["test_end_ts"] = test_only_bars[-1].ts.isoformat() if test_only_bars else ""

            results.append(WalkForwardWindow(
                window_idx=window_idx,
                train_bars=train_size,
                test_bars=len(test_only_bars),
                test_summary=summary,
            ))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.debug("Failed to clean up temp file %s: %s", tmp_path, e)

        start += test_size
        window_idx += 1

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        wf_path = out_dir / "walk_forward_summary.json"
        with wf_path.open("w", encoding="utf-8") as f:
            json.dump(
                [_json_safe(w.test_summary) for w in results],
                f,
                ensure_ascii=False,
                indent=2,
            )

    return results
