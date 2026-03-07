from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

from runner.backtest_runner import run_multi_backtest


def _write_csv(path: Path, base_price: int = 100, rows: int = 30) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for i in range(rows):
            ts = f"2024-01-01T{i:02d}:00:00Z" if i < 24 else f"2024-01-02T{i-24:02d}:00:00Z"
            p = base_price + (i % 5) - 2
            w.writerow([ts, p, p + 1, p - 1, p, 100])


def test_multi_backtest_basic(tmp_path):
    btc_csv = tmp_path / "btc.csv"
    eth_csv = tmp_path / "eth.csv"
    _write_csv(btc_csv, base_price=40000)
    _write_csv(eth_csv, base_price=2000)

    eq, fills = run_multi_backtest(
        csv_paths={"BTCUSDT": btc_csv, "ETHUSDT": eth_csv},
        starting_balance=Decimal("10000"),
        fee_bps=Decimal("0"),
        decision_modules=[],
    )
    # Should have equity points from interleaved bars
    assert len(eq) > 0
    # No decision modules → no fills
    assert len(fills) == 0


def test_multi_backtest_output(tmp_path):
    btc_csv = tmp_path / "btc.csv"
    eth_csv = tmp_path / "eth.csv"
    _write_csv(btc_csv, base_price=40000)
    _write_csv(eth_csv, base_price=2000)

    out = tmp_path / "out"
    eq, fills = run_multi_backtest(
        csv_paths={"BTCUSDT": btc_csv, "ETHUSDT": eth_csv},
        starting_balance=Decimal("10000"),
        fee_bps=Decimal("0"),
        decision_modules=[],
        out_dir=out,
    )
    assert (out / "equity_curve.csv").exists()
    assert (out / "fills.csv").exists()
    assert (out / "summary.json").exists()


def test_multi_backtest_single_symbol(tmp_path):
    csv_path = tmp_path / "btc.csv"
    _write_csv(csv_path, base_price=100)

    eq, fills = run_multi_backtest(
        csv_paths={"BTCUSDT": csv_path},
        starting_balance=Decimal("10000"),
        fee_bps=Decimal("0"),
        decision_modules=[],
    )
    assert len(eq) > 0
