"""Tests for scripts/run_alpha_research.py — factor library, custom loading, CLI."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.run_alpha_research import (
    BUILTIN_FACTORS,
    _find_csv,
    _momentum,
    _rsi_factor,
    _macd_hist_factor,
    _bb_pct_factor,
    _vol_momentum,
    _price_accel,
    load_custom_factor,
)
from runner.backtest.csv_io import OhlcvBar
from datetime import datetime, timedelta, timezone


# ── helpers ─────────────────────────────────────────────────

_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_bars(prices, volumes=None) -> list[OhlcvBar]:
    bars = []
    for i, p in enumerate(prices):
        v = Decimal(str(volumes[i])) if volumes else Decimal("100")
        dp = Decimal(str(p))
        bars.append(OhlcvBar(
            ts=_EPOCH + timedelta(hours=i),
            o=dp, h=dp + Decimal("1"), l=dp - Decimal("1"), c=dp, v=v,
        ))
    return bars


# ── Built-in factor library ────────────────────────────────


def test_momentum_factor():
    prices = [100 + i for i in range(25)]
    bars = _make_bars(prices)
    compute = _momentum(5)
    result = compute(bars)
    assert len(result) == len(bars)
    assert result[0] is None
    assert result[4] is None
    assert result[5] is not None
    # (105 / 100) - 1 = 0.05
    assert result[5] == pytest.approx(0.05)


def test_momentum_zero_price():
    bars = _make_bars([0, 0, 0, 100, 100])
    compute = _momentum(2)
    result = compute(bars)
    # prev = 0 => skip
    assert result[2] is None


def test_rsi_factor_warmup():
    prices = [100 + i * 0.5 for i in range(30)]
    bars = _make_bars(prices)
    compute = _rsi_factor(14)
    result = compute(bars)
    assert len(result) == len(bars)
    # First window+1 should be None
    for i in range(14):
        assert result[i] is None


def test_rsi_factor_uptrend():
    prices = [100 + i for i in range(40)]
    bars = _make_bars(prices)
    compute = _rsi_factor(14)
    result = compute(bars)
    valid = [v for v in result if v is not None]
    assert len(valid) > 0
    # Monotonic uptrend => RSI should be high (close to 100)
    assert valid[-1] > 70


def test_macd_hist_factor_warmup():
    prices = [100 + i * 0.1 for i in range(50)]
    bars = _make_bars(prices)
    compute = _macd_hist_factor()
    result = compute(bars)
    # First 26 should be None (needs slow EMA warmup)
    for i in range(26):
        assert result[i] is None


def test_macd_hist_factor_returns_values():
    prices = [100 + i * 0.5 for i in range(50)]
    bars = _make_bars(prices)
    compute = _macd_hist_factor()
    result = compute(bars)
    valid = [v for v in result if v is not None]
    assert len(valid) > 0


def test_bb_pct_factor():
    prices = [100.0] * 20 + [110.0]
    bars = _make_bars(prices)
    compute = _bb_pct_factor()
    result = compute(bars)
    assert result[18] is None  # window=20, first valid at index 19
    # All same price => std=0 => None
    assert result[19] is None
    # Spike at index 20: well above the band
    assert result[20] is not None
    assert result[20] > 0.5  # above midpoint since price jumped up


def test_vol_momentum():
    volumes = [100] * 20 + [500]
    bars = _make_bars([100] * 21, volumes=volumes)
    compute = _vol_momentum(20)
    result = compute(bars)
    assert result[19] is None
    # Index 20: current volume (500) / mean of prev 20 (100) = 5.0
    assert result[20] == pytest.approx(5.0)


def test_price_accel():
    prices = [100 + i for i in range(25)]
    bars = _make_bars(prices)
    compute = _price_accel()
    result = compute(bars)
    for i in range(20):
        assert result[i] is None
    assert result[20] is not None


def test_builtin_factors_registered():
    expected = {"momentum_20", "momentum_50", "rsi_14", "macd_hist",
                "bb_pct", "vol_mom_20", "price_accel"}
    assert set(BUILTIN_FACTORS.keys()) == expected


def test_builtin_factors_have_categories():
    categories = {f.category for f in BUILTIN_FACTORS.values()}
    assert "momentum" in categories
    assert "technical" in categories
    assert "volume" in categories
    assert "composite" in categories


# ── load_custom_factor ──────────────────────────────────────


def test_load_custom_factor_no_colon():
    with pytest.raises(ValueError, match="path:function"):
        load_custom_factor("somefile.py")


def test_load_custom_factor_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_custom_factor("/nonexistent/path.py:func")


def test_load_custom_factor_success(tmp_path):
    factor_file = tmp_path / "my_factor.py"
    factor_file.write_text(
        "def my_alpha(bars):\n"
        "    return [None] * len(bars)\n"
    )
    factor = load_custom_factor(f"{factor_file}:my_alpha")
    assert factor.name == "my_alpha"
    assert factor.category == "custom"
    # Verify the compute function works
    assert factor.compute_fn([]) == []


def test_load_custom_factor_missing_function(tmp_path):
    factor_file = tmp_path / "no_func.py"
    factor_file.write_text("x = 1\n")
    with pytest.raises(AttributeError, match="nonexistent"):
        load_custom_factor(f"{factor_file}:nonexistent")


# ── _find_csv ───────────────────────────────────────────────


def test_find_csv_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="No CSV data found"):
        _find_csv()


def test_find_csv_data_files_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data_files"
    data_dir.mkdir()
    csv_file = data_dir / "test.csv"
    csv_file.write_text("col1\n1\n")

    result = _find_csv()
    assert result.name == "test.csv"


# ── main (CLI, minimal) ────────────────────────────────────


def test_main_no_factors_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ts,open,high,low,close,volume\n2024-01-01,100,101,99,100,1000\n")

    with patch("sys.argv", ["run_alpha_research.py", "--csv", str(csv_file),
                             "--factors", "nonexistent_factor"]):
        from scripts.run_alpha_research import main
        main()

    captured = capsys.readouterr()
    assert "WARNING" in captured.out or "No factors" in captured.out
