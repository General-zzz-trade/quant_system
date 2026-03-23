"""Tests for demo_tracker.py parsing and aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ops.demo_tracker import (
    _empty_record,
    compute_rolling_sharpe,
    load_track_record,
    parse_bar_line,
    parse_close_line,
    parse_open_line,
    parse_incremental,
    save_track_record,
)


# ---------------------------------------------------------------------------
# parse_bar_line
# ---------------------------------------------------------------------------


class TestParseBarLine:
    def test_parse_bar_line_valid(self) -> None:
        line = "WS ETHUSDT bar 123: $3500.00 z=1.23 sig=1 hold=5 regime=trending dz=0.5"
        result = parse_bar_line(line)
        assert result is not None
        assert result["symbol"] == "ETHUSDT"
        assert result["bar_n"] == 123
        assert result["price"] == pytest.approx(3500.00)
        assert result["z"] == pytest.approx(1.23)
        assert result["sig"] == 1
        assert result["hold"] == 5
        assert result["regime"] == "trending"
        assert result["dz"] == pytest.approx(0.5)

    def test_parse_bar_line_negative_sig(self) -> None:
        line = "WS BTCUSDT bar 7: $60000.00 z=-2.10 sig=-1 hold=3 regime=ranging dz=0.8"
        result = parse_bar_line(line)
        assert result is not None
        assert result["symbol"] == "BTCUSDT"
        assert result["sig"] == -1
        assert result["z"] == pytest.approx(-2.10)

    def test_parse_bar_line_flat_sig(self) -> None:
        line = "WS SUIUSDT bar 1: $0.50 z=0.10 sig=0 hold=0 regime=flat dz=1.0"
        result = parse_bar_line(line)
        assert result is not None
        assert result["sig"] == 0

    def test_parse_bar_line_invalid(self) -> None:
        assert parse_bar_line("garbage line with no pattern") is None
        assert parse_bar_line("") is None
        assert parse_bar_line("WS HEARTBEAT sigs={} pm={} hedge={} store={}") is None

    def test_parse_bar_line_partial(self) -> None:
        # Missing fields
        assert parse_bar_line("WS ETHUSDT bar 1: $100.00") is None


# ---------------------------------------------------------------------------
# parse_close_line
# ---------------------------------------------------------------------------


class TestParseCloseLine:
    def test_parse_close_line_valid_long(self) -> None:
        line = "ETHUSDT CLOSE LONG: pnl=$12.50 (0.25%) total=$150.00 wins=3/5"
        result = parse_close_line(line)
        assert result is not None
        assert result["symbol"] == "ETHUSDT"
        assert result["side"] == "LONG"
        assert result["pnl_usd"] == pytest.approx(12.50)
        assert result["pct"] == pytest.approx(0.25)
        assert result["total_usd"] == pytest.approx(150.00)
        assert result["wins"] == 3
        assert result["total_trades"] == 5

    def test_parse_close_line_valid_short_negative_pnl(self) -> None:
        line = "BTCUSDT CLOSE SHORT: pnl=$-8.00 (-0.10%) total=$100.00 wins=2/4"
        result = parse_close_line(line)
        assert result is not None
        assert result["side"] == "SHORT"
        assert result["pnl_usd"] == pytest.approx(-8.00)

    def test_parse_close_line_prefixed_lowercase_without_pct(self) -> None:
        line = (
            "2026-03-18 02:00:01,116 INFO scripts.ops.portfolio_combiner: "
            "COMBO CLOSE long: pnl=$-35.8776 total=$-35.8776 wins=0/1"
        )
        result = parse_close_line(line)
        assert result is not None
        assert result["symbol"] == "COMBO"
        assert result["side"] == "LONG"
        assert result["pnl_usd"] == pytest.approx(-35.8776)
        assert result["pct"] is None
        assert result["total_usd"] == pytest.approx(-35.8776)
        assert result["wins"] == 0
        assert result["total_trades"] == 1

    def test_parse_close_line_prefixed_lowercase_with_pct(self) -> None:
        line = (
            "2026-03-18 14:00:01,991 INFO runner.alpha_runner: "
            "AXSUSDT CLOSE short: pnl=$-105.0948 (-3.00%) total=$-105.0948 wins=0/1"
        )
        result = parse_close_line(line)
        assert result is not None
        assert result["symbol"] == "AXSUSDT"
        assert result["side"] == "SHORT"
        assert result["pnl_usd"] == pytest.approx(-105.0948)
        assert result["pct"] == pytest.approx(-3.00)

    def test_parse_close_line_invalid(self) -> None:
        assert parse_close_line("garbage") is None
        assert parse_close_line("") is None
        assert parse_close_line("WS ETHUSDT bar 1: $100.00 z=1 sig=1 hold=1 regime=x dz=0.5") is None


# ---------------------------------------------------------------------------
# parse_open_line
# ---------------------------------------------------------------------------


class TestParseOpenLine:
    def test_parse_open_line_valid(self) -> None:
        line = "Opened LONG 0.01 @ ~$3500.00 stop=$3430.00"
        result = parse_open_line(line)
        assert result is not None
        assert result["side"] == "LONG"
        assert result["qty"] == pytest.approx(0.01)
        assert result["price"] == pytest.approx(3500.00)
        assert result["stop"] == pytest.approx(3430.00)

    def test_parse_open_line_short_no_stop(self) -> None:
        line = "Opened SHORT 5.0 @ ~$0.45"
        result = parse_open_line(line)
        assert result is not None
        assert result["side"] == "SHORT"
        assert result["qty"] == pytest.approx(5.0)
        assert result["price"] == pytest.approx(0.45)
        assert result["stop"] is None

    @pytest.mark.parametrize(
        ("line", "expected_side", "expected_qty", "expected_price", "expected_stop"),
        [
            (
                "2026-03-18 05:00:03,853 INFO runner.alpha_runner: "
                "Opened sell 4045.3000 @ ~$1.2 stop=$1.24 (ATR=0.86%): {'status': 'submitted'}",
                "SELL",
                4045.3,
                1.2,
                1.24,
            ),
            (
                "2026-03-18 06:00:01,105 INFO runner.alpha_runner: "
                "Opened buy 0.0670 @ ~$74475.7 stop=$72754.61 (ATR=0.78%): {'status': 'submitted'}",
                "BUY",
                0.067,
                74475.7,
                72754.61,
            ),
        ],
    )
    def test_parse_open_line_prefixed_buy_sell(
        self,
        line: str,
        expected_side: str,
        expected_qty: float,
        expected_price: float,
        expected_stop: float,
    ) -> None:
        result = parse_open_line(line)
        assert result is not None
        assert result["side"] == expected_side
        assert result["qty"] == pytest.approx(expected_qty)
        assert result["price"] == pytest.approx(expected_price)
        assert result["stop"] == pytest.approx(expected_stop)

    def test_parse_open_line_invalid(self) -> None:
        assert parse_open_line("garbage") is None
        assert parse_open_line("") is None


# ---------------------------------------------------------------------------
# load_track_record
# ---------------------------------------------------------------------------


class TestLoadTrackRecord:
    def test_load_track_record_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        result = load_track_record(path)
        assert "last_parsed_offset" in result
        assert "last_updated" in result
        assert "daily" in result
        assert "summary" in result
        assert result["last_parsed_offset"] == 0
        assert result["daily"] == {}

    def test_load_track_record_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "record.json"
        data = _empty_record()
        data["summary"]["total_trades"] = 42
        path.write_text(json.dumps(data), encoding="utf-8")
        result = load_track_record(path)
        assert result["summary"]["total_trades"] == 42

    def test_load_track_record_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json", encoding="utf-8")
        # Should return empty template without raising
        result = load_track_record(path)
        assert result["last_parsed_offset"] == 0
        assert result["daily"] == {}

    def test_load_track_record_adds_missing_keys(self, tmp_path: Path) -> None:
        """Partial JSON (older schema) should be upgraded with defaults."""
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"last_parsed_offset": 99}), encoding="utf-8")
        result = load_track_record(path)
        assert result["last_parsed_offset"] == 99
        assert "daily" in result
        assert "summary" in result


# ---------------------------------------------------------------------------
# parse_incremental
# ---------------------------------------------------------------------------


class TestParseIncremental:
    def test_parse_incremental_empty_log(self, tmp_path: Path) -> None:
        log = tmp_path / "empty.log"
        log.write_text("", encoding="utf-8")
        record = _empty_record()
        lines = parse_incremental(log, record)
        assert lines == 0
        assert record["last_parsed_offset"] == 0

    def test_parse_incremental_missing_log(self, tmp_path: Path) -> None:
        log = tmp_path / "missing.log"
        record = _empty_record()
        lines = parse_incremental(log, record)
        assert lines == 0

    def test_parse_incremental_with_content(self, tmp_path: Path) -> None:
        log = tmp_path / "alpha.log"
        log.write_text(
            "WS ETHUSDT bar 1: $3500.00 z=1.50 sig=1 hold=0 regime=trending dz=0.5\n"
            "WS ETHUSDT bar 2: $3510.00 z=1.60 sig=1 hold=1 regime=trending dz=0.5\n"
            "ETHUSDT CLOSE LONG: pnl=$15.00 (0.43%) total=$15.00 wins=1/1\n",
            encoding="utf-8",
        )
        record = _empty_record()
        lines = parse_incremental(log, record)
        assert lines == 3

        # Bar data should be reflected
        daily = record["daily"]
        assert len(daily) == 1
        date_key = list(daily.keys())[0]
        eth = daily[date_key]["symbols"]["ETHUSDT"]
        assert eth["bars"] == 2
        assert eth["signals"]["long"] == 2
        assert eth["trades"] == 1
        assert eth["pnl_usd"] == pytest.approx(15.00)
        assert eth["wins"] == 1

    def test_parse_incremental_incremental_offset(self, tmp_path: Path) -> None:
        """Second call should only parse new lines."""
        log = tmp_path / "alpha.log"
        log.write_text(
            "WS ETHUSDT bar 1: $3500.00 z=1.50 sig=1 hold=0 regime=trending dz=0.5\n",
            encoding="utf-8",
        )
        record = _empty_record()
        lines1 = parse_incremental(log, record)
        assert lines1 == 1
        offset_after_first = record["last_parsed_offset"]
        assert offset_after_first > 0

        # Append more lines
        with log.open("a", encoding="utf-8") as fh:
            fh.write("WS BTCUSDT bar 10: $60000.00 z=0.50 sig=0 hold=2 regime=flat dz=1.0\n")

        lines2 = parse_incremental(log, record)
        assert lines2 == 1  # Only the new line

    def test_parse_incremental_garbled_lines_skipped(self, tmp_path: Path) -> None:
        log = tmp_path / "alpha.log"
        log.write_text(
            "this is totally garbage\n"
            "WS ETHUSDT bar 1: $3500.00 z=1.00 sig=1 hold=0 regime=trending dz=0.5\n"
            "more garbage\n",
            encoding="utf-8",
        )
        record = _empty_record()
        lines = parse_incremental(log, record)
        assert lines == 3  # All lines counted, garbled ones silently skipped

        daily = record["daily"]
        date_key = list(daily.keys())[0]
        assert daily[date_key]["symbols"]["ETHUSDT"]["bars"] == 1

    def test_parse_incremental_uses_logged_dates_and_real_close_formats(self, tmp_path: Path) -> None:
        log = tmp_path / "alpha.log"
        log.write_text(
            "2026-03-17 23:59:59,999 INFO runner.alpha_runner: "
            "WS ETHUSDT bar 1: $3500.00 z=1.50 sig=1 hold=0 regime=trending dz=0.5\n"
            "2026-03-18 00:00:02,263 INFO scripts.ops.portfolio_combiner: "
            "COMBO CLOSE short: pnl=$-3.0576 total=$-3.0576 wins=0/1\n"
            "2026-03-18 14:00:01,991 INFO runner.alpha_runner: "
            "AXSUSDT CLOSE short: pnl=$-105.0948 (-3.00%) total=$-105.0948 wins=0/1\n",
            encoding="utf-8",
        )
        record = _empty_record()

        lines = parse_incremental(log, record)

        assert lines == 3
        assert set(record["daily"]) == {"2026-03-17", "2026-03-18"}
        assert record["daily"]["2026-03-17"]["symbols"]["ETHUSDT"]["bars"] == 1

        day_2026_03_18 = record["daily"]["2026-03-18"]
        assert day_2026_03_18["total_trades"] == 2
        assert day_2026_03_18["total_pnl_usd"] == pytest.approx(-108.1524)
        assert day_2026_03_18["symbols"]["COMBO"]["trades"] == 1
        assert day_2026_03_18["symbols"]["AXSUSDT"]["losses"] == 1
        assert record["summary"]["total_trades"] == 2
        assert record["summary"]["total_pnl_usd"] == pytest.approx(-108.1524)


# ---------------------------------------------------------------------------
# compute_rolling_sharpe
# ---------------------------------------------------------------------------


class TestComputeRollingSharpe:
    def test_compute_rolling_sharpe_basic(self) -> None:
        # Constant positive PnL → positive Sharpe
        pnl = [10.0] * 20
        # All same → std=0 → None
        result = compute_rolling_sharpe(pnl, window=7)
        assert result is None

    def test_compute_rolling_sharpe_varying(self) -> None:
        import random
        random.seed(42)
        pnl = [random.gauss(5.0, 2.0) for _ in range(30)]
        result = compute_rolling_sharpe(pnl, window=7)
        assert result is not None
        assert isinstance(result, float)

    def test_compute_rolling_sharpe_zero_std(self) -> None:
        # All identical values → zero std → should return None (not crash)
        pnl = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        result = compute_rolling_sharpe(pnl, window=7)
        assert result is None

    def test_compute_rolling_sharpe_empty(self) -> None:
        assert compute_rolling_sharpe([], window=7) is None

    def test_compute_rolling_sharpe_one_element(self) -> None:
        assert compute_rolling_sharpe([10.0], window=7) is None

    def test_compute_rolling_sharpe_uses_window_tail(self) -> None:
        # Long series; result should be based on last `window` elements
        # Use last 7 identical → None; if we make last 7 varied → float
        pnl = [0.0] * 100 + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        result = compute_rolling_sharpe(pnl, window=7)
        assert result is not None

    def test_compute_rolling_sharpe_negative_mean(self) -> None:
        pnl = [-10.0, -5.0, -8.0, -12.0, -3.0, -9.0, -6.0]
        result = compute_rolling_sharpe(pnl, window=7)
        assert result is not None
        assert result < 0


# ---------------------------------------------------------------------------
# save_and_load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "record.json"
        original = _empty_record()
        original["summary"]["total_trades"] = 7
        original["summary"]["total_pnl_usd"] = 123.45
        original["daily"]["2026-03-17"] = {
            "symbols": {
                "ETHUSDT": {
                    "bars": 24,
                    "signals": {"long": 3, "short": 1, "flat": 20},
                    "trades": 2,
                    "pnl_usd": 15.50,
                    "wins": 1,
                    "losses": 1,
                }
            },
            "total_pnl_usd": 15.50,
            "total_trades": 2,
            "max_drawdown": -5.00,
        }
        save_track_record(original, path)
        loaded = load_track_record(path)

        assert loaded["summary"]["total_trades"] == 7
        assert loaded["summary"]["total_pnl_usd"] == pytest.approx(123.45)
        eth = loaded["daily"]["2026-03-17"]["symbols"]["ETHUSDT"]
        assert eth["bars"] == 24
        assert eth["pnl_usd"] == pytest.approx(15.50)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "record.json"
        record = _empty_record()
        save_track_record(record, path)
        assert path.exists()

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        """Verify .tmp file is cleaned up after save."""
        path = tmp_path / "record.json"
        record = _empty_record()
        save_track_record(record, path)
        tmp = path.with_suffix(".tmp")
        assert not tmp.exists()
        assert path.exists()
