"""Tests for scripts/download_binance_klines.py — URL construction, interval conversion, dedup."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.download_binance_klines import (
    BASE_URL,
    ENDPOINT,
    LIMIT,
    _ts_str,
    fetch_klines,
    interval_ms,
)


# ── interval_ms ─────────────────────────────────────────────


def test_interval_ms_minutes():
    assert interval_ms("1m") == 60_000
    assert interval_ms("5m") == 300_000
    assert interval_ms("15m") == 900_000


def test_interval_ms_hours():
    assert interval_ms("1h") == 3_600_000
    assert interval_ms("4h") == 14_400_000


def test_interval_ms_days():
    assert interval_ms("1d") == 86_400_000


def test_interval_ms_invalid_unit():
    with pytest.raises(KeyError):
        interval_ms("1w")


# ── _ts_str ─────────────────────────────────────────────────


def test_ts_str_epoch():
    # 2020-01-01 00:00 UTC = 1577836800000 ms
    result = _ts_str(1577836800000)
    assert result == "2020-01-01 00:00"


def test_ts_str_known_date():
    # 2019-09-08 00:00 UTC = 1567900800000 ms
    result = _ts_str(1567900800000)
    assert result == "2019-09-08 00:00"


# ── fetch_klines ────────────────────────────────────────────


def test_fetch_klines_url_construction():
    mock_data = [[1577836800000, "7200", "7250", "7150", "7195", "100"]]
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_data).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("scripts.download_binance_klines.urlopen", return_value=mock_resp) as mock_urlopen:
        result = fetch_klines("BTCUSDT", "1h", 1577836800000, limit=100)

    call_args = mock_urlopen.call_args
    req = call_args[0][0]
    assert "symbol=BTCUSDT" in req.full_url
    assert "interval=1h" in req.full_url
    assert "startTime=1577836800000" in req.full_url
    assert "limit=100" in req.full_url
    assert result == mock_data


def test_fetch_klines_default_limit():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"[]"
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("scripts.download_binance_klines.urlopen", return_value=mock_resp) as mock_urlopen:
        fetch_klines("ETHUSDT", "15m", 1000)

    req = mock_urlopen.call_args[0][0]
    assert f"limit={LIMIT}" in req.full_url


# ── download_all ────────────────────────────────────────────


def test_download_all_writes_csv(tmp_path):
    from scripts.download_binance_klines import download_all

    def kline_row(ts):
        return [ts, "100", "105", "95", "102", "500",
                                ts + 3600000, "51000", "100", "250", "25500", "0"]

    # First batch must have exactly LIMIT rows to trigger continuation
    batch1 = [kline_row(1567900800000 + i * 3600000) for i in range(LIMIT)]
    # Second batch has fewer rows => signals end
    last_ts = 1567900800000 + LIMIT * 3600000
    batch2 = [kline_row(last_ts + i * 3600000) for i in range(3)]

    call_count = [0]

    def mock_fetch(symbol, interval, start_time, limit=LIMIT):
        call_count[0] += 1
        if call_count[0] == 1:
            return batch1
        elif call_count[0] == 2:
            return batch2
        return []

    out_path = str(tmp_path / "test_klines.csv")

    with patch("scripts.download_binance_klines.fetch_klines", side_effect=mock_fetch):
        with patch("time.time", return_value=(1567900800 + (LIMIT + 10) * 3600)):
            with patch("time.sleep"):
                count = download_all("BTCUSDT", "1h", out_path)

    assert count == LIMIT + 3
    assert Path(out_path).exists()

    with open(out_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "open_time" in header
        assert "close" in header
        rows = list(reader)
        assert len(rows) == LIMIT + 3


def test_download_all_single_batch(tmp_path):
    from scripts.download_binance_klines import download_all

    def kline_row(ts):
        return [ts, "100", "105", "95", "102", "500",
                                ts + 3600000, "51000", "100", "250", "25500", "0"]

    # Fewer than LIMIT => single batch, loop exits
    batch = [kline_row(1567900800000 + i * 3600000) for i in range(5)]

    with patch("scripts.download_binance_klines.fetch_klines", return_value=batch):
        with patch("time.time", return_value=1567900800 + 100 * 3600):
            count = download_all("BTCUSDT", "1h", str(tmp_path / "out.csv"))

    assert count == 5


def test_download_all_deduplicates(tmp_path):
    from scripts.download_binance_klines import download_all

    # Use timestamps relative to the hardcoded start_ms in download_all
    start_ms = 1567900800000  # same as download_all's start
    def kline_row(ts):
        return [ts, "100", "105", "95", "102", "500",
                                ts + 3600000, "51000", "100", "250", "25500", "0"]

    # Build batch1 with LIMIT rows
    batch1 = [kline_row(start_ms + i * 3600000) for i in range(LIMIT)]
    last_ts = batch1[-1][0]
    # batch2 overlaps: re-sends last row of batch1, plus 2 new
    batch2 = [kline_row(last_ts), kline_row(last_ts + 3600000), kline_row(last_ts + 2 * 3600000)]

    call_count = [0]

    def mock_fetch(symbol, interval, start_time, limit=LIMIT):
        call_count[0] += 1
        if call_count[0] == 1:
            return batch1
        elif call_count[0] == 2:
            return batch2
        return []

    out_path = str(tmp_path / "dedup.csv")

    # now_ms must be far enough in the future for the loop to proceed
    future_ts = (start_ms + (LIMIT + 10) * 3600000) / 1000
    with patch("scripts.download_binance_klines.fetch_klines", side_effect=mock_fetch):
        with patch("time.time", return_value=future_ts):
            with patch("time.sleep"):
                count = download_all("BTCUSDT", "1h", out_path)

    # LIMIT unique from batch1 + 2 new from batch2 (1 duplicate removed)
    assert count == LIMIT + 2


def test_download_all_creates_parent_dirs(tmp_path):
    from scripts.download_binance_klines import download_all

    def mock_fetch(symbol, interval, start_time, limit=LIMIT):
        return []

    out_path = str(tmp_path / "nested" / "dir" / "klines.csv")

    with patch("scripts.download_binance_klines.fetch_klines", side_effect=mock_fetch):
        with patch("time.time", return_value=1567900800 + 1):
            count = download_all("BTCUSDT", "1h", out_path)

    assert count == 0
    assert Path(out_path).exists()


# ── Constants ───────────────────────────────────────────────


def test_constants():
    assert BASE_URL == "https://fapi.binance.com"
    assert ENDPOINT == "/fapi/v1/klines"
    assert LIMIT == 1500
