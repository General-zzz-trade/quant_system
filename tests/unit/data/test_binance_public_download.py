"""Binance public downloader: closed bars only, causal funding marks, checksums."""
from __future__ import annotations

import hashlib
import io
import json
import urllib.parse
import zipfile
from datetime import date

import pytest

from data.downloads import binance_public as bp

H = bp.HOUR_MS
T0 = 1_700_000_000_000 // H * H


def _kline(open_time):
    return [open_time, "100", "101", "99", "100.5", "2", open_time + H - 1, "201", 7, "1", "100.5", "0"]


def _fake(routes):
    def fetch(url):
        for prefix, handler in routes:
            if url.startswith(prefix):
                return handler(url)
        raise AssertionError(f"unexpected url {url}")
    return fetch


def _query(url):
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}


def test_klines_drop_unclosed_bar_and_paginate():
    def klines(url):
        q = _query(url)
        start = int(q["startTime"])
        return json.dumps([_kline(t) for t in range(start, min(start + 2 * H, int(q["endTime"]) + 1), H)]).encode()

    rows = bp.fetch_klines("BTCUSDT", T0, T0 + 5 * H, now_ms=T0 + 4 * H + 10,
                           fetch=_fake([(bp.FAPI + "/fapi/v1/klines", klines)]))
    # Bar opening at T0+4H closes at T0+5H-1 > now: it must not be stored.
    assert [r[0] for r in rows] == [T0, T0 + H, T0 + 2 * H, T0 + 3 * H]
    assert rows[0][7] == 7  # trades column kept as int


def test_kline_with_wrong_boundaries_is_rejected():
    bad = _kline(T0)
    bad[6] = T0 + 10

    with pytest.raises(ValueError, match="boundaries"):
        bp.fetch_klines("BTCUSDT", T0, T0 + H, now_ms=T0 + 10 * H,
                        fetch=_fake([(bp.FAPI, lambda url: json.dumps([bad]).encode())]))


def test_funding_alignment_and_mark_proxy():
    events = [
        {"fundingTime": T0 + 8, "fundingRate": "0.0001", "markPrice": "42000"},
        {"fundingTime": T0 + 8 * H, "fundingRate": "-0.0002", "markPrice": ""},
    ]
    rows = bp.normalize_funding(events, {T0 + 8 * H: "43000"})
    assert rows[0] == (T0, "0.0001", "42000", "exchange", T0 + 8)
    assert rows[1] == (T0 + 8 * H, "-0.0002", "43000", "kline_open_proxy", T0 + 8 * H)


def test_funding_far_from_boundary_or_without_mark_is_rejected():
    with pytest.raises(ValueError, match="hour boundary"):
        bp.normalize_funding([{"fundingTime": T0 + 5000, "fundingRate": "0", "markPrice": "1"}], {})
    with pytest.raises(ValueError, match="no mark price"):
        bp.normalize_funding([{"fundingTime": T0, "fundingRate": "0", "markPrice": ""}], {})


def _metrics_zip(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-metrics-2024-01-01.csv", text)
    return buffer.getvalue()


def test_metrics_checksum_is_verified():
    text = ("create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
            "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
            "2024-01-01 00:05:00,BTCUSDT,100,4000000,1.1,1.2,1.3,0.9\n")
    payload = _metrics_zip(text)
    good = hashlib.sha256(payload).hexdigest().encode() + b"  BTCUSDT-metrics-2024-01-01.zip\n"
    routes = [(bp.VISION, lambda url: good if url.endswith(".CHECKSUM") else payload)]
    rows = bp.fetch_metrics_day("BTCUSDT", date(2024, 1, 1), fetch=_fake(routes))
    assert rows == [(1704067500000, "100", "4000000", "1.1", "1.2", "1.3", "0.9")]

    routes = [(bp.VISION, lambda url: b"0" * 64 if url.endswith(".CHECKSUM") else payload)]
    with pytest.raises(ValueError, match="checksum"):
        bp.fetch_metrics_day("BTCUSDT", date(2024, 1, 1), fetch=_fake(routes))
