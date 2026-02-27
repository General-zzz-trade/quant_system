"""Tests for paper runner live mode integration."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest

from runner.paper_runner import PaperRunnerConfig, run_paper_live


@dataclass
class FakeTransport:
    """In-memory WsTransport that replays pre-loaded messages."""
    messages: List[str] = field(default_factory=list)
    connected_url: Optional[str] = None
    _idx: int = 0

    def connect(self, url: str) -> None:
        self.connected_url = url

    def recv(self, *, timeout_s: Optional[float] = None) -> str:
        if self._idx < len(self.messages):
            msg = self.messages[self._idx]
            self._idx += 1
            return msg
        raise KeyboardInterrupt

    def close(self) -> None:
        self.connected_url = None


def _make_kline(
    ts_ms: int,
    close: str = "50000.00",
    symbol: str = "BTCUSDT",
) -> str:
    return json.dumps({
        "stream": f"{symbol.lower()}@kline_1m",
        "data": {
            "e": "kline",
            "E": ts_ms + 59999,
            "s": symbol,
            "k": {
                "t": ts_ms,
                "o": close,
                "h": str(Decimal(close) + 10),
                "l": str(Decimal(close) - 10),
                "c": close,
                "v": "100.0",
                "x": True,
            },
        },
    })


class TestRunPaperLive:

    def test_processes_klines_through_coordinator(self) -> None:
        """Feed klines via mock transport -> coordinator processes them."""
        prices = ["100.00", "101.00", "102.00", "103.00", "104.00",
                   "105.00", "106.00", "107.00", "108.00", "109.00"]
        messages = [
            _make_kline(ts_ms=1700000000000 + i * 60000, close=p)
            for i, p in enumerate(prices)
        ]
        transport = FakeTransport(messages=messages)

        config = PaperRunnerConfig(
            symbol="BTCUSDT",
            starting_balance=Decimal("10000"),
            order_qty=Decimal("0.01"),
            ma_window=3,
            fee_bps=Decimal("4"),
            slippage_bps=Decimal("2"),
            log_interval_sec=9999,
        )

        run_paper_live(config, _transport=transport)

    def test_empty_stream_exits_cleanly(self) -> None:
        """No messages -> KeyboardInterrupt -> clean shutdown."""
        transport = FakeTransport(messages=[])

        config = PaperRunnerConfig(
            symbol="BTCUSDT",
            starting_balance=Decimal("10000"),
            order_qty=Decimal("0.01"),
            ma_window=3,
        )

        run_paper_live(config, _transport=transport)
