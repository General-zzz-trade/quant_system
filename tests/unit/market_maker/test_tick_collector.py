from __future__ import annotations

import json
import sys
import types

from execution.market_maker.tick_collector import TickCollector, TickCollectorConfig


def _trade_msg() -> str:
    return json.dumps(
        {
            "data": {
                "e": "aggTrade",
                "T": 1710000000000,
                "a": 123,
                "p": "2000.0",
                "q": "0.5",
                "m": False,
            }
        }
    )


def _depth_msg() -> str:
    return json.dumps(
        {
            "data": {
                "E": 1710000000000,
                "u": 456,
                "b": [["2000.0", "1.0"], ["1999.5", "2.0"]],
                "a": [["2001.0", "1.5"], ["2001.5", "3.0"]],
            }
        }
    )


def test_trade_microstructure_failure_logs_once_and_preserves_last_values(caplog, monkeypatch, tmp_path):
    import execution.market_maker.tick_collector as collector_mod

    monkeypatch.setitem(sys.modules, "_quant_hotpath", types.ModuleType("_quant_hotpath"))
    collector = TickCollector(TickCollectorConfig(db_path=str(tmp_path / "ticks.db")))
    collector._micro = type(
        "Micro",
        (),
        {"on_trade": staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("trade failed")))},
    )()
    collector._last_vpin = 0.31
    collector._last_imbalance = -0.12

    monkeypatch.setattr(collector_mod.time, "monotonic", lambda: 100.0)
    with caplog.at_level("WARNING", logger=collector_mod.log.name):
        collector._on_trade(_trade_msg())
        collector._on_trade(_trade_msg())

    assert collector._last_vpin == 0.31
    assert collector._last_imbalance == -0.12
    assert caplog.text.count("Tick collector microstructure trade update failed; keeping previous state") == 1


def test_depth_microstructure_failure_logs_once_and_uses_previous_snapshot_values(caplog, monkeypatch, tmp_path):
    import execution.market_maker.tick_collector as collector_mod

    monkeypatch.setitem(sys.modules, "_quant_hotpath", types.ModuleType("_quant_hotpath"))
    collector = TickCollector(TickCollectorConfig(db_path=str(tmp_path / "ticks.db")))
    collector._micro = type(
        "Micro",
        (),
        {"on_depth": staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("depth failed")))},
    )()
    collector._last_vpin = 0.55
    collector._last_imbalance = 0.21

    monkeypatch.setattr(collector_mod.time, "monotonic", lambda: 100.0)
    with caplog.at_level("WARNING", logger=collector_mod.log.name):
        collector._on_depth(_depth_msg())
        collector._on_depth(_depth_msg())

    assert collector._last_vpin == 0.55
    assert collector._last_imbalance == 0.21
    assert len(collector._depth_batch) == 2
    first_row = collector._depth_batch[0]
    assert first_row[10] == 0.55
    assert first_row[11] == 0.21
    assert caplog.text.count("Tick collector microstructure depth update failed; keeping previous state") == 1
