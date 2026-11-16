"""The production entry point (``runner.binance_target.main``) takes the stop distance from STOP_PCT when no
``--stop-pct`` is given, 8 % when neither is set, and the flag over STOP_PCT (owner 2026-11-16: the mainnet unit passes
no flag and the pilot's 4.5 % comes from STOP_PCT). Adapted from the 2026-11-16 audit's reproduction: the run stops at
the pre-flight check, before anything contacts an exchange."""
from __future__ import annotations

from decimal import Decimal as D

import pytest


class Stop(Exception):
    pass


@pytest.mark.parametrize("argv_extra,env_stop,expected", [([], "0.045", D("0.045")), ([], None, D("0.08")),
                                                          (["--stop-pct", "0.03"], "0.045", D("0.03"))])
def test_the_entry_point_takes_stop_pct(tmp_path, monkeypatch, argv_extra, env_stop, expected):
    from trading.infrastructure.cli import target as cli
    import runner.binance_target as entry
    seen = {}

    def preflight(spec, *, stop_pct, config, **kw):
        seen["stop"], seen["config"] = stop_pct, config.stop_pct
        raise Stop()
    monkeypatch.setattr(cli, "preflight", preflight)
    for name, value in (("BINANCE_TESTNET", "0"), ("RISK_SCOPE", "binance-ut"),
                        ("HEARTBEAT_URL", "https://h.invalid/x"), ("STRATEGY_CAPITAL_USDT", "150"),
                        ("BINANCE_EXCHANGE_LEVERAGE", "10"), ("BINANCE_MARGIN_TYPE", "ISOLATED"),
                        ("STRATEGY_LEVERAGE", "2")):
        monkeypatch.setenv(name, value)
    if env_stop is None:
        monkeypatch.delenv("STOP_PCT", raising=False)
    else:
        monkeypatch.setenv("STOP_PCT", env_stop)
    with pytest.raises(Stop):
        entry.main(["--spec", str(tmp_path / "spec.json"), "--enable-execution", "--state-dir", str(tmp_path / "s"),
                    *argv_extra])
    assert seen["stop"] == seen["config"] == expected
