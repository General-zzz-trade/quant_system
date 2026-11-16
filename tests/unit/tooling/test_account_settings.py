"""The settings report of the account check (tech lead 2026-11-16: I may not read the service's configuration file
myself; the owner asked for "which settings are missing"). ``--settings`` reads the file in memory, sends nothing,
prints only set / not set for every secret and the values of the plain settings, and runs the runtime's own
configuration checks on them: what the mainnet start would refuse, by name. Never contacts anything."""
from __future__ import annotations

import io

from scripts import account_check as ac

SECRETS = {"BINANCE_API_KEY": "KEY-0123456789abcdef", "BINANCE_API_SECRET": "SECRET-fedcba9876543210",
           "HEARTBEAT_URL": "https://hc.example.invalid/ping/PING-SECRET-42",
           "QUANT_MODEL_SIGN_KEY": "SIGN-KEY-SECRET-77"}
PILOT = {"BINANCE_TESTNET": "0", "RISK_SCOPE": "binance-mainnet-pilot", "STRATEGY_CAPITAL_USDT": "150",
         "STRATEGY_LEVERAGE": "2", "STOP_PCT": "0.045", "BINANCE_EXCHANGE_LEVERAGE": "10",
         "BINANCE_MARGIN_TYPE": "ISOLATED", "ALERT_FILE": "/var/lib/quant/alerts/mainnet.jsonl"}


def _report(tmp_path, values):
    path = tmp_path / "service-settings"
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()) + "# a comment\nexport OTHER_THING='x'\n")
    calls, out = [], io.StringIO()
    code = ac.main(["--env-file", str(path), "--settings"], fetch=lambda url, headers: calls.append(url), stdout=out)
    return code, out.getvalue(), calls


def test_secrets_are_reported_as_set_and_never_shown(tmp_path):
    code, text, calls = _report(tmp_path, {**SECRETS, **PILOT})
    assert calls == []                                                   # nothing leaves the host
    for name, value in SECRETS.items():
        assert f"{name}: set" in text and value not in text
    assert "PING-SECRET" not in text and "hc.example.invalid" not in text


def test_plain_settings_are_shown_with_their_values(tmp_path):
    code, text, _ = _report(tmp_path, {**SECRETS, **PILOT})
    for line in ("STRATEGY_CAPITAL_USDT: 150", "STRATEGY_LEVERAGE: 2", "STOP_PCT: 0.045",
                 "BINANCE_EXCHANGE_LEVERAGE: 10", "BINANCE_MARGIN_TYPE: ISOLATED", "RISK_SCOPE: binance-mainnet-pilot"):
        assert line in text
    assert "TELEGRAM_BOT_TOKEN: not set" in text
    assert "other names in the file: OTHER_THING" in text


def test_a_complete_pilot_configuration_passes_the_runtime_s_own_checks(tmp_path):
    code, text, _ = _report(tmp_path, {**SECRETS, **PILOT})
    assert code == 0
    assert "the runtime's configuration checks: passed" in text
    assert '"strategy_leverage": "2"' in text and '"stop_pct": "0.045"' in text and '"alerts": "file"' in text


def test_what_the_start_would_refuse_is_named(tmp_path):
    for broken, named in ((dict(STRATEGY_LEVERAGE="4"), "STRATEGY_LEVERAGE"),
                          (dict(STOP_PCT=None), "stop"),                  # 8 % on 10x isolated
                          (dict(ALERT_FILE=None), "ALERT_FILE"),
                          (dict(STRATEGY_CAPITAL_USDT=None, STRATEGY_LEVERAGE=None), "STRATEGY_CAPITAL_USDT")):
        values = {k: v for k, v in {**SECRETS, **PILOT, **broken}.items() if v is not None}
        code, text, _ = _report(tmp_path, values)
        assert code == 1 and "refused" in text and named in text, (broken, text)
        for value in SECRETS.values():
            assert value not in text


def test_a_missing_heartbeat_is_named(tmp_path):
    values = {k: v for k, v in {**SECRETS, **PILOT}.items() if k != "HEARTBEAT_URL"}
    code, text, _ = _report(tmp_path, values)
    assert code == 1 and "HEARTBEAT_URL" in text and "HEARTBEAT_URL: not set" in text


def test_an_unreadable_file_says_so(tmp_path):
    out = io.StringIO()
    assert ac.main(["--env-file", str(tmp_path / "missing"), "--settings"], stdout=out) == 2
    assert "not readable" in out.getvalue()
