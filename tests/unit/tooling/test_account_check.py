"""The read-only account check: GET requests on the allowlist only, and no credential in its output.

Runs against a fake transport; never contacts Binance.
"""
from __future__ import annotations

import io
import json
import urllib.parse

import pytest

from scripts import account_check as ac

KEY, SECRET = "fake-key-1234567890", "fake-secret-abcdefghij"

CANNED = {
    "/sapi/v1/account/apiRestrictions": {"ipRestrict": True, "enableReading": True, "enableFutures": True,
                                         "enableWithdrawals": False},
    "/fapi/v1/time": {"serverTime": 1_800_000_000_000},
    "/fapi/v2/account": {"canTrade": True, "totalWalletBalance": "167.5", "availableBalance": "150.1",
                         "assets": [{"asset": "USDT", "walletBalance": "167.5", "unrealizedProfit": "-1.2",
                                     "availableBalance": "150.1"},
                                    {"asset": "BNB", "walletBalance": "0", "unrealizedProfit": "0"}]},
    "/fapi/v1/positionSide/dual": {"dualSidePosition": False},
    "/fapi/v1/multiAssetsMargin": {"multiAssetsMargin": False},
    "/fapi/v1/feeBurn": {"feeBurn": False},
    "/fapi/v2/positionRisk": [
        {"symbol": "SOXLUSDT", "positionSide": "BOTH", "positionAmt": "-3", "entryPrice": "30", "markPrice": "31",
         "unRealizedProfit": "-3", "liquidationPrice": "80", "notional": "-93", "leverage": "5",
         "marginType": "cross", "isolatedMargin": "0"},
        {"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "0", "leverage": "50", "marginType": "cross"},
        {"symbol": "ETHUSDT", "positionSide": "BOTH", "positionAmt": "0", "leverage": "10",
         "marginType": "isolated"}],
    "/fapi/v1/openOrders": [],
    "/fapi/v1/openAlgoOrders": [],
    "/fapi/v1/commissionRate": {"makerCommissionRate": "0.0002", "takerCommissionRate": "0.0005"},
    "/fapi/v1/apiTradingStatus": {"indicators": {}},
}


class Venue:
    def __init__(self, errors=None):
        self.calls, self.errors = [], errors or {}

    def __call__(self, url, headers):
        parsed = urllib.parse.urlsplit(url)
        self.calls.append((parsed.scheme + "://" + parsed.netloc, parsed.path, parsed.query, dict(headers)))
        if parsed.path in self.errors:
            return 401, json.dumps({"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}).encode()
        return 200, json.dumps(CANNED[parsed.path]).encode()


def _env(tmp_path, text):
    path = tmp_path / "credentials"
    path.write_text(text)
    return path


def _run(tmp_path, venue, text=None):
    text = text if text is not None else (
        f"TELEGRAM_BOT_TOKEN=tg-not-read\nexport BINANCE_API_KEY='{KEY}'\nBINANCE_API_SECRET=\"{SECRET}\"\n")
    out = io.StringIO()
    code = ac.main(["--env-file", str(_env(tmp_path, text))], fetch=venue, stdout=out)
    return code, out.getvalue()


def test_only_the_two_key_names_are_taken_from_the_file(tmp_path):
    path = _env(tmp_path, f"# comment\nTELEGRAM_BOT_TOKEN=x\nexport BINANCE_API_KEY='{KEY}'\n"
                          f"BINANCE_API_SECRET=\"{SECRET}\"\nBINANCE_TESTNET_API_KEY=t\n")
    assert ac.read_credentials(path) == {"BINANCE_API_KEY": KEY, "BINANCE_API_SECRET": SECRET}


def test_every_request_is_an_allowlisted_read_and_nothing_secret_is_printed(tmp_path):
    venue = Venue()
    code, out = _run(tmp_path, venue)
    assert code == 0
    assert {(base, path) for base, path, _, _ in venue.calls} <= {(b, p) for b, p, _ in ac.ALLOWED}
    for base, path, query, headers in venue.calls:
        signed = (base, path, True) in ac.ALLOWED
        assert ("signature=" in query) is signed and (headers.get("X-MBX-APIKEY") == KEY) is signed
        assert SECRET not in query
    signatures = [urllib.parse.parse_qs(q)["signature"][0] for _, _, q, _ in venue.calls if "signature=" in q]
    for secret_like in [KEY, SECRET, "tg-not-read", *signatures]:
        assert secret_like not in out
    assert "OPEN SOXLUSDT BOTH: qty -3" in out and "open positions: 1" in out
    assert "setting BTCUSDT BOTH: leverage 50x, margin cross" in out
    assert "enableWithdrawals: False" in out and "BTCUSDT: maker 0.0002, taker 0.0005" in out


def test_the_client_refuses_anything_off_the_allowlist_before_sending():
    venue = Venue()
    client = ac.ReadOnlyClient(KEY, SECRET, fetch=venue)
    for base, path in [(ac.FAPI, "/fapi/v1/order"), (ac.FAPI, "/fapi/v1/leverage"), (ac.FAPI, "/fapi/v1/marginType"),
                       (ac.FAPI, "/fapi/v1/algoOrder"), (ac.SAPI, "/sapi/v1/capital/withdraw/apply")]:
        with pytest.raises(PermissionError):
            client.get(base, path)
    assert venue.calls == []


def test_the_transport_only_ever_sends_get(monkeypatch):
    seen = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(ac.urllib.request, "urlopen", lambda request, timeout: seen.append(request.method)
                        or Response())
    assert ac._urlopen("https://fapi.binance.com/fapi/v1/time", {}) == (200, b"{}")
    assert seen == ["GET"]


def test_missing_credentials_make_no_request_and_name_only_the_variables(tmp_path):
    venue = Venue()
    code, out = _run(tmp_path, venue, text=f"BINANCE_API_KEY={KEY}\n")
    assert code == 2 and venue.calls == []
    assert "BINANCE_API_SECRET" in out and KEY not in out


def test_a_refused_section_is_reported_and_the_others_still_run(tmp_path):
    venue = Venue(errors={"/sapi/v1/account/apiRestrictions"})
    code, out = _run(tmp_path, venue)
    assert code == 0
    assert "unavailable: RuntimeError: HTTP 401, Binance code -2015" in out
    assert "totalWalletBalance: 167.5" in out
