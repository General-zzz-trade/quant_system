"""Read-only look at the live (mainnet) Binance USD-M futures account.

Shows the API key's permissions, balances, open positions, open orders, the leverage and margin mode of BTCUSDT
and ETHUSDT, the position mode and the fee rates.

    python3 -m scripts.account_check [--env-file PATH]

Allowed by the tech lead on 2026-11-15 as the one command Claude may run that reads the credentials file:
- it takes only BINANCE_API_KEY and BINANCE_API_SECRET from it, keeps them in memory and never prints or writes
  them (nor the request signature);
- it sends only GET requests, and only to the paths in ALLOWED; anything else raises before a request is made;
- it places no order and changes no account setting (no leverage, margin type or position mode call).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

FAPI = "https://fapi.binance.com"
SAPI = "https://api.binance.com"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
NAMES = ("BINANCE_API_KEY", "BINANCE_API_SECRET")
SYMBOLS = ("BTCUSDT", "ETHUSDT")
# (base, path, signed): every request this command can make. All are GET (read-only) endpoints.
ALLOWED = {
    (FAPI, "/fapi/v1/time", False),
    (FAPI, "/fapi/v2/account", True),
    (FAPI, "/fapi/v2/positionRisk", True),
    (FAPI, "/fapi/v1/openOrders", True),
    (FAPI, "/fapi/v1/openAlgoOrders", True),
    (FAPI, "/fapi/v1/positionSide/dual", True),
    (FAPI, "/fapi/v1/multiAssetsMargin", True),
    (FAPI, "/fapi/v1/feeBurn", True),
    (FAPI, "/fapi/v1/commissionRate", True),
    (FAPI, "/fapi/v1/apiTradingStatus", True),
    (SAPI, "/sapi/v1/account/apiRestrictions", True),
}

Fetch = Callable[[str, dict[str, str]], tuple[int, bytes]]


def read_credentials(path: Path) -> dict[str, str]:
    """Only the two mainnet key names; every other line of the file is ignored and never kept."""
    found: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        if sep and name.strip() in NAMES:
            found[name.strip()] = value.strip().strip("'\"")
    return found


def _urlopen(url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class ReadOnlyClient:
    def __init__(self, key: str, secret: str, *, fetch: Fetch = _urlopen, clock: Callable[[], float] = time.time):
        self._key, self._secret, self._fetch, self._clock = key, secret, fetch, clock

    def get(self, base: str, path: str, *, signed: bool = True, **params: Any) -> Any:
        if (base, path, signed) not in ALLOWED:
            raise PermissionError(f"not a read-only request on the allowlist: {base}{path}")
        headers = {}
        if signed:
            params = {**params, "recvWindow": 10000, "timestamp": int(self._clock() * 1000)}
            query = urllib.parse.urlencode(params)
            signature = hmac.new(self._secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            query = f"{query}&signature={signature}"
            headers["X-MBX-APIKEY"] = self._key
        else:
            query = urllib.parse.urlencode(params)
        status, body = self._fetch(f"{base}{path}" + (f"?{query}" if query else ""), headers)
        data = json.loads(body.decode() or "null")
        if status != 200:
            detail = data if isinstance(data, dict) else {}
            raise RuntimeError(f"HTTP {status}, Binance code {detail.get('code')}: {detail.get('msg')}")
        return data


def _section(out: list[str], title: str, produce: Callable[[], list[str]]) -> None:
    out.append(f"== {title}")
    try:
        out.extend(f"  {line}" for line in produce())
    except Exception as exc:  # one failing section never hides the others
        out.append(f"  unavailable: {type(exc).__name__}: {exc}")


def report(client: ReadOnlyClient) -> list[str]:
    out: list[str] = []

    def permissions() -> list[str]:
        r = client.get(SAPI, "/sapi/v1/account/apiRestrictions")
        keys = ("ipRestrict", "enableReading", "enableFutures", "enableWithdrawals", "enableInternalTransfer",
                "permitsUniversalTransfer", "enableSpotAndMarginTrading", "enableMargin", "enableVanillaOptions")
        return [f"{k}: {r.get(k)}" for k in keys]

    def clock() -> list[str]:
        server = client.get(FAPI, "/fapi/v1/time", signed=False)["serverTime"]
        return [f"server time {server}, local offset {int(time.time() * 1000) - server} ms"]

    def account() -> list[str]:
        a = client.get(FAPI, "/fapi/v2/account")
        lines = [f"{k}: {a.get(k)}" for k in ("canTrade", "canDeposit", "canWithdraw", "feeTier",
                                                "totalWalletBalance", "totalUnrealizedProfit", "totalMarginBalance",
                                                "totalInitialMargin", "totalMaintMargin", "availableBalance",
                                                "maxWithdrawAmount")]
        for asset in a.get("assets", []):
            if float(asset.get("walletBalance", 0) or 0) or float(asset.get("unrealizedProfit", 0) or 0):
                lines.append(f"asset {asset['asset']}: wallet {asset['walletBalance']}, "
                             f"unrealized {asset.get('unrealizedProfit')}, available {asset.get('availableBalance')}")
        return lines

    def modes() -> list[str]:
        dual = client.get(FAPI, "/fapi/v1/positionSide/dual").get("dualSidePosition")
        multi = client.get(FAPI, "/fapi/v1/multiAssetsMargin").get("multiAssetsMargin")
        burn = client.get(FAPI, "/fapi/v1/feeBurn").get("feeBurn")
        return [f"hedge mode (dualSidePosition): {dual}", f"multi-assets margin: {multi}", f"BNB fee burn: {burn}"]

    def positions() -> list[str]:
        rows = client.get(FAPI, "/fapi/v2/positionRisk")
        lines = []
        for p in rows:
            if float(p.get("positionAmt", 0) or 0):
                lines.append(f"OPEN {p['symbol']} {p.get('positionSide')}: qty {p['positionAmt']}, entry "
                             f"{p.get('entryPrice')}, mark {p.get('markPrice')}, "
                             f"unrealized {p.get('unRealizedProfit')}, liquidation {p.get('liquidationPrice')}, "
                             f"notional {p.get('notional')}, leverage "
                             f"{p.get('leverage')}x {p.get('marginType')}, isolated margin {p.get('isolatedMargin')}")
        lines.append(f"open positions: {len(lines)}")
        for p in rows:
            if p["symbol"] in SYMBOLS:
                lines.append(f"setting {p['symbol']} {p.get('positionSide')}: leverage {p.get('leverage')}x, "
                             f"margin {p.get('marginType')}")
        return lines

    def orders() -> list[str]:
        rows = client.get(FAPI, "/fapi/v1/openOrders")
        lines = [f"{o['symbol']} {o['side']} {o['type']} qty {o.get('origQty')} price {o.get('price')} stop "
                 f"{o.get('stopPrice')} reduceOnly {o.get('reduceOnly')} closePosition {o.get('closePosition')}"
                 for o in rows]
        return lines + [f"open orders: {len(rows)}"]

    def algo_orders() -> list[str]:
        r = client.get(FAPI, "/fapi/v1/openAlgoOrders")
        rows = r.get("orders", r) if isinstance(r, dict) else r
        lines = [f"{o.get('symbol')} {o.get('side')} {o.get('orderType') or o.get('type')} trigger "
                 f"{o.get('triggerPrice')} closePosition {o.get('closePosition')} status {o.get('algoStatus')}"
                 for o in rows]
        return lines + [f"open conditional (algo) orders: {len(rows)}"]

    def fees() -> list[str]:
        lines = []
        for symbol in SYMBOLS:
            r = client.get(FAPI, "/fapi/v1/commissionRate", symbol=symbol)
            lines.append(f"{symbol}: maker {r.get('makerCommissionRate')}, taker {r.get('takerCommissionRate')}")
        return lines

    def trading_status() -> list[str]:
        return [json.dumps(client.get(FAPI, "/fapi/v1/apiTradingStatus"), sort_keys=True)]

    for title, produce in (("API key permissions", permissions), ("clock", clock), ("futures account", account),
                           ("account modes", modes), ("positions and symbol settings", positions),
                           ("open orders", orders), ("open conditional orders", algo_orders),
                           ("fee rates", fees), ("quantitative trading status", trading_status)):
        _section(out, title, produce)
    return out


def main(argv: list[str] | None = None, *, fetch: Fetch = _urlopen, stdout=sys.stdout) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args = parser.parse_args(argv)
    try:
        found = read_credentials(args.env_file)
    except OSError as exc:
        print(f"credentials file not readable: {type(exc).__name__}", file=stdout)
        return 2
    missing = [n for n in NAMES if not found.get(n)]
    if missing:
        print(f"not set in the credentials file: {', '.join(missing)}", file=stdout)
        return 2
    print("mainnet, read-only (GET requests only; no order, no setting change)", file=stdout)
    for line in report(ReadOnlyClient(found[NAMES[0]], found[NAMES[1]], fetch=fetch)):
        print(line, file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
