#!/usr/bin/env python3
"""Interactive OKX credential setup — writes to /quant_system/.env without echo.

Usage:
    python3 /quant_system/scripts/setup_okx_credentials.py

Behavior:
  1. Prompts for OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE using
     getpass (input is NOT echoed to the terminal or saved anywhere).
  2. Optionally tests the credentials immediately by calling
     /api/v5/account/balance.
  3. Preserves every non-OKX_* line in /quant_system/.env.
  4. Writes atomically via .env.tmp → .env rename.
  5. chmod 600 /quant_system/.env at the end.

Does NOT print the key/secret/passphrase at any point. Does NOT log them.
Re-running the script updates the existing OKX_* lines in-place.
"""
from __future__ import annotations

import os
import re
import stat
import sys
from getpass import getpass
from pathlib import Path
from typing import Tuple

ENV_PATH = Path("/quant_system/.env")

OKX_KEYS = (
    "OKX_API_KEY",
    "OKX_API_SECRET",
    "OKX_API_PASSPHRASE",
    "OKX_BASE_URL",
    "OKX_MAX_ORDER_NOTIONAL",
    "OKX_MAX_DAILY_NOTIONAL",
    "OKX_MAX_RATE_PER_SEC",
)


def _prompt_secret(label: str, min_len: int = 1) -> str:
    """Prompt once via getpass, re-prompt if empty or clearly invalid."""
    while True:
        val = getpass(f"{label}: ").strip()
        if not val:
            print("  (empty — please try again)")
            continue
        if len(val) < min_len:
            print(f"  (too short: {len(val)} chars — please check)")
            continue
        return val


def _read_env() -> list[str]:
    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(encoding="utf-8").splitlines()


def _write_env_atomic(lines: list[str]) -> None:
    tmp = ENV_PATH.with_suffix(".env.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600 before rename
    tmp.replace(ENV_PATH)
    os.chmod(ENV_PATH, stat.S_IRUSR | stat.S_IWUSR)


def _merge_env(existing: list[str], new_values: dict[str, str]) -> list[str]:
    """Replace or append each OKX_* key; leave everything else untouched."""
    seen: set[str] = set()
    out: list[str] = []
    pattern = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=")

    for line in existing:
        m = pattern.match(line)
        if m and m.group(1) in new_values:
            key = m.group(1)
            out.append(f"{key}={new_values[key]}")
            seen.add(key)
        else:
            out.append(line)

    # Append any keys that weren't in the file
    to_append = [k for k in OKX_KEYS if k in new_values and k not in seen]
    if to_append:
        # blank line + header for readability
        if out and out[-1].strip() != "":
            out.append("")
        out.append("# OKX USDT SWAP (live trading)")
        for k in to_append:
            out.append(f"{k}={new_values[k]}")
    return out


def _test_credentials(
    api_key: str, api_secret: str, passphrase: str, base_url: str,
) -> Tuple[bool, str]:
    """Call /api/v5/account/balance with the supplied creds.  Returns (ok, msg)."""
    try:
        sys.path.insert(0, "/quant_system")
        from execution.adapters.okx.rest import (
            OkxRestClient,
            OkxRestConfig,
            OkxNonRetryableError,
        )
    except Exception as e:
        return False, f"import error: {e}"

    cfg = OkxRestConfig(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        base_url=base_url,
    )
    client = OkxRestClient(cfg)
    try:
        resp = client.request_signed(
            method="GET",
            path="/api/v5/account/balance",
            params={"ccy": "USDT"},
        )
    except OkxNonRetryableError as e:
        return False, f"auth/business error: {e}"
    except Exception as e:
        return False, f"network error: {e}"

    data = resp.get("data") or []
    if not data:
        return True, "no USDT balance detail (account OK but empty)"
    details = data[0].get("details", [])
    for d in details:
        if d.get("ccy") == "USDT":
            return True, f"USDT eq={d.get('eq')} avail={d.get('availBal')}"
    return True, "USDT detail absent (account OK)"


def main() -> int:
    print("=" * 60)
    print("  OKX credential setup for /quant_system/.env")
    print("=" * 60)
    print("Input is NOT echoed. Paste + Enter after each prompt.")
    print("Ctrl+C aborts without writing anything.\n")

    try:
        api_key = _prompt_secret("OKX_API_KEY", min_len=20)
        api_secret = _prompt_secret("OKX_API_SECRET", min_len=20)
        passphrase = _prompt_secret("OKX_API_PASSPHRASE", min_len=4)
    except KeyboardInterrupt:
        print("\nAborted — no changes written.")
        return 130

    base_url = os.environ.get("OKX_BASE_URL", "https://www.okx.com")

    # Optional live auth check BEFORE writing to disk
    print("\nTesting credentials against OKX /account/balance ...")
    ok, msg = _test_credentials(api_key, api_secret, passphrase, base_url)
    if not ok:
        print(f"  ✗ FAIL: {msg}")
        print("Aborted — credentials NOT written to .env")
        return 1
    print(f"  ✓ OK: {msg}")

    new_values = {
        "OKX_API_KEY": api_key,
        "OKX_API_SECRET": api_secret,
        "OKX_API_PASSPHRASE": passphrase,
        "OKX_BASE_URL": base_url,
        # Conservative first-24h caps.  Raise only after ≥5 successful trades.
        "OKX_MAX_ORDER_NOTIONAL": "50",
        "OKX_MAX_DAILY_NOTIONAL": "500",
        "OKX_MAX_RATE_PER_SEC": "2",
    }

    existing = _read_env()
    merged = _merge_env(existing, new_values)
    _write_env_atomic(merged)

    # Confirm permissions
    st = ENV_PATH.stat()
    perm = stat.S_IMODE(st.st_mode)
    print(f"\n✓ Wrote {ENV_PATH}  (mode={oct(perm)})")
    print("  OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE set")
    print(f"  Safety caps: order=${new_values['OKX_MAX_ORDER_NOTIONAL']}, "
          f"daily=${new_values['OKX_MAX_DAILY_NOTIONAL']}")
    print("\nNext step: tell Claude '已完成' and we run the live smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
