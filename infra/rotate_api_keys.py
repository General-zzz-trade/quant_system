#!/usr/bin/env python3
"""API Key Rotation Helper.

Validates a new Binance API key pair before swapping it in.
Does NOT auto-deploy — prints instructions for manual credential update.

Usage:
    python3 -m scripts.rotate_api_keys --new-key XXXXX --new-secret YYYYY
    python3 -m scripts.rotate_api_keys --validate-only  # Test current keys
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

sys.path.insert(0, "/quant_system")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_api_key(api_key: str, api_secret: str, testnet: bool = False) -> dict:
    """Validate a Binance API key pair.

    Returns dict with validation results.
    """
    base_url = (
        "https://testnet.binancefuture.com"
        if testnet
        else "https://fapi.binance.com"
    )

    result = {
        "api_key": api_key[:8] + "..." + api_key[-4:],
        "testnet": testnet,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # 1. Connectivity
    try:
        req = urllib.request.Request(
            f"{base_url}/fapi/v1/ping",
            headers={"User-Agent": "quant_system/1.0"},
        )
        urllib.request.urlopen(req, timeout=10)
        result["connectivity"] = True
    except Exception as e:
        result["connectivity"] = False
        result["connectivity_error"] = str(e)
        return result

    # 2. Server time drift
    try:
        req = urllib.request.Request(
            f"{base_url}/fapi/v1/time",
            headers={"User-Agent": "quant_system/1.0"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        server_time = resp.get("serverTime", 0)
        local_time = int(time.time() * 1000)
        drift_ms = abs(server_time - local_time)
        result["time_drift_ms"] = drift_ms
        result["time_ok"] = drift_ms < 5000
    except Exception as e:
        result["time_error"] = str(e)

    # 3. Account access (signed request)
    try:
        timestamp = str(int(time.time() * 1000))
        params = f"timestamp={timestamp}"
        signature = hmac.new(
            api_secret.encode(), params.encode(), hashlib.sha256,
        ).hexdigest()

        url = f"{base_url}/fapi/v2/account?{params}&signature={signature}"
        req = urllib.request.Request(
            url,
            headers={
                "X-MBX-APIKEY": api_key,
                "User-Agent": "quant_system/1.0",
            },
        )
        resp_data = json.loads(urllib.request.urlopen(req, timeout=10).read())

        result["can_trade"] = resp_data.get("canTrade", False)
        result["account_type"] = resp_data.get("accountType", "unknown")

        for asset in resp_data.get("assets", []):
            if asset.get("asset") == "USDT":
                result["usdt_balance"] = float(asset.get("availableBalance", 0))
                break

        result["credentials_valid"] = True
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else ""
        result["credentials_valid"] = False
        result["credentials_error"] = f"HTTP {e.code}: {body[:200]}"
    except Exception as e:
        result["credentials_valid"] = False
        result["credentials_error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="API Key Rotation Helper")
    parser.add_argument("--new-key", default=None, help="New API key to validate")
    parser.add_argument("--new-secret", default=None, help="New API secret to validate")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate current keys from env")
    parser.add_argument("--testnet", action="store_true",
                        help="Validate against testnet")
    args = parser.parse_args()

    print("=" * 60)
    print("  BINANCE API KEY ROTATION HELPER")
    print(f"  Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    # Validate current keys
    current_key = os.environ.get("BINANCE_API_KEY", "")
    current_secret = os.environ.get("BINANCE_API_SECRET", "")

    if current_key and current_secret:
        print("\n-- Current Key Validation --")
        result = validate_api_key(current_key, current_secret, testnet=args.testnet)
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("\n  WARNING: No BINANCE_API_KEY/BINANCE_API_SECRET in environment")

    if args.validate_only:
        return 0

    # Validate new keys
    if args.new_key and args.new_secret:
        print("\n-- New Key Validation --")
        result = validate_api_key(args.new_key, args.new_secret, testnet=args.testnet)
        for k, v in result.items():
            print(f"  {k}: {v}")

        if result.get("credentials_valid") and result.get("can_trade"):
            print("\n  New key is VALID and has TRADING permission")
            print("\n  To rotate keys:")
            print("  1. Update environment variables:")
            print(f"     export BINANCE_API_KEY='{args.new_key}'")
            print(f"     export BINANCE_API_SECRET='{args.new_secret}'")
            print("  2. Restart the runner:")
            print("     sudo systemctl restart quant-runner")
            print("  3. Verify health:")
            print("     curl localhost:9090/health")
            print("  4. Revoke old key in Binance dashboard")
        else:
            print("\n  New key FAILED validation -- do NOT rotate")
            return 1
    elif not args.validate_only:
        print("\n  Usage: --new-key KEY --new-secret SECRET")
        print("  Or:    --validate-only (to check current keys)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
