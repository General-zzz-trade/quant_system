#!/usr/bin/env python3
"""Security scan — checks for common security issues in the codebase.

Verifies:
  1. No hardcoded secrets in Python files
  2. .env is gitignored (not tracked)
  3. MAX_ORDER_NOTIONAL hard limit exists
  4. No bare except: blocks (except-pass pattern)
  5. Dockerfile exists for CI

Usage:
    python3 -m scripts.ops.security_scan
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# Patterns that suggest hardcoded secrets
_SECRET_PATTERNS = [
    re.compile(r"""(?:api_key|api_secret|password|token)\s*=\s*['"][A-Za-z0-9+/=]{16,}['"]""", re.I),
    re.compile(r"""(?:BYBIT|BINANCE|POLYMARKET)_API_(?:KEY|SECRET)\s*=\s*['"][^'"]{10,}['"]""", re.I),
    re.compile(r"""Bearer\s+[A-Za-z0-9._-]{20,}"""),
]

# Files/dirs to skip
_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules", ".venv"}
_SKIP_FILES = {".env.example", "security_scan.py"}


def _scan_hardcoded_secrets() -> list[str]:
    """Scan .py files for hardcoded secret patterns."""
    issues = []
    for root, dirs, files in os.walk(_ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            if not fname.endswith(".py") or fname in _SKIP_FILES:
                continue
            fpath = Path(root) / fname
            try:
                text = fpath.read_text(errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                # Skip comments and print/log statements (format strings)
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if "print(" in stripped or "logger." in stripped:
                    continue
                for pat in _SECRET_PATTERNS:
                    if pat.search(line):
                        rel = fpath.relative_to(_ROOT)
                        issues.append(f"  {rel}:{i}: possible hardcoded secret")
    return issues


def _check_env_gitignored() -> bool:
    """Verify .env is in .gitignore and not tracked."""
    gitignore = _ROOT / ".gitignore"
    if not gitignore.exists():
        return False
    text = gitignore.read_text()
    if ".env" not in text:
        return False
    # Check not tracked
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=_ROOT, capture_output=True, text=True,
        )
        if result.returncode == 0:
            return False  # .env IS tracked — bad
    except FileNotFoundError:
        pass
    return True


def _check_max_order_notional() -> tuple[bool, float | None]:
    """Verify MAX_ORDER_NOTIONAL exists and has a reasonable value."""
    config_path = _ROOT / "scripts" / "ops" / "config.py"
    if not config_path.exists():
        return False, None
    text = config_path.read_text()
    match = re.search(r"MAX_ORDER_NOTIONAL\s*=\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return False, None
    val = float(match.group(1))
    return val <= 1000, val  # Should be <= $1000 for safety


def _check_bare_except() -> list[str]:
    """Find bare except: or except-pass blocks."""
    issues = []
    for root, dirs, files in os.walk(_ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = Path(root) / fname
            try:
                text = fpath.read_text(errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped == "except:":
                    rel = fpath.relative_to(_ROOT)
                    issues.append(f"  {rel}:{i}: bare except:")
    return issues


def _check_dockerfile_exists() -> bool:
    return (_ROOT / "Dockerfile").exists()


def main() -> None:
    print("=" * 60)
    print("  SECURITY SCAN")
    print("=" * 60)
    all_pass = True

    # 1. Hardcoded secrets
    print("\n[1] Hardcoded secrets scan...")
    secrets = _scan_hardcoded_secrets()
    if secrets:
        print("  FAIL: Found potential hardcoded secrets:")
        for s in secrets[:10]:
            print(s)
        all_pass = False
    else:
        print("  PASS: No hardcoded secrets found")

    # 2. .env gitignored
    print("\n[2] .env gitignore check...")
    if _check_env_gitignored():
        print("  PASS: .env is gitignored and not tracked")
    else:
        print("  FAIL: .env may be tracked or not in .gitignore")
        all_pass = False

    # 3. MAX_ORDER_NOTIONAL
    print("\n[3] MAX_ORDER_NOTIONAL check...")
    ok, val = _check_max_order_notional()
    if ok:
        print(f"  PASS: MAX_ORDER_NOTIONAL = ${val:.0f}")
    else:
        print(f"  FAIL: MAX_ORDER_NOTIONAL {'= $' + str(val) + ' (too high)' if val else 'not found'}")
        all_pass = False

    # 4. Bare except blocks
    print("\n[4] Bare except: scan...")
    bare = _check_bare_except()
    if bare:
        print(f"  WARN: Found {len(bare)} bare except: blocks")
        for b in bare[:5]:
            print(b)
    else:
        print("  PASS: No bare except: blocks")

    # 5. Dockerfile
    print("\n[5] Dockerfile check...")
    if _check_dockerfile_exists():
        print("  PASS: Dockerfile exists")
    else:
        print("  FAIL: Dockerfile missing (required by CI)")
        all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("  OVERALL: PASS")
    else:
        print("  OVERALL: FAIL")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
