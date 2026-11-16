"""The failure alert of the systemd units (ADR-0016 point 4): ``quant-alert@<unit>.service`` runs

    python -m trading.infrastructure.unit_alert <unit>

when a unit fails (``OnFailure=``), including a failure that systemd then restarts. It sends one Telegram message
and waits for Telegram's confirmation. Exit 0 when confirmed, 1 when not (systemd then shows the alert unit failed
too, so the failure is visible on the host), 2 when Telegram is not configured. Never prints the token.
"""
from __future__ import annotations

import argparse
import socket
import sys
from collections.abc import Callable, Mapping
from typing import Any, TextIO

from trading.adapters.notify.telegram import TelegramNotifier
from trading.infrastructure.config import notifier_from_env

TIMEOUT_SECONDS = 20.0


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None, stdout: TextIO = sys.stdout,
         make: Callable[[Mapping[str, str] | None], Any] = notifier_from_env) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("unit", help="the unit that failed (systemd passes %%i)")
    args = parser.parse_args(argv)
    try:
        notifier = make(env)
    except RuntimeError as exc:                    # half a configuration: the message names the variables only
        print(f"no alert sent: {exc}", file=stdout)
        return 2
    if not isinstance(notifier, TelegramNotifier):
        print("no alert sent: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are not set", file=stdout)
        return 2
    title = f"{args.unit} failed on {socket.gethostname()}"
    detail = ("systemd reports the unit failed. The testnet unit restarts by itself at most 3 times an hour; the "
              "mainnet unit is never restarted. Read its log, then a person decides what happens next.")
    if notifier.deliver(title, detail, timeout_s=TIMEOUT_SECONDS):
        print(f"alert sent: {title}", file=stdout)
        return 0
    print(f"alert not confirmed by Telegram: {title}", file=stdout)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
