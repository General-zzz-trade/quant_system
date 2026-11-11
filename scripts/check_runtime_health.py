"""Read runtime evidence only; never instantiate an adapter or contact accounts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from shared.runtime_status import STATUS_NAME, evaluate_status, read_json


def check_runtime_health(state_dir: Path, *, scope: str, expected_release: dict | None = None,
                         now: float | None = None, check_pid: bool = True) -> dict:
    manifest = read_json(state_dir / "runtime-state.json")
    if type(manifest.get("version")) is not int or manifest["version"] != 1 or manifest.get("scope") != scope:
        raise ValueError("Runtime manifest does not match configured health scope")
    if not isinstance(manifest.get("state_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", manifest["state_id"]):
        raise ValueError("Runtime manifest needs an explicit valid state identity")
    return evaluate_status(read_json(state_dir / STATUS_NAME), now=now, expected_scope=scope,
                           expected_state_id=manifest.get("state_id"),
                           expected_release=expected_release, check_pid=check_pid)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=os.environ.get("RUNTIME_STATE_DIR"))
    parser.add_argument("--scope", default=os.environ.get("RISK_SCOPE"))
    parser.add_argument("--require", choices=("live", "ready", "execution"), default="ready")
    parser.add_argument("--expected-revision", default=os.environ.get("QUANT_RELEASE_REVISION"))
    parser.add_argument("--expected-image-id", default=os.environ.get("QUANT_RELEASE_IMAGE_ID"))
    parser.add_argument("--expected-input-digest", default=os.environ.get("QUANT_RELEASE_INPUT_DIGEST"))
    args = parser.parse_args()
    try:
        if not args.state_dir or not args.scope:
            raise ValueError("Runtime health requires explicit state directory and scope")
        release = {"revision": args.expected_revision, "image_id": args.expected_image_id,
                   "input_digest": args.expected_input_digest}
        if not any(release.values()):
            release = None
        result = check_runtime_health(args.state_dir, scope=args.scope, expected_release=release)
        passed = result["execution_eligible" if args.require == "execution" else args.require]
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        result = {"live": False, "ready": False, "execution_eligible": False,
                  "reasons": ["runtime_status_invalid"], "error_type": type(exc).__name__}
        passed = False
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
