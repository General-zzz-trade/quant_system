#!/usr/bin/env bash
# Compose paper only. Does not install/restart host systemd units or build images.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

check_only=false
if [[ "${1:-}" == --check ]]; then
    check_only=true
    shift
fi
if [[ "$#" -ne 1 || "$1" != quant-paper ]]; then
    echo 'Usage: bash scripts/deploy.sh [--check] quant-paper (only supported service)' >&2
    exit 2
fi

# Fail before Docker mutations when the checked-in runtime entry is unavailable.
python3 - <<'PY'
from pathlib import Path
for name in ("docker-compose.yml", "scripts/run_bybit_alpha.py", "scripts/check_paper_runtime.py"):
    path = Path(name)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"Deployment blocked: required runtime file unavailable: {name}")
PY
docker compose config --quiet
docker compose config --format json | python3 -m scripts.check_paper_runtime
if "$check_only"; then
    echo 'Paper deployment preflight passed; no services changed'
    exit 0
fi
: "${QUANT_RELEASE_REVISION:?Exact reviewed revision is required}"
: "${QUANT_RELEASE_REPOSITORY:?Release repository is required}"
mkdir -p .cache/release
release_preflight=.cache/release/preflight.json
release_status=.cache/release/runtime-status.json
release_report=${QUANT_RELEASE_REPORT:-.cache/release/postflight.json}

docker compose config --format json | python3 -m scripts.check_release_acceptance \
    --revision "$QUANT_RELEASE_REVISION" --repository "$QUANT_RELEASE_REPOSITORY" \
    --image "${QUANT_RELEASE_IMAGE:-quant-paper:latest}" --output "$release_preflight"

# Rollback may reuse a verified image only with the same model/data bytes.
# No service changes happen if the previous release cannot be reproduced.
python3 - "$release_preflight" "${QUANT_RELEASE_BASELINE:-}" <<'PY'
import json
from pathlib import Path
import sys
current = json.loads(Path(sys.argv[1]).read_text())
if sys.argv[2]:
    prior = json.loads(Path(sys.argv[2]).read_text())
    if prior.get("paper_accepted") is not True or prior.get("release") != current["release"]:
        raise SystemExit("Rollback blocked: previous accepted release inputs/image cannot be reproduced")
PY
read -r QUANT_RELEASE_IMAGE_ID QUANT_RELEASE_INPUT_DIGEST < <(
    python3 - "$release_preflight" <<'PY'
import json
from pathlib import Path
import sys
release = json.loads(Path(sys.argv[1]).read_text())["release"]
print(release["image_id"], release["input_digest"])
PY
)
# Pin the exact inspected image even if a mutable tag changes concurrently.
export QUANT_RELEASE_IMAGE_ID QUANT_RELEASE_INPUT_DIGEST
export QUANT_RELEASE_IMAGE="$QUANT_RELEASE_IMAGE_ID"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo 'service_change_started=true' >> "$GITHUB_OUTPUT"
fi
docker compose up -d --no-deps --no-build --force-recreate --wait --wait-timeout 120 quant-paper
docker compose exec -T quant-paper python3 -m scripts.check_runtime_health --require ready \
    --expected-revision "$QUANT_RELEASE_REVISION" --expected-image-id "$QUANT_RELEASE_IMAGE_ID" \
    --expected-input-digest "$QUANT_RELEASE_INPUT_DIGEST"
docker compose exec -T quant-paper cat /app/data/runtime/runtime-status.json > "$release_status"
docker compose config --format json | python3 -m scripts.check_release_acceptance \
    --revision "$QUANT_RELEASE_REVISION" --repository "$QUANT_RELEASE_REPOSITORY" \
    --image "$QUANT_RELEASE_IMAGE_ID" --output "$release_report" \
    --preflight "$release_preflight" --runtime-status "$release_status"
