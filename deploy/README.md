# Deploy Status

This directory contains deployment-related artifacts, but not every path here is
the current default release path.

## Default Truth Source

Use these paths first:

- repo-root `docker-compose.yml`
- `.github/workflows/ci.yml`
- `.github/workflows/deploy.yml`
- `scripts/deploy.sh`
- `runner/live_runner.py` for the default live runtime

This is the only default release path that should be assumed current.

## Candidate / Non-default Paths

- `deploy/docker/docker-compose.yml`
  Uses the repo-root `Dockerfile` and is kept as a non-default deployment example.
- `deploy/systemd/quant-trader.service`
  Candidate-production systemd unit for environments that do not use the default compose/GitHub Actions path.
- `deploy/systemd/logrotate-quant.conf`
  Candidate-production support file for the systemd path; not part of the default release path.
- `deploy/k8s/`
  Candidate-production Kubernetes manifests. These are not the current default release path.
- `deploy/argocd/`
  Experimental/candidate GitOps manifests. Some values remain placeholders and must not be treated as current production truth.

## Host Ops Support

- `deploy/grub-trading.cfg`
- `deploy/install-lowlatency.sh`
- `deploy/sysctl-trading.conf`
- `deploy/trading-tune.service`
- `deploy/tune-os.sh`

These are optional host-tuning helpers. They are not the default release path
and should not be interpreted as deploy truth.

## Rule

If a deploy artifact disagrees with the repo-root compose/workflow/runtime path, the
repo-root path wins unless another document explicitly promotes that artifact.
