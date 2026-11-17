#!/usr/bin/env bash
# Run a research job behind the trading process on a shared host (2026-11-17): the lowest CPU priority and idle disk
# priority, so a backtest never slows the runtime (whose units run at Nice=-5, CPUWeight=1000, IOWeight=1000).
#   scripts/dev/lab_nice.sh python3 research/lab/runs/<date>/judge.py screen
set -euo pipefail
exec nice -n 19 ionice -c3 "$@"
