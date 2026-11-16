# Mutants of the mutation self-checks

One file per step, as `scripts/dev/mutate.py` reads it: a JSON list of `[label, file, old, new]`
(an optional fifth item names the test expected to catch it). Rerun one with

    python3 scripts/dev/mutate.py "<the step's test files>" scripts/dev/mutants/<step>.json 2

The test files and the counts (first round and final) are in the step's section of `docs/arch/execution-log.md`.
A mutant whose `old` text no longer exists in the code is stale: the code moved on since that step.
