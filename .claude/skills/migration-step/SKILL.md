---
name: migration-step
description: The checklist for one migration step or code change in this repository - branch, tests first, small commits, local checks, mutation self-check, execution-log entry, PR, CI, merge. Use whenever you change production code here (trading/, runner/, lab/, data/, infra/), including fixes for audit findings.
---

# One migration step, start to finish

Read `CLAUDE.md` first; its rules win over this list. Reply to the tech lead in Chinese, with evidence
tags 【实际运行结果】/【测试日志】/【推测】.

## Which process (the tech lead's decision of 2026-11-15)

- **Money path, full process:** exits (flatten, stops), risk gates (kill switch, drawdown, limits, pauses,
  exposure), the order state machine and journal, sizing, reconciliation and fills. Everything below applies.
- **Light process:** deletions, docs, tooling, test-only changes. Tests still pass locally for what you
  touched and CI must be green, but no mutation self-check, a one-paragraph log entry, merge on green.
- When unsure which applies, use the full process.

## Before writing code

1. Find the step and its approved decisions in `docs/arch/migration-plan.md`, `docs/arch/roadmap.md`
   (section 6) and `docs/adr/`. If a choice is uncertain or is a trade-off nobody decided, stop and ask
   the tech lead (AskUserQuestion, recommended option first). Do not assume.
2. Check that no other session is working in the shared checkout: `git status`, `git branch --show-current`,
   `ps aux | grep -c claude`. Then `git checkout master && git pull --ff-only` and
   `git checkout -b <short-step-name>`.
3. Never during an independent audit: then only deletions and docs are allowed.

## Tests first

- Write the new tests before the implementation. Run each new test once by itself
  (`python3 -m pytest <file>::<test> -q -p no:cacheprovider`) and see it fail for the right reason,
  then pass after the change. A test that was never run is not evidence.
- Never change a test assertion or a risk parameter to make a test pass. When a known-defect test flips
  (the defect is fixed), the log entry quotes the old and the new assertion.
- Test setup changes (fakes, fixtures, patch targets) are allowed but each one is logged.
- Tests never contact real exchanges or Telegram.

## Implementation

- Small commits, each under 400 handwritten lines and doing one thing (pure moves and deletions do not
  count). Strangler pattern: old import paths keep working through shells until M4.
- Invariants: risk limits, kill switch, no lookahead, the order state machine, backtest/live parity.
- Fail closed: unreadable pause, kill, drawdown or limit state blocks new exposure; unknown equity or
  price sizes to zero. No automatic path may raise a limit, release a pause, restart a service or swap a
  model. Model and spec artifacts load only through `infra/model_signing.py`.
- Commit messages end with the attribution line from the session instructions and contain no model
  identifier.

## Local checks (the full suite runs only in CI)

```bash
python3 -m pytest <the related test files> -q -p no:cacheprovider
ruff check --select E,W,F .
mypy-isolated trading/
lint-imports-isolated
python3 -m pytest tests/architecture -q -p no:cacheprovider
```

## Mutation self-check (money path only)

Write mutants for the new or changed production lines (`[label, file, old, new]`, optional fifth item =
the test that should catch it) and run
`python3 scripts/dev/mutate.py "<related tests>" <mutants.json> 2`. The baseline must pass. For every
survivor add or sharpen a test, rerun, and report the counts honestly (first round and final).

## Record

Add a section to `docs/arch/execution-log.md` with: 改了什么; 已有测试的改动 (exact, or "没有");
测试辅助的改动; 新增测试 (file and count); 变异自检 (numbers, survivors and what was added); 本地.
Correct `docs/arch/migration-plan.md`, `current-state.md`, `CLAUDE.md` where they disagree with the code.

## PR, CI, merge

1. `git push -u origin <branch>` (retry network errors with backoff). Force pushes and pushes to master
   are blocked by `.claude/hooks/guard.py`.
2. `gh pr create --base master --head <branch> --title ... --body-file <file>`; the body is in Chinese
   and ends with the PR attribution line. Edit a body with
   `gh api -X PATCH repos/General-zzz-trade/quant_system/pulls/N -F body=@file` (`gh pr edit` fails silently).
3. Poll `gh pr checks N` in a background command; keep working meanwhile.
4. Merge only when every check passes:
   `gh api -X PUT repos/General-zzz-trade/quant_system/pulls/N/merge -f merge_method=merge -f sha=<tested head>`,
   then `git push origin --delete <branch>`, `git checkout master && git pull --ff-only`,
   `git branch -d <branch>`.
5. Report to the tech lead in Chinese: what changed, evidence, what is left.
