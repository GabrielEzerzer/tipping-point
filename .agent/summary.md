# Summary: #29 S5, matched null-stretch rule (PR #150, draft)

**State: Waiting.** Both review seats approved. A full green gate needs the laptop, because the cloud sandbox
cannot install torch or build DuckDB. After that, the owner's merge is the only step left.

## What changed (54b189c..b1c330a)

- `market/msvol.py`: null stretches now follow "matched-calm-windows". Bar t is in a stretch iff:
  - H[t] == 0;
  - no event onset lies in [t-Q, t+Q];
  - t >= first+Q;
  - t < n-Q.

  Stretches are maximal runs of such bars. The old "refractory-from-episodes" rule is gone, with no option that keeps it.
- `ews/prereg.py`: `NULL_RULES = ("matched-calm-windows",)`.
- `paper/prereg/TEMPLATE-market-labels.toml`: `nulls.rule` changed. No frozen prereg file changed.
- Tests:
  - `test_msvol`: the boundary, Q out and Q+1 in;
  - `test_lookahead`: the finality bound `end <= T + 1 - d_min - Q` and a new edge case;
  - `test_prereg_market` and `test_reports`: updated.

## Suite (head b1c330ab0891538fbb351e99994e9988078c73f3, cloud, local run because the repo has no CI)

| Leg | Result |
|---|---|
| ruff-check | GREEN |
| ruff-format | GREEN, 142 files |
| cargo-fmt | GREEN |
| pyright | not clean. 40 errors: 39 are an unresolved `torch` import, and 1 is the Windows-only `subprocess.BELOW_NORMAL_PRIORITY_CLASS`. On the changed files, pyright reports 0 errors. |
| pytest | not clean. 655 passed, 33 skipped, 7 failed, 13 errors. Every failure and error is `No module named 'torch'`, in test_explore and the ising, jepa and latent tests, none of which this branch touches. All market and prereg tests pass. |
| self-test | not run. torch is missing. |
| cargo-clippy, cargo-test | not run. libduckdb-sys can't build in the sandbox, and the change is Python only. |

The torch wheel host (download-r2.pytorch.org) is blocked by this environment's network policy.
**Open: run `harness/gate.py` on the laptop on this head before merging.**

## Reviews

- `reviewer` (Claude Opus, fresh context), round 1: **APPROVE**, with all five success criteria met. It left two nits.
- `sol` (gpt-6.1-sol, high effort, Codex CLI), round 1: **APPROVE**, with no new findings. Codex ran with
  `-s workspace-write`, because the auto-mode classifier denied `danger-full-access`. It ran five focused tests, and they passed.
- No stand-in was needed.

## Follow-ups (not filed: the cloud has no `gh issue`)

1. (test) `test_msvol.py:186-204`: the planted-series test checks only C-length windows. Assert `member(t)` for
   every bar of every stretch, which is stronger and would catch a short spurious stretch.
2. (docs) `msvol.py:404`: `confirm = onset + d_min` is one bar later than the o+d_min-1 that the lookahead
   docstring uses for when an event is known. Either note this or align the two.
3. (env) The cloud environment cannot fetch torch from download-r2.pytorch.org or build libduckdb-sys, so
   cloud runs can't produce a green gate.

## Open decisions for Gabriel

- Whether to apply nits 1 and 2 on this PR or file them.
- Merge, after the laptop gate is green.
