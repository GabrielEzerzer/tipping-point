# Journal: #29 S5, matched null-stretch rule

## Goal and reading

reading: away · done · feature · logic

Goal: the market labeller's null stretches (the controls for the #30 warning AUC) use the
owner-decided rule "matched-calm-windows", and the PR passes the review gate. The finish line is
Waiting: a draft PR with reviews clean, which the owner merges.

Owner decision, recorded on issue #29 (comment 6664595938): the matched rule plus D_min 36. Do not
fix the labeller now. The labeller's high state covers ordinary intraday activity (about 47% of
bars), which is a known limitation filed as #149. Changing it is out of scope.

## The rule

Bar t is in a null stretch if and only if all of these hold:
- H[t] == 0;
- no event onset o has t-Q <= o <= t+Q;
- t >= first+Q;
- t < n-Q.

Stretches are the maximal runs of such bars. Any C = 72-bar window inside a stretch is calm, with no
event onset within Q = 288 bars either side. The old rule, "refractory-from-episodes", is retired.

## State at handoff (2026-11-05, laptop)

- Branch `feat/29-matched-nulls` on base main 54b189c. The builder commit is 87decc5. It touches:
  - `market/msvol.py`;
  - `ews/prereg.py` (`NULL_RULES`);
  - the template's `nulls.rule`;
  - the tests `test_msvol`, `test_lookahead` (finality bound and a new edge case), `test_prereg_market` and `test_reports`.
- Builder checks on 87decc5:
  - ruff check and ruff format are clean;
  - pyright reports 0 errors;
  - `pytest python/tipping_point/tests/market python/tipping_point/tests/ews/test_prereg_market.py harness/tests`
    gives 480 passed and 26 skipped. The skips are cascade tests that need real data or `cascade.SWITCH`,
    and figure tests that need matplotlib.
- Cross-check on the laptop against the S5 pilot labels (2022-07-01..2023-12-31, local data only), counting
  72-bar windows with the real code. The counts equal the exploratory script's exactly:

  | D_min | BTC | ETH |
  |---|---|---|
  | 12 | 60 | 76 |
  | 36 | 217 | 200 |
- The full gate (`harness/gate.py`) has not run on this head yet. The repo has no GitHub CI.

## Open (cloud session)

1. Run the suite once on the head. The command is `uv run --no-sync python harness/gate.py` after
   `uv sync --extra cpu`; unset NODE_ENV first. If the Rust toolchain is missing or too slow, run
   at least the Python legs: `--leg ruff-check`, `--leg ruff-format`, `--leg pyright` and `--leg pytest`.
   The change is Python only, so record the cargo legs as `not run <reason>`. Write the summary lines here.
2. Run the review gate:
   - `reviewer` (Opus, fresh context) rounds until it approves;
   - then `sol-loop` on the final diff `54b189c..HEAD`.

   Success condition for reviewers:
   - the rule above holds exactly, including the boundary: a bar exactly Q from an onset is out, and Q+1 away is in;
   - no option keeps the old rule;
   - the lookahead finality bound `end <= T + 1 - d_min - Q` is argued from event confirmation and tested;
   - the template and `NULL_RULES` agree, and the template tests still tie the constants to the code;
   - no frozen file under `paper/prereg/` other than the template changes.
3. Fix must-fix findings with the smallest change. File other findings as follow-ups in the summary,
   because the cloud cannot use `gh issue`.
4. Stop at Waiting. Write `.agent/summary.md` with what changed, the review results and the open
   decisions, then push.

## Rules for this repo (not in a repo CLAUDE.md)

- **No Claude attribution on commits.** The owner is the sole author: no Co-Authored-By lines, and no
  "Generated with" lines.
- Never run the scorer on confirmatory Ising seeds 6–10. Never generate or read market labels from
  2024-01-01 on: `market-labels-v1` is not frozen.
- Do not touch the git tags `prereg/*`, the frozen files `paper/prereg/ising-*.toml` or main.

## Needs the laptop (skip in the cloud)

- Market pilot data and the archive (local data dir). The cross-check above is already done.
- The study note `docs/study/29-*.md` and the vault project note, both outside git.
- Opening or updating the PR, issues and merging.

## Cloud session, 2026-11-05

### Suite on head b1c330a

`uv sync --extra cpu` failed: the sandbox's network policy blocks download-r2.pytorch.org (torch wheel,
"tunnel error"). Plain `uv sync` succeeded without torch. The gate, `harness/gate.py`, gave these summary lines:

- cargo-fmt     GREEN  no diff reported
- cargo-clippy  RED    failed to run custom build command for `libduckdb-sys` (sandbox has no DuckDB library). Environment, not the change.
- cargo-test    RED    same build failure (exit 101). Environment.
- ruff-check    GREEN  All checks passed!
- ruff-format   GREEN  142 files already formatted
- pyright       RED    40 errors: 39 are `Import "torch" could not be resolved`, and 1 is `subprocess.BELOW_NORMAL_PRIORITY_CLASS`
  (Windows-only attribute, harness/tests/test_gate.py:843). On the changed files, pyright reports 0 errors.
- pytest        RED    torch is missing, so collection errors. With `--continue-on-collection-errors`:
  7 failed, 655 passed, 33 skipped, 13 errors. Every failure and error is `ModuleNotFoundError: No module named 'torch'`
  (test_explore, the ising, jepa and latent tests). The branch does not touch any of them.
- self-test     RED    No module named 'torch'.

Builder's focused set, `pytest python/tipping_point/tests/market tests/ews/test_prereg_market.py harness/tests`:
442 passed, 33 skipped, 7 failed and 4 errors, all of them torch in harness/tests/test_explore.py. All market and prereg tests pass.

A full green gate still needs the laptop, where torch and DuckDB are installed.

### Review gate

- Round 1, `reviewer` (Opus, fresh context): APPROVE, no blockers. It left two nits:
  1. test_msvol.py:186-204: the planted-series test checks only C-length windows. Assert `member(t)` for every bar of each stretch.
  2. msvol.py:404: `confirm = onset + d_min` is one bar later than the o+d_min-1 the lookahead docstring uses. This is harmless.
- Round 1, `sol` (gpt-6.1-sol, high, via Codex, `-s workspace-write` after the auto-mode classifier denied
  `danger-full-access`): APPROVE, no new findings. Codex session 01a1accc-d6d3-7660-9e07-780040754a39.
- The loop stopped early because both seats approved and returned only nits. The nits were not applied and are follow-ups.

State: Waiting. Next step: the laptop runs the full gate with torch and DuckDB, then the owner merges.
