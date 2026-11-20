# Pre-registration and scoring

Each test is fixed in a TOML file in [`paper/prereg/`](../paper/prereg/) and frozen at a git tag
before its confirmatory data are read. This page covers the classic indicators and how a file is
scored, the latent files and their runner, the market tests' decision rules, the exploratory
pipeline that reads only data no test scores, and the gate that checks the code.

| file | tag | status |
|---|---|---|
| [ising-ews-v1.toml](../paper/prereg/ising-ews-v1.toml) | `prereg/ising-ews-v1` | scored; see [results.md](results.md#ising-ews-v1-the-classic-baseline) |
| [ising-latent-v1.toml](../paper/prereg/ising-latent-v1.toml) | `prereg/ising-latent-v1` | scored; see [results.md](results.md#ising-latent-v1-the-latent-experiment) |
| [market-labels-v1.toml](../paper/prereg/market-labels-v1.toml) | `prereg/market-labels-v1` | frozen; labels only, no hypothesis |
| [market-ews-v1.toml](../paper/prereg/market-ews-v1.toml) | `prereg/market-ews-v1` | frozen; not yet scored |
| [market-latent-v1.toml](../paper/prereg/market-latent-v1.toml) | `prereg/market-latent-v1` | frozen; not yet scored |

The `TEMPLATE*.toml` files beside them are the starting points for new files.

## Early-warning signals

`python/tipping_point/src/tipping_point/ews/` holds the classic indicators (rolling variance,
autocorrelation and skewness after Gaussian detrending), their Kendall-tau trend, surrogate tests,
the scoring protocol (AUC, alarms, lead times, hierarchical bootstrap) and finite-size-scaling
verdicts, on numpy alone. The definitions are ewstools 2.1.3's, pinned by a golden fixture that
`python/tipping_point/tests/fixtures/make_ews_golden.py` regenerates in an isolated environment.
Experiments are pre-registered in `paper/prereg/<id>-v<k>.toml` (start from `TEMPLATE.toml`) and
frozen as git tags; the scorer refuses an unfrozen or edited file, reads tau_c(L) from a pilot
run's `pilot.json`, never from the pre-registration, and scores only runs whose manifest names the
commit the freeze tag points at, from a clean tree: ramps and null runs made against the pilot
record the file pins, and staircases whose dwell was read from the tau.json it pins at that L
(`data.staircase.tau_sha256`), so a staircase given `--tau` as a number is refused.

```sh
uv run python -m tipping_point.ews prereg freeze ising-ews           # tags prereg/ising-ews-v<k>
uv run python -m tipping_point.ews score --prereg paper/prereg/ising-ews-v1.toml
uv run python -m tipping_point.ews calibrate                         # 2,000 nulls x 999 surrogates
```

Results land in `<data dir>/ews/<run_id>/`, and every evaluated cell is appended to
`<data dir>/ews/trials.jsonl`.

## Latent signals

A pre-registration with a `[latents]` section scores a JEPA's readouts instead of |m|. It names
each training it scores, by its run id and the sha256 of `final.pt` and of its untrained twin, and
the scorer takes both through every table, once per checkpoint.

- **The join.** `ising.series(run, "matched", latents=<table>)` joins
  `<run>/latents/<training>-step<step>/`, which `jepa extract` writes, onto the run's saved frames
  with a plain Parquet reader, since `jepa.extract` imports the adapter. The six key columns must
  equal those of the frames the table's own selection rule keeps, so a subsampled staircase reads
  as the frames it holds. A key mismatch, a table without `complete.json`, or the dense cadence, for
  which no latents exist, is refused with an error naming the run and the table.
- **Two more indicators.** `level` (rolling mean) and `slope` (rolling OLS slope per sample) read
  the raw series: the rolling mean of residuals is zero by construction, so `Pipeline` skips the
  detrend for them. A level's rows carry the phase surrogate alone, and its bandwidth axis
  collapses to one cell per window. The extremum of `slope` is the dynamic statistic of a channel
  with no interior peak: finite-size scaling puts the steepest change at a fixed eps_L, 1/nu = 1,
  and a feature anchored to a fixed T gives 1/nu = 0. The harness plants both, and the second is the
  dynamic test's negative control. Positions are read half the headline window late (0.2 eps_L),
  and a read position within noise of 0, a true steepest change near +0.2 eps_L rather than at Tc,
  is inconclusive. A channel that only keeps rising towards an edge of the search band has its
  maximum a few noisy samples inside it, at the same eps_L at every L, which would read as 1/nu = 1
  with no peak there: so `fss.dynamic.edge_margin = "window"`, which every latent file holds,
  counts an extremum within one headline window of either edge as not located, and over half of an
  L's replicas not located is inconclusive. ising-ews-v1 refuses the key and reads the band's first
  and last samples alone. The `peaks` table of a file with the key says which way a peak was lost:
  `edge_fraction` for an edge, `undefined` for a replica with no value in the band, as a bug that
  empties a channel leaves. Both count as not located. v1's table keeps one `edge_fraction`, which
  holds both.
- **Rows.** Every row of every table has a `checkpoint` column, the table's folder name, and so does
  each line of `trials.jsonl`; a classic row's is `""`. The file pins each checkpoint's sha256 and
  role, and the result's manifest gives the sha256 of each table's `complete.json` per run.
- **What a file holds.** `indicators.pairs` lists the (observable, indicator) pairs judged by AUC,
  `indicators.direction_by_observable` overrides a direction for one observable, and
  `fss.dynamic.statistics` lists the peak statistics the dynamic test reads. A statistic of role
  `descriptive` is computed with the same intervals and edge rule and has no verdict. A static row
  with a verdict, or a dynamic pipeline check, may say `judged = "twin"`: it is then judged on the
  step-0 twins alone, and on a trained checkpoint its row is computed all the same and reads
  `reported`. A primary checkpoint and its twin take the whole grid; an ablation's take the
  headline cell, with `scoring.surrogates.n_ablation` surrogates. `prereg.cells` counts that.
  Static quantities are `l2_var_<column>`, `mean_<column>` and `one_minus_ac1_<column>`, read from
  the matched cadence.
- **The freeze.** A latent file reads the runs of the version its `[baseline]` names, so those runs
  must come from that tag's commit, and each table from the file's own tag and the checkpoint it
  pins: the freeze comes before any extraction on a scored seed. `paper/prereg/ising-latent-v1.toml`
  was rewritten on 2 November after exploratory reads of the m1 and m2 checkpoints, with its eleven
  checkpoints pinned and `scoring.margins.tolerance` set. On 4 November its last placeholder,
  `baseline.result`, took `20261104T095100Z-ising-ews-v1`, the run id of ising-ews-v1's result on
  these runs, and the file was frozen as the tag `prereg/ising-latent-v1` on commit b19c2df (sha256
  `33cf5917a224ca6583ceedf1469cf3e921ce9decd968d7638ba0e5b69343dc23`).
  `python/tipping_point/tests/fixtures/ising_latent_draft.toml` is the schema's fixture, with every
  value the file was to set left open.
- **What a file may say.** Frozen text must not claim what the scorer ignores, so a key outside the
  latent schema is refused, wherever it stands: v1's `scoring.alarms.indicators` and
  `indicators.thin`, `fss.dynamic.band`, anything else. So is a value the scorer would not act on: a
  name, observable or direction no pair of `indicators.pairs` reads, a column nothing scores, a
  tolerance on a descriptive row, `sign_test = false`. `latents.extraction`, `probes.fit`,
  `probes.test` and `probes.selectivity` are held to their sentence, word for word where they state
  a rule: the scorer takes the frame limit and the probe seed from `probes.fit`, and holds each
  table's manifest to the rule `latents.extraction` gives its checkpoint's role
  (`prereg.SELECTIONS`), refusing a table extracted under another.

## The latent experiment

A latent file is scored whole by one runner, from the frozen file to the result and its figures:

```sh
uv run python harness/ising_experiment.py --prereg paper/prereg/ising-latent-v1.toml [--device cuda]
```

`ews score` refuses such a file without `--cells-only`, since it would spend the whole trial budget
on a result with no probes or margins, and the budget is spent once. The runner refuses a file that
is not frozen or still holds a placeholder, and a scoring whose cells would pass `trials.budget`
(`prereg.charge`: cell 3,133 of `ising-latent-v1` is never evaluated), before it reads anything.

- **The data dir.** The runner reads each run's latents tables at `<run>/latents/<table>/`, the
  checkpoints at `<data dir>/jepa/<training>/`, the baseline result and the trial log at
  `<data dir>/ews/`. `harness/latent_data_dir.py` assembles such a folder from the Kaggle outputs
  that hold the extracted runs and the trainings, and a data dir for the rest (the probes' fit
  runs, the pilot record), hard linking every file (copying where a link is refused, across
  volumes). It then checks that every run the file scores is there, complete and with each table
  it pins, with each L's fit staircase, and every file against its `complete.json`, each table's
  keys against its run's frames included. Its `ews` is a link to the data dir whose trial log
  counts the budget, since `--data-dir` moves the trial log with everything else: a data dir with
  an `ews/` of its own would start an empty log, and the budget could be spent again there. Remove
  that link before deleting the folder: a delete that follows it, as PowerShell 5.1's
  `Remove-Item -Recurse` does, deletes the trial log and the baseline result.
- **Checking first.** `--check` runs every refusal that comes before the result folder is made, the
  checkpoints' sha256 and a `--resume`'s resume key included, prints what the run would read and
  log and whether the checkout is clean, and stops, writing nothing and charging nothing:

  ```sh
  uv run python harness/latent_data_dir.py --prereg paper/prereg/ising-latent-v1.toml \
      --into <new data dir> --from <kaggle out> [--from <kaggle out> ...] --from <data dir>
  uv run python harness/ising_experiment.py --prereg paper/prereg/ising-latent-v1.toml \
      --data-dir <new data dir> --check
  ```

- **Tables.** `auc`, `surrogates`, `alarms`, `peaks`, `fss` as `ews score` writes them; `probes`, per
  checkpoint, L, target and column, fitted on the validation seed and scored on the scored seeds;
  `margins`; `curves`, each column's mean and L^2 Var per plateau, with the classic |m| on every
  saved frame as checkpoint `""`; and `baseline`, the rows read from the result `baseline.result`
  names, which must be that version's, on these runs. Every row has its checkpoint. All of them are
  also in `csv/`, from the Parquet.
- **Margins.** Paired by model seed, with a percentile bootstrap over the seeds nested in one data
  resample per replicate, and the sign test. Three rows are judged, over the primary checkpoints: the
  energy probe's R^2 and its selectivity, each checkpoint's value being the mean over L
  (`scoring.margins.sizes`), and the surprise FSS verdicts, by the hypothesis's own rule: a surprise
  statistic passing for at least 4 of 5 trained checkpoints and for none of their twins. The third
  is absent from a file with no surprise statistic of role `test`. With
  `scoring.margins.sizes = "every"` the judged value and its CI stay the mean over L, and the sign
  test also asks every seed's margin at every L to carry the predicted sign. Each L's value, AUC
  (parity, with |m|'s beside it as `reference`), the chi alignment and all of an ablation's rows
  report a margin and its CI with no verdict.
- **Figures.** `figures/` holds `scaling.png`, `peaks.png`, `curves.png`, `collapse.png` and
  `probes.png`, primary checkpoints only, drawn by `harness/ising_figures.py` from the CSV tables
  alone, in a throwaway environment (matplotlib is no dependency of the project). The tables are
  written and logged first: if the figures fail the run exits 2 and prints the command to draw them.
  A run that dies with its result complete but its CSV not has spent its budget and left that
  command nothing to read: `harness/ising_experiment.py --figures-only <result folder>` writes
  `csv/` again, over what is there, then draws the figures, charging nothing and logging no trial.
  It refuses a folder with no `complete.json`, whose cells may be unlogged. A latent variance is
  divided by a per-checkpoint number before anything is drawn together, because a PCA basis is per
  checkpoint and raw latent magnitudes do not compare.
- **Planted runs.** `--planted` scores a file with no freeze behind it and refuses a data dir holding
  any run the planted generator did not write, so it can never read the confirmatory data early.
  `harness/ews_planted.py` plants the rest (`latent_doc`, `write_experiment_tables`,
  `planted_embedding`, `write_baseline`), and two runs on a planted dataset give byte-identical
  tables and figures.
- **Surrogates in a latent file** stop drawing once a row's decision at alpha is settled
  (`scoring.sequential_test`, Besag and Clifford 1991): the surrogate table reads only p <= alpha, and
  on the same surrogates every decision is the one a full N_s gives. A null row draws about a quarter
  of N_s. ising-ews-v1 keeps its whole test and its draws. `scoring.surrogates.cells` says where the
  tests run: `"all"` at every cell, `"headline"` at the headline cell for every h, as ising-ews-v1
  has it, and `"headline-h0"` at the headline cell for h = 0 only, as ising-latent-v1 has it.
- **Run time.** ising-latent-v1's scoring took about 39 hours, from 16 to 18 November with other
  work running, on a 4-core laptop CPU (i7-1185G7): the probes' embeddings 20.3 h, the ten primary
  passes 15.2 h and the twelve ablation passes, at the headline cell with 199 surrogates, 3.1 h.
  The embeddings and the first two passes ran on CPython 3.13, where the scoring threads used
  about one core between them; the run was then resumed on the free-threaded CPython 3.14t build
  with 6 threads. The two primary passes on 3.13 took 2.6 and 2.8 hours; the eight on 3.14t took
  from 48 minutes to 2.5 hours, the first of them counting the resume's loading. Scoring runs on up
  to 8 threads by default; on a host with more cores, `TP_THREADS=<n>` sets the count to n, and the
  results do not depend on it.
- **Crashes and the trial log.** A run's folder is made before its probes, so a run interrupted
  anywhere leaves a folder with no `complete.json`, which `--resume` (below) carries on. Its trials
  are logged in one write after its tables and before `complete.json`, so a crash in that write
  leaves a folder with no `complete.json`, its trials logged or not, or a complete result with all
  of them. `prereg.charge`, for any file, `ews score` included, refuses to score past a folder
  stamped with the file's id and version and with no `complete.json`, since its tables can be read
  whether or not their cells were logged: carry it on with `--resume`, or delete it or amend the
  file. Deleting it gives back no trial the log holds, so a crash after the log spends the budget:
  the tables are there, and scoring again means amending the file. A crash inside the log's one
  write leaves a line that is no whole record, which `prereg.trials` refuses by file and line:
  count that run as spent and amend the file. Every count reads the whole log, so first, keeping a
  copy, replace the line with one whole record holding the `params.prereg` stamp from that run's
  manifest. Cutting it instead could leave the version no line at all, and with its folder deleted
  it could be scored again. With the record in place the budget refuses it, folder or no folder,
  wherever `trials.budget` equals the file's cells, as both Ising files' do. If no folder is left,
  the record gives the id and version of the file that was being scored, or there is one for each
  file it could have been. A blank line in the log is skipped.
- **Resuming.** A run stopped partway, by a crash, a reboot or a sleep it never woke from, is
  carried on by the same command with `--resume`, which `ews score` takes too. Each probe (one L
  and one checkpoint) and each checkpoint's pass of the scorer is saved in the result folder's
  `parts/` as it ends, the pass with the random generator's state, and a resumed run reads them
  back, so its tables are the bytes an uninterrupted run writes. It refuses, saying what differs,
  unless the torn folder began under the same file and stamp, seed, inputs, pilot record, numpy and
  pyarrow versions and (for the runner) torch and device, from a clean checkout of the same commit:
  commit everything, untracked files included, before a long run starts. Resume only once the run
  that began the folder has stopped, since two at once would both write it. Its trials are logged
  once, at the end, even when the crash came between the log and `complete.json`, and the manifest
  lists every start under `resumptions`.

## Scoring the market tests

Both market files are frozen and not yet scored. They score BTCUSDT and ETHUSDT on 2024-01-01 to
2026-09-30, on the regime-shift events and control segments cut from market-labels-v1's labels
(see [market.md](market.md#the-market-warning-test)), and neither is scored on any of those days
until both are frozen. Each counts one trial per AUC cell it evaluates, against its budget.

**market-ews-v1** ([file](../paper/prereg/market-ews-v1.toml)) asks whether the trend (Kendall's
tau) of the rolling variance and of the rolling lag-1 autocorrelation of 1-minute returns, over the
four hours ending h bars before an onset, is higher than over matched calm windows. Horizons are
h = 5, 6, 12 and 24 bars; h = 5 and 6 decide, 12 and 24 are reported. A cell passes when its AUC's
95 % CI lies above 0.5, above the GARCH floor's AUC at that horizon, and above the 97.5th
percentile of the event-time shift (100 draws, each onset moved at least 288 bars). A hypothesis
holds when all six cells (pooled, BTCUSDT and ETHUSDT, at h = 5 and 6) pass. The bootstrap takes
2,000 draws, the seed is 30, and the trial budget is 32. The same-input tier is reported beside it
and counts no trial.

**market-latent-v1** ([file](../paper/prereg/market-latent-v1.toml)) scores four latent pairs
(the variance and lag-1 autocorrelation of pc_1, the variance and level of the surprise) from the
five primary checkpoints per symbol and their untrained twins, each model reading both symbols.
The pairs are two-sided: the statistic is |AUC - 0.5|. At h = 5 and 6 each pair gets three
verdicts, the later two read only for a pair that warns: whether it warns at all (against the
event-time shift), whether training mattered (trained minus twin, with a tolerance of 0.02), and
whether it beats the matching classic indicator. The trial budget is 1,312.

Both files list what was seen before the freeze in `[disclosure]`, including the 2023 exploratory
reads and the floor, and both record a known limit: the bootstrap treats adjacent control segments
and coinciding BTC and ETH events as independent, so the CIs can be too narrow.

## The exploratory pipeline

The exploratory pipeline, `harness/explore/`, scores the seeds 1-5 Ising datasets, which neither
Ising pre-registration scores: on |m|, or on a latents table with `--latents <table>`, one size at
a time, through this adapter. It refuses any seed outside 1-5 and cannot log a trial, so nothing it
prints counts as one. `join_pilot` links a Kaggle output's tables onto the local runs in a new data
dir, and `latent_pass` runs every step for a training, its trained checkpoint beside its twin:

```sh
uv run --no-sync python harness/explore join_pilot --output <kaggle out> --into <folder>
uv run --no-sync python harness/explore latent_pass --training <run id> --step <step> \
    --data-dir <folder> --out <outputs>
```

Its scripts run at below-normal priority with at most 4 worker threads (`common.THREADS`), so they
share the machine with other jobs.

`market_pass` is the market JEPA's exploratory 2023-Q4 pass. It proposes a direction for each
latent pair and which latent columns to judge, and was read before market-latent-v1 was written;
nothing in it is a trial. It reads each model's trained checkpoint and its untrained twin (step 0),
each on its own symbol's Q4 and on the other's: four cells per table (BTC->BTC, BTC->ETH, ETH->ETH,
ETH->BTC).

- **Segments** (the floor's rule in `market.nulls`): an onset is the start of a 5-minute bar; at
  horizon h bars the event segment is the `Pipeline.span` = 299 minutes of series ending just
  before minute 5 (onset - h), the 240 indicator values of a 60-minute window. The full 240 values
  are used at every horizon, so at h = 24 a segment starts before the 72-bar clean window. Controls
  are the non-overlapping 299-minute spans cut in time order from each matched-calm null stretch
  (`nulls.null_rows`), clipped to the quarter. A segment touching a minute that any table read on
  that symbol marks invalid is excluded for every indicator and table alike, and counted; so is an
  event too early in the quarter for its segment. Every comparison is paired on the same segments.
- **Statistics** per (table, read symbol, column, indicator, h): the trend score s, Kendall's tau of
  the rolling indicator over the segment (`ews.trend.Pipeline(indicator, 60, 240)`, no
  detrending); median s on events and on controls; the AUC of events against controls
  (`ews.scoring.auc`) with a percentile 95% CI from `--draws` bootstrap draws resampling events and
  controls independently; and the twin margin, trained AUC minus twin AUC on the same draws. The
  four pairs are ising-latent-v1's: level and variance of surprise, variance and ac of pc_1. The
  screen adds every column: level and variance of surprise and surprise5, variance and ac of each
  pc.
- **Proposals**, by rules fixed before the numbers were read. A pair's direction is the sign of the
  median event s on the trained tables ("greater" when > 0), per cell and h, reported as one word
  when every same-asset cell and h agrees and as "disagree" otherwise, with the cross-asset cells
  beside it; when the signs disagree the direction is left open. A column is proposed for judging
  when, in both same-asset cells, some indicator of it at some h has a trained AUC whose CI
  excludes 0.5 and lies further from 0.5 than the twin's AUC there. Columns are also ranked by the
  mean over same-asset cells and h of |AUC - 0.5|, trained minus twin.
- **Memorisation:** mean surprise by month on the model's own training-span days and on the
  quarter, trained beside twin.

```sh
uv run --no-sync python harness/explore market_pass --latents <market_latents' out> \
    --labels <labels run> [--out <folder>] [--draws 1000]
```

It writes `<out>/market-pass.json` (every row), `market-proposal.json` and `market-report.md`.

Nothing here changes what ising-ews-v1 scores. `harness/ews_v1_golden.py` wrote every row and value
of a v1 scoring on planted data from aa29cbc, the freeze branch's merge, and
`python/tipping_point/tests/ews/test_v1_unchanged.py` holds later commits to them, RNG stream
included. The script runs `ising.evaluate` (fss mode) on the planted v1 dataset under the committed
`paper/prereg/ising-ews-v1.toml`, its settings changed only as far as the planted data need (seeds,
replicas, the tau_c record, the tau.json pins) and to run in seconds: 40 bootstrap draws, one
surrogate, alarm scores every 200 tau_c, one window, two bandwidths and two horizons of the grid.
Both observables, cadences and surrogate tests stay, so every loop of the scorer turns, and every
draw of the one RNG feeds the CIs and surrogate rates of the rows after it, so a scorer that draws
once more, once less or in another order moves them. The golden was written before any latent code
existed. To check it again, run `uv run python harness/ews_v1_golden.py [--out FILE]` on aa29cbc's
tree, which needs nothing newer, never on a later one: a golden rewritten from the code it checks
would prove nothing.

## The gate

```sh
uv sync --extra cpu                      # or --extra cu126 on a CUDA machine (the PyTorch build)
uv run python harness/gate.py            # all legs
uv run python harness/gate.py --leg pytest
```

`harness/gate.py` runs every Rust and Python check and needs `cargo` on `PATH`. Its legs are cargo
fmt, clippy and test; ruff check and format; pyright; pytest; and the Kaggle runner's
`--self-test`. A leg is green only when its exit code is 0 and its own summary shows the checks
ran. The gate exits nonzero if any leg is red, and its last line counts any check a leg skipped
(`gate GREEN (1 skipped)`). The full record, with each leg's output tail and every failing test's
output, lands in `harness/out/gate-<timestamp>.json`. A leg still running at its time limit is
killed and judged red. `cargo test` runs with `--no-fail-fast`. Before and after the legs, the gate
kills stray debug processes of the engine's binaries left by a killed test run
(`uv run python harness/strays.py --dry-run` lists them). Cargo builds on half the logical cores
unless `CARGO_BUILD_JOBS` is set, and on Windows the legs run at BelowNormal priority;
`TP_GATE_PRIORITY=normal` runs them at Normal, which keeps the timing test `api_load` from being
starved by other jobs. The cargo legs cover only the root workspace's members:
`harness/tests/test_gate.py` fails if `engine/` or any crate under `harness/` is outside it.
