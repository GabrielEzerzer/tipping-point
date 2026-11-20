# Ising

The 2D Ising model is the test bed with known answers: its critical temperature Tc and its
exponents are exact. This page covers the GPU simulator that makes the data, how a run is
stamped and seeded, the pilot record that sets the time unit, the run protocol, and the JEPA
trained on the lattices.

## The simulator

`python/tipping_point/src/tipping_point/ising/` simulates the 2D Ising model with checkerboard
Metropolis in PyTorch: int8 spins, a batch of independent replicas, one seeded generator per run.
Runs follow the shared layout `<data dir>/ising/<run_id>/` (`manifest.json`, Parquet chunks under
`stats/`, `configs/`, `balance/`, and `complete.json` after a clean end). A run id is the UTC start
time plus a tag in the engine's charset (`tipping_point.engine.run_id`: ASCII letters, digits, `-`
and `_`), so a number's `.` is written `p`, as in `20261023T010000Z-L16-ramp-x0p02-s1`; the
engine's API opens no other name. The data dir is `--data-dir`, else the engine's (see
[engine.md](engine.md#market-data-on-disk)); `--device` defaults to CUDA when available.

```sh
uv run python -m tipping_point.ising validate                        # Gates A-C on a small L = 4 run
uv run python -m tipping_point.ising pilot --L 16 --sweeps 50000      # tau_int per T -> tau.json
uv run python -m tipping_point.ising tauc <pilot run> ...             # tau_c(L) at Tc -> pilot.json
uv run python -m tipping_point.ising staircase --L 16 --tau <pilot run>/tau.json
uv run python -m tipping_point.ising ramp --L 16 --x 0.02 --pilot <record>/pilot.json --eps-start 12 --eps-end -8  # x: 0.02 slow, 2 fast
uv run python -m tipping_point.ising null --L 16 --epsilon-L 4 --pilot <record>/pilot.json --seed 6   # epsilon_L 2, 4, 6; v1's seeds 6-10
uv run python -m tipping_point.ising resume <run dir>                 # after a crash or a killed session
```

## Which commit a run records

Every manifest names the code that wrote it in `producer.git_sha` and `producer.dirty`: Ising
runs, JEPA trainings and EWS results all take them from `io.code_version()`. Run from the
checkout, where `uv sync` installs the package editable, they are git's HEAD and whether the tree
has uncommitted changes, untracked files included. A wheel install sits in no checkout, so runs
made from a wheel are stamped from the build: the build hook (`python/tipping_point/hatch_build.py`)
writes `tipping_point/_build_info.py` (`commit`, `dirty`, `kernel_sha256` of `ising/kernel.py`,
`built_at` in UTC) into every sdist and wheel built from a git checkout, and `code_version` reads
it. `tauc` takes a stamped commit exactly as it takes one from git. An installed copy never reads
a repository that only encloses it, such as a venv inside another project. A build made where no
repository tracks the source records no commit, and `tauc` refuses its runs. The hook warns when it
makes one, but `pip` and `uv pip install` hide build output, so `pilot` refuses to start from such
an install rather than leave the refusal to `tauc`, hours later. To build a wheel for a GPU
machine, check out the commit, keep the tree clean, and run:

```sh
uv build --package tipping-point --wheel     # -> dist/tipping_point-0.1.0-py3-none-any.whl, stamped
uv run python harness/wheel_provenance.py --wheel dist/tipping_point-0.1.0-py3-none-any.whl
```

The second line checks a wheel end to end in throwaway venvs: its runs must record the stamped
commit, inside or outside another repository, and `tauc` must accept them. `tauc` also needs
every size from one commit and one kernel, so a record over new sizes re-runs the old sizes from
the same build.

## Which stream a run draws

A run's generator is seeded from its identity, not its seed alone: `run.stream_key` hashes the
seed, L, replicas, start and schedule. The manifest holds the result in `params.rng.seed` and names
the rule in `params.rng.derivation`, so the key can be checked from the manifest alone. Runs at one
seed therefore draw different numbers, and one run made twice draws the same whatever its tag,
which only names the folder. `--checkpoint-every` and Gate C's tallies stay out of it: they change
what is written, not what is drawn.

Runs made before this rule (the seeds 1-5 exploratory datasets and the grid pilots) drew from the
seed alone, so runs of one shape at one seed shared their random numbers. That touches one
earlier check: the surrogate calibration's binomial band (3.1-6.9%) treated its 480 slices as
independent while the three null runs at each of the five seeds shared a stream, so that
pre-flight check is somewhat weaker than stated. The confirmatory runs draw one stream per run.

## The pilot record: tau_c(L)

The time unit of everything downstream (ramp rates, null-run lengths, the EWS scorer's windows) is
tau_c(L) = tau_int(|m|) at exactly Tc, the integrated autocorrelation time of |m|, read from a
pilot record and never from a constant. Run one pilot per L at Tc, then `tauc` fits
tau_c = A L^z and writes the record:

```sh
uv run python -m tipping_point.ising pilot --L 16 --T 2.269185314213022 --replicas 64 --therm 2000 --sweeps 50000
uv run python -m tipping_point.ising pilot --L 32 --T 2.269185314213022 --replicas 64 --therm 10000 --sweeps 100000
uv run python -m tipping_point.ising pilot --L 64 --T 2.269185314213022 --replicas 16 --therm 40000 --sweeps 300000
uv run python -m tipping_point.ising tauc <L16 run> <L32 run> <L64 run>    # -> <data dir>/ising/<id>-tauc-L16-32-64/pilot.json
```

`pilot.json` is `{"kernel_commit", "kernel_sha256", "per_L": {"16": {...}, ...}, "fit": {"A", "z",
"A_ci", "z_ci"}}`, each `per_L` entry `{"tau_c", "tau_c_binning", "tau_e", "n_over_tau", "replicas",
"sweeps", "seed"}` plus `"tau_c_ci"` (95%, optional for readers; field notes in `ising/pilot.py`).
A size missing from `per_L` reads tau_c = A L^z. `tauc` refuses a run that is incomplete, never
held exactly Tc, or holds fewer than 300 tau_c: subtracting each series' mean pulls the windowed
estimate about 10 / (N / tau) low, so a recorded tau_c is at most ~3% short (`test_autocorr` pins
the bias).

**A stale record is refused.** `kernel_sha256` is the sha256 of `ising/kernel.py` (the sweep and
its acceptance table) that the pilot runs ran; every run's manifest records its own as
`params.algorithm.kernel_sha256`. `ramp --x`, `null` and `ews score` refuse a record whose
`kernel_sha256` is not the running kernel's, since its tau_c would clock another kernel. The
commit cannot decide this: it moves with every merge, kernel or not. A record with no
`kernel_sha256` counts as stale. `--allow-stale-pilot` uses a stale record anyway, with a
warning. Either way the run's manifest names the record it read as `params.pilot`:
`{"run_id", "sha256", "kernel_commit", "kernel_sha256", "stale"}`, and EWS results repeat that stamp
for each input run beside their own record's `kernel_sha256` and `stale` in `params.pilot`. The
record below measured the running kernel, so it reads as current.

The record ising-ews-v1 pins (run `20261025T194846Z-tauc-L16-32-64-128`, commit 655949a, seed 1),
committed byte for byte as `python/tipping_point/tests/fixtures/ising_pilot.json`, sha256
`7c5a4a296f2d8d1d55e223e4c775cd98267c43e6ae89f5e6ec594ac5c0581698`:

| L | replicas x sweeps | tau_c = tau(\|m\|) [95% CI] | binning | tau(e) | N / tau_c |
|---|---|---|---|---|---|
| 16 | 64 x 50,000 | 53.3 [51.4, 55.2] | 51.1 | 26.3 | 939 |
| 32 | 64 x 100,000 | 221.2 [209.9, 232.9] | 216.7 | 94.2 | 452 |
| 64 | 64 x 400,000 | 1043.5 [994.3, 1102.5] | 988.9 | 346.2 | 383 |
| 128 | 32 x 2,000,000 | 4619.5 [4246.6, 5006.8] | 4264.2 | 1342.5 | 433 |

Fit: A = 0.132 [0.114, 0.152], z = 2.155 [2.113, 2.197], 95% CIs from a bootstrap over replicas.
The interval holds the 2.17 of Liu, Polkovnikov and Sandvik's Kibble-Zurek collapse for
Metropolis; the local slopes between consecutive sizes are 2.05, 2.24 and 2.15. v1 reads L = 16,
32 and 64 only.

## Protocol

- **Staircase**, the quasi-static arm: the 40-point grid from 3.5 down to 1.5, each plateau
  thermalised for max(500, 20 tau) sweeps and measured for max(2000, 100 tau), with tau from a
  `pilot` run's tau.json at that temperature. The manifest records that file's sha256 as
  `params.schedule.params.tau_sha256`, null when `--tau` is a number. The plateau nearest Tc costs
  1,043 + 5,211 sweeps at L = 16.
- **Ramps**, the dynamic arm: linear cooling at r = tau_c(L) L / (x Tc) sweeps per unit T, so
  x = tau_c(L) / (sweeps per unit of epsilon_L = L (T - Tc) / Tc) at every L. tau_c(L) is the
  pilot record's measurement at L; the fit A L^z stands in only for a size it did not measure,
  and the manifest's `params.schedule.params.rate_source` says which (`measured` or `fit`).
  The ramp runs from 3.5 to 1.5, or over an epsilon_L window: `--eps-start 12 --eps-end -8` sets
  T0 = Tc (1 + 12/L) and T1 = Tc (1 - 8/L), the window ising-ews-v1 scores at every L, and the
  manifest records it as `eps_L`. The slow ramp, x = 0.02, spends 50 tau_c per unit epsilon_L and
  crosses the critical window |epsilon_L| < 1 in 100 tau_c; the fast one, x = 2, crosses it in one
  tau_c, the Kibble-Zurek regime. Sweeps of cooling (equilibration and hold not counted) at the
  committed record's measured tau_c:

  | | L = 16 | L = 32 | L = 64 | L = 128 |
  |---|---|---|---|---|
  | x = 0.02: critical window | 5,326 | 22,120 | 104,351 | 461,954 |
  | x = 0.02: epsilon_L 12 to -8 | 53,258 | 221,199 | 1,043,507 | 4,619,544 |
  | x = 0.02: 3.5 to 1.5 | 37,552 | 311,934 | 2,943,102 | 26,057,880 |
  | x = 2: critical window | 53 | 221 | 1,044 | 4,620 |
  | x = 2: epsilon_L 12 to -8 | 533 | 2,212 | 10,435 | 46,195 |
  | x = 2: 3.5 to 1.5 | 376 | 3,120 | 29,432 | 260,578 |

  Bursts are saved every 0.005 of cooling, except inside the EWS analysis band epsilon_L in
  [10, -5], where they come 28 per unit epsilon_L at every L (spaced evenly in T, L = 128
  got 3.5 there); the manifest's `schedule.saves` records the gaps and the burst counts.

- **Null runs**, the EWS scorer's negatives: `null --epsilon-L {2,4,6}` for each L, at least 5
  seeds each, holds T = Tc (1 + epsilon_L / L) for 20 tau_c of thermalisation and 700 tau_c of
  measurement, two scored slices at x = 0.02 (310-340 tau_c each), with a stats row every
  tau_c / 50 sweeps and bursts at the slow ramp's in-band cadence. The manifest is a `fixed`
  schedule whose params add `epsilon_L` and `tau_c`, and complete.json judges the run
  thermalised like a staircase plateau. With the committed record one run is 38,346 sweeps at
  L = 16, 159,264 at 32, 751,327 at 64 and 3,326,072 at 128. A size the record did not measure
  is refused, since the manifest could not tell a fitted tau_c from a measured one.

## JEPA on Ising

`python/tipping_point/src/tipping_point/jepa/` holds the Ising JEPA: one per-frame encoder shared
by every L (2x2 patchify, three ConvNeXt blocks with circular depthwise 3x3 convolutions; 0.40M
parameters, a receptive field of 14 sites, below the smallest L), a convolutional predictor
(0.16M parameters, ~50 sites of range) and an EMA (exponential moving average) target encoder. It
hides blocks of frame c (4-8 sites a side, 20-25% of the cells, a one-cell buffer cut from the
context) and predicts their LayerNorm'd target features in frame c + delta, delta 0-7 inside a
burst; VICReg variance and covariance terms on the visible tokens guard against collapse, and a
detector with absolute floors on feature spread watches for it. Bursts are flipped to positive
magnetisation, D4 and rolls are augmentations, and training reads staircase plateaus only, of
dedicated simulator seeds: the seeds 1-10 (ising-ews-v1's 6-10 and its exploratory pilot's 1-5),
ramps and null runs are refused.

Its Ising modules: `lattice` (the circular ConvNeXt encoder, convolutional predictor, torus masks,
canonicalisation, torus symmetries, bit unpacking) and `data` (simulator runs to packed bursts in
RAM, run selection by seed, the held-out band). Its generic modules: `objective` (MSE on
LayerNorm'd targets at hidden cells, VICReg terms, schedules), `monitor` (the collapse detector with
absolute floors, effective rank, normalised error) and `train` (the loop: seeding, AMP per device,
checkpoints, resume, manifest, GPU hours). `extract` fits the PCA basis and writes per-frame
surprise and pc_k as a dataset run's `latents` table, and `probes` fits ridge probes from frozen
frame embeddings, trained beside untrained. `python -m tipping_point.jepa --help` lists the
commands.

```sh
uv run python -m tipping_point.jepa train --L 16 32 64 --train-seeds 101 102 --val-seed 103 \
    --seed 1 --pilot <pilot run>/pilot.json [--held-out] [--prereg paper/prereg/<id>.toml]
uv run python -m tipping_point.jepa resume <training>                # after a killed session
uv run python -m tipping_point.jepa extract <training>/final.pt <simulator run> [--select tc-plus]
uv run python -m tipping_point.jepa probe <training> --fit <val run> --test <scored run>
uv run python -m tipping_point.jepa verify <training> <run>/latents/<set> ...
uv run python -m tipping_point.jepa budget --quota 30                # GPU hours per ISO week
```

- **A training** lives in `<data dir>/jepa/<run_id>/`: the manifest (every input run with its
  `complete.json` sha256, the pilot record's sha256, the pre-registration's id, the seeds, the git
  commit), `train/` and `monitor/` Parquet, `ckpt/step_0000000.pt` (the untrained twin, saved
  before any update), `final.pt`, and `complete.json` with every file's sha256 and the collapse
  verdict. Both models carry a PCA basis fitted on training frames. Every session appends its
  hours to `<data dir>/jepa/gpu_hours.jsonl`.
- **Reproducibility:** one master seed gives each random stream its own generator;
  `use_deterministic_algorithms(True)` is a tripwire and `CUBLAS_WORKSPACE_CONFIG` is set before
  CUDA starts. Two CPU runs of a seed give bitwise-identical losses, and a resumed run continues
  bit for bit. A T4 trains in fp16 autocast, anything else in float32; never bf16, never
  `torch.compile`.
- **The latents table** is written beside the frames, in
  `<simulator run>/latents/<training>-step<step>/`: one row per `configs` row, keyed and ordered
  like `configs` (step, replica, temperature, segment, burst, frame), with `surprise` (the delta = 0
  prediction error over two fixed evaluation masks) and `pc_1`..`pc_8`, float32. Its manifest names
  the checkpoint and the run by sha256; `tipping_point.jepa.extract.read` returns it. `--select`
  subsamples staircases: `tc-plus` keeps the T = 2.269231 plateau whole and one burst in 8 (by
  burst index) at every other plateau, and refuses a staircase whose plateaus save a number of
  bursts 8 does not divide; `tc-only` keeps that plateau alone, for ablation checkpoints;
  ramps and null runs are always read whole. The manifest records the rule, and `verify` checks
  that the table holds exactly the rows it keeps.
- **Probes** read |m|, the energy and L^2 (|m| - <|m|>_T)^2 from frozen embeddings by ridge
  regression, each beside two control tasks: an i.i.d. Gaussian target (R^2 near 0) and the target
  shuffled among the frames of each plateau. Selectivity, R^2 minus the shuffle's, is what the
  embedding holds of a frame beyond its temperature.
- **The untrained twin is the null:** a local encoder's pooled readouts inherit Ising scaling with
  random weights, so learning is claimed only as a margin over each run's step-0 twin, which goes
  through every test. `tipping_point.ews.margins.paired` judges one: the CI of the mean margin
  over model seeds (nested with paired data-bootstrap draws) must exclude 0 in the predicted
  direction, and every seed's margin must carry the predicted sign.
- **Kaggle:** `notebooks/kaggle_ising_runs.py` runs a JSON plan from `notebooks/plans/` as a
  kernel: simulator runs, `jepa train` and `jepa extract`, one queue per T4. It installs the
  commit-stamped wheel from a wheel dataset (no clone and no token), stages the dataset runs a
  plan reads, checks every run with `jepa verify`, and resumes from the previous version's output.
  `--write-kernel <plan> <dir>` writes the folder `kaggle kernels push` takes; `--dry-run-jepa`
  rehearses training, resume and extraction on the CPU. Its docstring is the manual. ising-latent-v1's
  extraction runs from five plans, `extract-s6.json` to `extract-s10.json`, one per confirmatory
  seed so that each kernel's output stays under 18 GB, from the wheel built at the freeze tag's
  commit, which each plan's `commit` key pins.

The harness trains a planted-stripe toy on the CPU (`harness/jepa_toy.py`: validation error 1.37
to 0.34 in 150 steps; probes cannot tell it from its twin, prediction error can), breaks the loss
to check that the detector fires, and runs the extract round trip on runs the simulator itself
writes (`harness/jepa_ising.py`). The toy uses d = 32 and 32 bursts a step at a flat twice the
production learning rate after warm-up: a cosine to 1e-6 stalls a run this short (0.42 at step 200
instead of 0.31). Its budget is a number of steps, never of wall-clock time, so the toy is
deterministic.
