# Market

The market experiment asks whether warnings read from a JEPA trained on order books come earlier
or more reliably than the classic indicators, before volatility-regime shifts in the BTCUSDT and
ETHUSDT perpetual futures. This page covers the regime labels and the cascade proxy, the reports
and the GARCH floor, how the warning test cuts its segments, the same-input tier, and the market
JEPA: its pilot, its primary trainings and the latent extraction. The last section covers the
coin panel, a separate study on 30 perpetuals.

The three market pre-registrations are frozen at their git tags. The labels file is in use; the
two warning tests are not yet scored:

| file | tag | status |
|---|---|---|
| [market-labels-v1.toml](../paper/prereg/market-labels-v1.toml) | `prereg/market-labels-v1` | frozen; the confirmatory labels are made from it |
| [market-ews-v1.toml](../paper/prereg/market-ews-v1.toml) | `prereg/market-ews-v1` | frozen; not yet scored |
| [market-latent-v1.toml](../paper/prereg/market-latent-v1.toml) | `prereg/market-latent-v1` | frozen; not yet scored |

The spans: the models train on 18 January to 31 August 2023 and validate on September 2023, the
exploratory span is 2023-Q4, and the confirmatory span, which the tests score, is 1 January 2024
to 30 September 2026.

## Regime labels

The labels come from Binance's free USD-M archive and need no engine
(`python/tipping_point/src/tipping_point/market/`, stdlib, numpy and pyarrow). `fetch` downloads a
span's 1-minute klines, 5-minute metrics and monthly funding into
`<data dir>/archives/binance/<kind>/<symbol>/` (`--dest` for another folder), checks each file's
SHA-256 against the `.CHECKSUM` beside it, skips files already verified, and lists every file in
that folder's `manifest.json` with its URL, size, sha256 and row checks: gaps in the minute or
5-minute grid, duplicate stamps, rows out of order. `labels` turns a fetched span into 5-minute
bars (Garman-Klass variance, close, volume, taker-buy share, open interest, funding) and labels
them: a two-state Gaussian HMM (hidden Markov model) on deseasonalised, baseline-relative log
variance, refitted every Monday by deterministic EM, read through as-of filtered probabilities and
a hysteresis rule into episodes, events, flickers and null stretches. A run lands in
`<data dir>/labels/<run_id>/`: `manifest.json` with every source file and parameter, Parquet
tables `bars`, `labels`, `refits`, `episodes` and `nulls` keyed by `ts_ns`, then `complete.json`.
Without `--prereg`, every labeller parameter is a command-line option and days from 2024-01-01 are
refused. The clean window C (bars with no high-volatility label before an onset) and the
refractory Q (bars after an onset in which no other event begins) are fixed: a labels file with
another value is refused.

```sh
uv run python -m tipping_point.market fetch --start 2022-07-01 --end 2023-12-31
uv run python -m tipping_point.market labels --start 2022-07-01 --end 2023-12-31
```

The confirmatory span opens only through the frozen file. With `--prereg`, `labels` and `cascade`
take the span, the symbols and every parameter from it, and refuse to start unless the file is the
one frozen at its tag, the modules that make labels and cascades are byte for byte the tag's
commit's, the command line and `ews/prereg.py` have no change beyond HEAD, and the checkout has no
uncommitted change or untracked file (the manifest's `producer.dirty` is false). The manifest
records the sha256 of `uv.lock`, which pins the locked libraries but not the installed environment,
so pre-registered runs use `uv run --locked --exact`: `--exact` syncs the environment to the lock,
removing any package the lock does not name, and `--locked` refuses a lock that is out of date with
`pyproject.toml` instead of rewriting it. A cascade run lists the windows that close from the
confirmatory day's 00:00 on, as `cascade.recall` reads, so a candidate whose window closes in the
first half hour can have its onset (`ts_ns`) up to 30 minutes before midnight.

```sh
uv run --locked --exact python -m tipping_point.market labels --prereg paper/prereg/market-labels-v1.toml
uv run --locked --exact python -m tipping_point.market cascade --prereg paper/prereg/market-labels-v1.toml
```

## Cascade proxy

The cascade proxy, `market/cascade.py`, reads the same fetched files. No free source holds the
span's liquidations, so a cascade is inferred per symbol from a trailing 30-minute window: open
interest in contracts falls by at least `a` (as a log ratio), the price moves at least `b` one way,
taker flow leans at least `c` the same way, and volume reaches `v` times the median 30-minute
volume of the trailing 28 days. A long cascade falls and a short squeeze rises; a drop in open
interest alone fires neither. A window touching a missing open-interest step is not evaluated and
is counted. Candidates within 24 hours merge into the earliest, and both symbols firing within an
hour count as one market event with two asset instances; both counts are reported. The pooled list,
with each instance's onset and the bounds of its window 12 hours either side, lands as the table
`cascades` of a run in `<data dir>/labels/<run_id>/`. Without `--prereg`, `a`, `b`, `c` and `v` are
options (defaults 3 %, 2.5 %, 0.15 and 3), and days from 2024-01-01 are refused as the labels' are;
the frozen values are in `market-labels-v1.toml`. How open interest lines up with price is not an
option, because the feed's timing fixes it: a bar's row holds the open interest at its start and
its close the price at its end, so each stamp pairs with the klines before it and all the legs span
the same 30 minutes (in every month of 2023, on both symbols, a change in open interest correlates
most with the 1-minute returns between its two stamps). A `liquidations` table in a market run is
read when there is one; until then the run says `liquidations: not recorded`. The first 28 days
only fill the trailing medians, so this span lists 2023 alone:

```sh
uv run python -m tipping_point.market cascade --start 2022-12-04 --end 2023-12-31
```

`market/documented_cascades.toml` lists documented cascades with their sources. With
`TP_CASCADE_RECALL` set, `uv run pytest python/tipping_point/tests/market/test_cascade.py -k
documented_cascades_on_real_days` runs the rule over each event found fetched, with the 29 days
before it; without it the test skips and says so. Five of the six events lie in the confirmatory
span, so the check gives only what the rule says inside each event's own days and reports recall
with no floor: nothing is to be tuned on it. Two more checks read 2023 days if they are fetched, and
otherwise skip and say so: the metrics stamp's timing, and 2023-08-17 at the default values.

## Reports and the GARCH floor

`market/reports.py` writes two reports on labelled series, each as Parquet tables plus a Markdown
summary. The label-count report counts events, episodes, flickers, durations and time high per
asset and month, clocks the onsets by UTC hour and weekday, and counts onsets that coincide across
assets, the null hours and the 4-hour slices they hold. The stability report compares the as-of
labels with an oracle (the final theta, smoothed over the whole sample) and with each declared
variant and a plain threshold rule, follows theta over the refits with its flagged fits and the
jumps they make in P, and gives the filter's detection delay. The calendar of CPI, NFP and FOMC
releases, the cascade candidates and the aggTrades cross-check are optional inputs; a report
without one says "not supplied". `market/releases.csv` is the committed calendar: the 18 CPI, 18
payroll and 12 FOMC statement times of July 2022 to December 2023 in UTC, each with the BLS or
Federal Reserve page that gives its date. `market/nulls.py` holds the GARCH(1,1)-t simulator, the
delay simulation, the event-time shift and the circularity floor.

The floor simulates a minute GARCH(1,1)-t, bridges each minute to 10-second sub-steps, builds
1-minute OHLC, runs the real bars and labeller, and scores 1-minute close-to-close returns, which
gives the AUC a clustering process with no critical dynamics produces by construction. Its GARCH
parameters, pipelines and `scoring.auc` are inputs, and no module in `market/` may import the
feature code (`tests/market/test_circularity.py`). `harness/market_garch.py` fits the parameters
per symbol on the 2023 pilot year's 1-minute returns and writes them as the report `garch` of a
labels run; `harness/market_floor.py` holds them frozen in `PILOT_GARCH`.

`reports` writes the label-count and stability reports of a labels run into the run's `reports/`
folder. It labels the run's span again from the fetched files and refuses to write anything unless
that gives back the run's own tables. The label-count report always counts the committed calendar.
`--cascades` adds the pooled list of a cascade run of the same symbols whose span lies inside the
labels run's, and the summary says when the candidates cover less than the labels span.
`harness/market_floor.py` runs the floor with the run's own labeller and the variance and AC1
pipelines, and writes it to the same folder, with one AUC per path beside the pooled one so that
the floor's own spread shows.

```sh
uv run python -m tipping_point.market reports --run <labels run folder> --cascades <cascade run folder>
uv run python harness/market_garch.py --run <labels run folder>
uv run python harness/market_floor.py --run <labels run folder> --garch BTCUSDT --days 546
```

## The market warning test

`ews/market_segments.py` cuts what market-ews-v1 scores from a labels run of the frozen
market-labels file:

- `close_returns`: a symbol's 1-minute close-to-close log returns on the minute grid, NaN for a
  minute with no kline or whose previous minute has none (`prereg.MISSING_MINUTE`).
- `slot_profile`: per slot of the labeller's 2 x 288 (weekday or weekend, by 5-minute time of day),
  over a fit span of pilot days, the returns' root mean square ("scale") or mean ("mean"), the two
  `prereg.DESEASON_METHODS`. `deseasonalise` divides each return by its slot's scale, or subtracts
  its slot's mean.
- `check_labels`: the labels run's manifest, once the labels file is frozen at its tag and the run
  was labelled with its parameters (and its stamp, when it has one).
- `build`: the event segments per (symbol, horizon) and the control segments per symbol.
- `eth_only`: which ETH onsets have no BTC event onset within a window (reporting's eth-only).

A segment is `span` minutes, `Pipeline.span` = window + length - 1. An onset is the start of a
5-minute bar; at horizon h bars its segment ends with the minute that closes at onset - 5h minutes,
so it holds nothing of the h bars before the onset (`market.nulls.pre_onset_rows`, the floor's
rule). The length is the same at every horizon, so at long horizons a segment starts before the
labels' clean window of C bars and may read bars with H = 1 (the high-volatility state): each
segment's share of minutes in such bars is reported. Controls are the non-overlapping spans cut
in time order from the start of each null stretch (`market.nulls.null_rows`), one set for every
horizon, then optionally a seeded subsample of them: with subsample n > 0, when a symbol has more
than n kept controls, n are drawn without replacement, symbols in sorted order from one generator
seeded with `seed`, and kept in time order. A segment that holds a minute with no return is
excluded (the same segments for every indicator), and so is one that reaches outside the series
given; both are counted. With `score_from`, an onset before that day, and a control segment
starting before it, is not cut at all: the series may start earlier, as history for the first
days' segments, without its own days being scored.

The confirmatory span (from 2024-01-01) opens only through a frozen market-ews file, as the market
command's `--prereg` opens it for labels: `build` refuses a series, onset or control segment
reaching it unless `scoring` is a market-ews file frozen at its tag that names this labels file,
and a run labelling those days must carry the frozen labels' stamp in its manifest. The span also
stays closed until a market-latent file is frozen in the same repository: at least one
`prereg/market-latent-v<k>` tag, each holding `paper/prereg/market-latent-v<k>.toml`, a
market-latent file of that id and version that passes the strict validator and whose `[baseline]`
names this market-ews file's tag and sha256, read from the tag itself, so a checkout of the
market-ews tag's commit is admitted once the latent tag is fetched; where the working tree also
holds that file, it must be the tagged blob. Neither test is scored on those days before both are
pre-registered. An environment switch, as `cascade.recall` takes, would open those days to any
caller; the frozen file ties them to the pre-registered test. With it, `build` also takes nothing
the file fixes from its caller: the horizons, the span (window + length - 1), the subsample and the
seed must be the file's. And it scores only the confirmatory span, as `score_from` 2024-01-01
would: an onset before it, and a control segment starting before it, is not cut, so a labels run
reaching back into 2022 and 2023 puts none of those events into a confirmatory cell, while their
minutes still serve as history.

`build` also returns each symbol's `eligible` bars (`market.nulls.eligible_onsets` on its labels),
the bars the event-time shift may move its onsets to, restricted to the scored span the same way.
`shifted_rows` moves the onsets kept in one horizon's event block to those bars, at least
`min_offset_bars` away, through `market.nulls.shift_onsets`, and cuts their segments: one that
holds a minute with no value or reaches outside the series has NaN there, so it drops out of that
draw's AUC and is never redrawn. `shift_threshold` takes the 97.5th percentile of the draws' AUCs
against the controls, and NaN, an inconclusive cell, when any draw has no AUC.

Every random draw of the scoring has its own stream, so that no result hangs on the order a
scorer visits the cells: `bootstrap_aucs` draws a cell's AUC bootstrap (events, then controls,
each resampled per symbol) from `default_rng([seed, 1, indicator, horizon, group])`, the shift
draws from `default_rng([seed, 2, horizon, group])`, and `surrogate_pvalues` draws each segment's
phase surrogates from `default_rng([seed, 3, indicator, horizon, group, control, row])`, a control
segment at horizon 0; `group_code` numbers the groups (pooled 0, each symbol its position plus 1,
eth-only last). The second entry keeps the streams apart, since numpy pads a seed list with zeros.
The module reads labels and is fed the series; the labels package (`market/`) imports none of it.

### Scoring a market-ews file

`ews/market_score.py` applies a market-ews file as written. The confirmatory scoring is
`python -m tipping_point.ews score --prereg paper/prereg/market-ews-v1.toml --run <labels run>
--cascades <cascade run> --book <bybit-book root>`, with `--archive` defaulting to
`<data dir>/archives/binance`. Before it reads any table, kline or book feature, it refuses in
these cases:

- the file is not frozen, fails the strict validator, or has a prediction other than the decision
  rule it carries out (`PREDICTION`);
- the labels file `data.labels` names is not frozen at that tag with that sha256;
- the labels run does not span the labels file's whole span;
- the market-latent gate is closed;
- the segment, shift, same-input and floor code (`HELD`) differs from the market-ews tag's commit,
  or the scorer itself is uncommitted, or the checkout is dirty;
- the trial budget would be passed.

Before it reads any table, the scorer checks every archive file it will read (klines, metrics and
funding from the day before the first alarm day to the labels run's end) against the labels run's
manifest: each must be recorded there with the same URL, pass its `.CHECKSUM` and carry the
recorded sha256, and a file the labels run recorded as missing must still be missing. It refuses on
any difference, in both modes. Once the archive is read, it runs the same check again before
evaluating, so a file replaced during the read refuses the run before any trial is logged.

The cascade run must carry the frozen labels' stamp and end at the same day. Once its folder is
begun, the scoring charges the budget again, so a concurrent scoring's torn folder refuses it. It
appends its 32 cells to `ews/trials.jsonl` before it writes any table or the report, so a crash
leaves them spent, and a second scoring is refused.

`--exploratory` runs the same evaluation on pilot days as a dry run. It needs a labels run ending
before 2024-01-01 and scores from the run's first labelled day (start plus the labeller's warm-up).
Its folder goes to `--out`, by default under `<data dir>/ews-exploratory`, never inside
`<data dir>/ews`. Its manifest and report are marked exploratory, and it logs no trial. It refuses
a labels or cascade run reaching 2024-01-01 from the manifest alone, and `build` refuses any series,
onset or control segment reaching it.

The result folder holds `manifest.json`, `report.md` and `complete.json`, plus one parquet table
per folder:

- `cells`: the 32 counted cells, each with its AUC and CI, floor, shift threshold, the three
  comparisons, a verdict and `shift_dropped_near_btc`, the moved eth-only onsets the shift dropped
  because they fell near a BTC onset (0 outside the eth-only group);
- `hypotheses`: each hypothesis's verdict and how many decisive cells passed, failed or were
  inconclusive;
- `shift_draws`: every draw's AUC;
- `seasonality`: the raw and scale series' cells, descriptive;
- `strata`: AUCs per UTC onset hour, no CI;
- `cascades`: the cascade stress test;
- `surrogates`: the share of event and of control segments with phase-surrogate p <= 0.05;
- `alarms`: the threshold from the 2023 control segments, then the false alarms per day and the
  share of events alarmed 5h minutes ahead;
- `same_input`: the book tier's cells, with `price` the returns on the tier's segments;
- `floor`: the floor's cells;
- `segments` and `same_input_segments`: cut, kept and excluded counts;
- `high_share`: each kept event segment's share of minutes in bars with H = 1, keyed by symbol,
  horizon and onset;
- `high_share_mean`: the mean of those shares per symbol and horizon.

A cell passes when its CI's lower end is above 0.5, above its floor AUC and above the shift's
97.5th percentile. It is inconclusive when any of the three is NaN, and fails otherwise. A
hypothesis passes when all six decisive cells pass, fails when one fails, and is inconclusive
otherwise.

The scorer settles the points the file leaves open as follows:

- Eth-only cells take ETH's floor. In each shift draw the eth-only rule is applied again to the
  moved ETH onsets: a moved onset with a real BTC onset within `eth_only_bars` either side is
  dropped from that draw and counted in `shift_dropped_near_btc`.
- Descriptive CIs reuse the counted cell's bootstrap stream.
- The 2023 alarm controls are cut from the labels run's null stretches by the builder's control
  rule, keeping those wholly in 2023. A control segment carries one score, so it alarms when that
  score is above the threshold, and its days are its minutes / 1440.
- The cascade segments are cut as event segments are, from the first scored day. Eth-only
  cascades are the ETH cascades with no BTC cascade onset within `eth_only_bars`.
- The book tier deseasonalises each series on its own profile and then applies one common mask:
  every minute where the deseasonalised return or any deseasonalised feature is not finite is
  emptied in all of them.

### Scoring a market-latent file

`ews/market_latent_score.py` applies a market-latent file as written, on the segments of the
market-ews file its `[baseline]` names. Its segments come from `market_segments.build`, and it
builds its own event-time shift on `market.nulls.shift_onsets`. It reuses `ews/market_score.py`
for the archive, labels and cascade checks and the same-input tier's series, and `ews/prereg.py`
for the trial budget. The confirmatory scoring is `python -m tipping_point.ews score --prereg
paper/prereg/market-latent-v1.toml --run <labels run> --cascades <cascade run> --book <bybit-book
root> --archive <binance archive> --latents <merged latents root> --gate <extraction gate.json>
--pins <pins.json>`. Before it reads any latent table, kline or book feature, it refuses in these
cases:

- the file is not frozen, fails the strict validator, has a prediction other than the decision rule
  it carries out (`PREDICTION`), or a `[scoring.twin]` or `[scoring.classic]` rule other than the
  one it carries out (seeds combined by the nested seed bootstrap, every pair judged, each against
  its matching classic indicator, no ablation);
- the baseline is not frozen at its tag with its sha256, or fails the market-ews checks above;
- the labels run does not span the labels file's whole span;
- the code (`HELD`: the market-ews scorer's held code plus `margins`, `fss` and `ising`) differs
  from the tag's commit, the scorer is uncommitted, or the checkout is dirty;
- the trial budget would be passed;
- the extraction gate record is not format 2, its prereg stamp (id, version, sha256, tag, commit,
  tag object and text) is not the frozen file's, it was made from changed or uncommitted code, its
  pins sha256 is not that of the pins file given, or its checkpoints are not the file's rows;
- any `[[latents.checkpoints]]` row differs from pins.json, value for value;
- a merged table's manifest names another gate record, another checkpoint, PCA or scaling sha256,
  symbol, span or minute count, or the parquet it names does not carry its sha256; a table whose
  `ts` column is not the expected minute grid is refused when read.

The archive check, the labels check and the cascade run's stamp are the market-ews scorer's. The
scoring charges the budget again once its folder is begun, and appends its 1312 cells (1280 latent
cells, one per pair, horizon, primary, trained or twin checkpoint and group, and the 32 classic
cells) to `ews/trials.jsonl` before it writes any table or the report, so a crash leaves them spent
and a second scoring is refused.

Each checkpoint's latents are read as two series, the fit span and the confirmatory span with its
lead-in minutes, and each column is deseasonalised on a slot profile fitted per column, checkpoint
and read symbol over the fit span. Every random draw comes from a stream the file states:
bootstrap `[seed, 4, pair, horizon, model, group]`, shared by a unit's ten checkpoints and its
classic indicator; shift `[seed, 5, horizon, read symbol]`; phase surrogates `[seed, 6, pair,
horizon, model, primary seed, checkpoint, read symbol, control, row]` (horizon 0 for a control
segment);
twin margins `[seed, 7, pair, horizon, model, group]`. A classic cell's own CI keeps the market-ews
stream `[seed, 1, indicator, horizon, group]`.

A unit is a pair, a horizon, a model and a group; the decisive units are h = 5 and 6 in the pooled
and per-symbol groups, twelve per pair. A unit warns when all five trained AUCs' CIs exclude 0.5
on the same side and each |AUC - 0.5| is above the 95th percentile of the shift draws' |AUC -
0.5|. It is learned when the combined trained-minus-twin margin's CI is above 0.02 and every
primary's margin has its sign; it fails when the CI is below 0.02 or a sign differs, and is
inconclusive otherwise. It is beyond classic when every primary's |AUC - 0.5| minus the classic
indicator's AUC - 0.5 is above 0 and above the 97.5th percentile of the same difference over the
shift draws, both sides moved together. Any NaN makes the unit inconclusive. A pair warns when every
decisive unit warns; learned and beyond classic are read only for a pair that warns, each passing
when every decisive unit passes. A hypothesis passes when it warns, is learned and is beyond
classic; it fails when any of the three fails and is inconclusive otherwise.

The result folder holds `manifest.json`, `report.md` and `complete.json`, plus one parquet table
per folder:

- `cells`: the 1280 latent cells, each with its AUC, CI, counts and shift threshold;
- `classic_cells`: the 32 classic cells on the same segments, with no verdict of their own;
- `units`: each unit's warns, learned and beyond-classic outcomes;
- `margins`: each unit's per-seed and combined trained-minus-twin margins and the combined CI;
- `beyond_classic`: each primary's latent-minus-classic difference, its shift threshold and the
  paired bootstrap CI, which is reported and decides nothing;
- `hypotheses`: each hypothesis's verdict and its three parts;
- `shift_draws` and `classic_shift_draws`: every draw's AUC;
- `segments`: cut, kept and excluded counts, with `excluded_finite` for the finite-score mask;
- `surrogates`: the share of event and control segments with two-sided phase-surrogate p <= 0.05;
- `strata`: AUCs per UTC onset hour, no CI;
- `cascades` and `classic_cascades`: the cascade stress test on the same checkpoints;
- `same_input` and `same_input_segments`: the book tier on the latents' segments;
- `incremental_rv`: whether each latent score adds to realised volatility;
- `high_share` and `high_share_mean`: as the market-ews scorer writes them.

The scorer settles the points the file leaves open as follows:

- A minute has no latent value when its table marks it not valid or a column is not finite after
  deseasonalising, in any of the 20 primary tables read on that symbol; the slot profile is fitted
  on valid minutes only.
- A combined margin of exactly 0 fails the sign test. The rule that all five trained AUCs sit on
  one side of 0.5 is also a condition of learned, so a unit whose AUCs straddle 0.5 fails it.
- The eth-only shift drops a moved ETH onset near a real BTC onset, as the market-ews scorer does.
- The classic side's direction is taken from the baseline's hypothesis ("greater": AUC - 0.5 as it
  stands).
- The same-input tier keeps the paired segments whose book rows are finite, with stream-1 CIs.
- Incremental RV is the sum of squared raw returns over the segment's last 60 minutes. Logistic fits
  on standardised RV alone and on RV plus the latent score, with a small ridge, give two AUCs whose
  difference takes its CI from the unit's stream-4 draws.
- Cascades are paired and finite-masked as events are, with stream-4 CIs on the unit's key.
- Learned and beyond classic carry no verdict for a pair that does not warn.
- No floor: the file names none for latents.

`--exploratory` runs the same evaluation on the latents' fit span, 2023-01-18 to 2023-09-30, the
days the models were trained and validated on, with a labels and cascade run ending before
2024-01-01. Its numbers check the code and are no evidence for or against any hypothesis; the
manifest and report say so. It reads only the fit series, scores the fit span with eligible onsets
inside it, writes to `--out`, by default under `<data dir>/ews-exploratory`, never inside
`<data dir>/ews`, and logs no trial. It refuses a labels run, cascade run, latent table, kline or
book series reaching 2024-01-01.

## The same-input tier

The same-input tier, `ews/same_input.py`, scores the Bybit book features themselves as classic
series, so that a latent win cannot be put down to the book holding information Binance's price
series lacks. In market-ews-v1 it is descriptive. `aggregate` turns one symbol-day's 1-second
feature table (`book.features.reduce_run`) into 1-minute values per feature: "sum" adds the
minute's 60 seconds, "mean" averages them. A minute with any gapped second (`gap`, which a missing
second also sets) has no value, the rule `jepa.days` applies to a minute frame
(`prereg.SAME_INPUT_MISSING`); a value that is not finite outside gap seconds is refused, as
`jepa.days` refuses it. `read` gives a symbol's minutes over a span of days from a bybit-book data
folder, one `market_segments.Minutes` per feature on the minute grid; a day with no feature file
has no value in any minute. The minutes go through the same steps as the returns:
`market_segments.slot_profile` and `deseasonalise` (`[seasonality]`'s method and fit span, per
feature), then `market_segments.build`, which cuts the event and control segments by the same
rule and excludes and counts a segment that holds a minute with no value. Every feature has a
value in the same minutes, so every feature keeps the same segments. `read` refuses any day from
2024-01-01 before it opens a file, so the data folder's `confirmatory` range is never opened; days
route by `market.book.RANGES` to `training` or `exploratory`, and a day in neither (before the
archive starts) has no file. With `scoring`, a market-ews file frozen at its tag, as `build` takes
it, `read` also routes days to `confirmatory`, and the aggregation must be the file's
`[same_input]` one. A feature file must name its symbol and day and be the feature format this
code reduces (`book.features.FORMAT`), as `jepa.days.ParquetSource` demands; the format read is
recorded.

## The market JEPA pilot

`harness/m4_pilot.py` holds the market JEPA's CPU pilot and its rules, written and committed before
any pilot ran. It chose the primaries' size and step count.

- **Runs.** Four trainings: BTCUSDT and ETHUSDT, each at size S (dim 64) and M (dim 96), model seed
  100 (never a primary or spare), 20,000 steps, warm-up 1,000 (steps / 20, as on Ising), every
  other setting `minutes.Config`'s default, on the CPU in float32. `stop_on_collapse` is off: the
  floors are what this pilot recalibrates, so a run is not cut short by floors set on a toy; the
  detector's verdict is still recorded. Each run checkpoints every 500 steps and resumes by the
  same command.
- **Split.** The training span [2023-01-18, 2023-10-01), split as `jepa/days.py` asks, with the
  validation days a block after the training days: training 2023-01-18 .. 2023-08-31 (226 days),
  validation 2023-09-01 .. 2023-09-30 (30 days). The primaries use the same split.
- **Validation error.** The JEPA's own error is measured against each model's own target encoder,
  which moves during training and differs between sizes: on a 1,000-step smoke run (BTC, 10 days)
  the monitor's `val_error` fell to 0.44 at step 250 and rose to 0.59 (S) and 0.63 (M) by step
  1,000 while effective rank grew, and the richer M latent scored worse than S. A lower JEPA error
  can mean a poorer latent, so it can rank neither sizes nor steps. The validation error is
  instead a fixed task in input space: a ridge probe (`jepa.probes.ridge`, alpha 1e-3) from the
  target encoder's latent of minute t (`minutes.readouts`) to the mean of each scaled feature over
  minute t + 1, over minute pairs both valid, fitted on validation days 1-15 and scored per day on
  days 16-30 as the mean over features of the squared error over the fit set's variance (1 = the
  fit mean). It is measured for `final.pt`, for the copy saved at step 10,000 and for the step-0
  twin; the JEPA surprise of `final.pt` is kept beside it, descriptive. A run is usable when
  `final.pt`'s mean probe error is below its twin's.
- **Size.** M, unless S reaches the same validation error. S is chosen when, on both assets, both
  runs are usable and S matches M: the upper end of the 95% bootstrap interval of the mean over the
  scored days of (S - M), paired by day (2,000 draws, seed 0), is at most `TOLERANCE` = 2% of M's
  mean. Otherwise M. If an M run is not usable, the rule chooses no size and `decide` stops there.
- **Steps.** On a run of the chosen size, e(s) is the mean probe error of its model at step s. The
  primaries train 10,000 steps if, on both assets, e(10,000) - e(20,000) <= 2% of e(20,000), and
  20,000 otherwise: never below 10,000 (about 21 passes over the training minutes), never above
  20,000 (the pilot sees no further). In a 20,000-step run the learning rate at step 10,000 is
  still high, so e(10,000) overstates what a 10,000-step run reaches: the rule leans to 20,000,
  which costs time, never validity.
- **Floors.** For each floored statistic (`spatial_std`, here the spread of a latent over the
  minutes of a chunk; `pooled_std`, the spread of chunk means over chunks; `effective_rank`), h is
  its minimum over the usable runs of the chosen size, both assets, at evaluations from step 1,000
  on; the new floor is h / 5 rounded down to one significant figure. Patience stays 3. The new
  floors replace `minutes.Config`'s defaults in a commit before the primaries, and `test_market`'s
  collapse mutant must still fire under them; if it does not, the floors are set outside this
  rule. The VICReg weights stay unless a usable pilot run's monitors fire under the new floors.

```sh
uv run --extra cpu python harness/m4_pilot.py train --asset btc --size M \
    --data <book features>/training --out <data>/m4-pilot      # again to resume
uv run --extra cpu python harness/m4_pilot.py decide --out <data>/m4-pilot
```

`train` writes `<out>/<asset>-<size>/jepa/<run_id>/` (the trainer's layout) and, once the training
is complete, `<out>/<asset>-<size>/validation.json`. `--steps` and `--limit` (the last K training
and first K validation days) are for smoke runs only; `decide` refuses a run that used either.

The outcome, as `market-latent-v1.toml` discloses it: probe error at step 10,000 was 0.6327 for
BTC S and 0.7746 for ETH S, beside 0.7463 and 0.9385 for their untrained twins. The upper 95 % CI
of S - M was -0.010 (BTC) and -0.009 (ETH), and the step-10,000 model was within 2 % of the
step-20,000 one on both symbols (BTC 0.6327 against 0.6313, ETH 0.7746 against 0.7752). By the
rules: size S (d = 64) and 10,000 steps. The pilot read book features reduced before a fix that
seeds each day's first second from the previous day's book (feature format 4); the size and step
count were kept for the primaries without rerunning it.

## The market JEPA primaries

`harness/m8_primaries.py` trains the models market-latent-v1 scores and pins them. Per symbol there
are five primaries (model seeds 101 to 105) and two spares (106 and 107), each a size S (d = 64)
training of 10,000 steps on its own symbol's 1-second book features, feature format 4, with the
pilot's split and settings. The untrained twin of each is its step-0 checkpoint. `train` refuses
any other seed.

```sh
uv run --extra cpu python harness/m8_primaries.py train --asset btc --seed 101 \
    --data <book features>/training --out <data>/m8-primaries      # again to resume
uv run --extra cpu python harness/m8_primaries.py pins --out <data>/m8-primaries
```

`train` runs one training per process in `<out>/<asset>-s<seed>/jepa/<run_id>/`, beside a copy of
the training range folder's `manifest.json`, and resumes after a kill by the same command. The
range folder must be a `training` one of this code's feature format naming no day on or after
2024-01-01. `pins` writes `<out>/pins.json`: per run the asset, seed, role, run id, the data
manifest's commit and feature format, and the sha256 of the trained model (`final.pt`, step
10,000), of its twin, of its PCA basis and of the scaling record. It refuses an incomplete run, one
that fails `train.verify`, one whose data reach 2024, an asset and seed trained twice, and any
planned asset and seed without its one pinnable run, so no seed can be dropped after its run is
seen. market-latent-v1 pins all 14 checkpoints; a spare replaces a primary only when the primary's
training fails `train.verify` or the extraction refuses its checkpoint, never for its scores.

## Market latent extraction

`tipping_point.jepa.market_extract` reads every pinned checkpoint and its twin, each on both
symbols, into per-minute series: `surprise` (the error of the one-minute-ahead prediction of a
minute's latent from the 95 minutes before it), `surprise5` (the five-minute-ahead error),
`pc_1` to `pc_8` (the latent on the checkpoint's own PCA basis) and `valid`. Features are scaled
by the read symbol's own record, never refitted. The spans are `fit` (the frozen file's
seasonality fit span), `exploratory` (2023-10-01 to 2023-12-31) and `confirmatory` (to 2026-09-30,
starting 514 minutes before 2024-01-01 so that the first segments have their full context). The
fit and confirmatory spans open only under a gate record, which `gate` writes only from a
market-latent file frozen at its tag, a clean tree, and checkpoints equal to the pins.

```sh
uv run python -m tipping_point.jepa.market_extract gate --prereg paper/prereg/market-latent-v1.toml --pins <pins.json> --out <gate.json>
uv run python -m tipping_point.jepa.market_extract run --pins <pins.json> --primaries <dir> --data <book features> --span exploratory --out <dir>
uv run python -m tipping_point.jepa.market_extract merge --units <dir> --span exploratory --pins <pins.json> --out <dir>
uv run python -m tipping_point.jepa.market_extract verify <dir>
```

`run` cuts a span into blocks of 16 days and writes one unit per checkpoint and block, one shard at
a time, resuming after a kill. `merge` checks every unit of a span, from one machine or several,
and joins them into one series per checkpoint and symbol. `verify` checks units or series against
the sha256 their records name and exits 1 on any problem. `python -m tipping_point.jepa.market_extract
--help` gives the full rules.

## Coin panel

The panel study asks whether leverage and positioning data warn of liquidation cascades on 30
Binance USDT perpetuals. Its data and label layer is `python/tipping_point/src/tipping_point/panel/`,
which reuses the archive reader, the bars and the cascade rule above unchanged. Bars start on
2021-10-01; thresholds and cascades start on each coin's first day with a threshold, and the label
grid covers the development span 2022-01-01 to 2023-12-31. Every day from 2024-01-01 is held back
as the test span: every command refuses one, and every request is checked for one before it is
sent.

**Coins.** A symbol enters the ranking when it is a USD-M symbol matching `[A-Z0-9]+USDT`, is not
one of the index and stablecoin products `BTCDOMUSDT`, `DEFIUSDT` and `USDCUSDT`, publishes a
monthly 1-day kline file for every month of 2023, and publishes a daily metrics file on both
2021-12-01 and 2023-12-31. The ranking is by 2023 quote volume summed from those 1-day klines,
each file checked against its `.CHECKSUM` and required to hold one row for each day of its month,
none twice. The panel is BTCUSDT, ETHUSDT and the top 28 others; the command fails when fewer
symbols are eligible. It is committed in `panel/coins.txt`, which every other command reads:
BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT BNBUSDT LTCUSDT MATICUSDT BCHUSDT LINKUSDT AVAXUSDT ADAUSDT
TRBUSDT FILUSDT 1000SHIBUSDT DYDXUSDT MASKUSDT FTMUSDT GALAUSDT ETCUSDT LINAUSDT DOTUSDT NEARUSDT
EOSUSDT CRVUSDT SANDUSDT ATOMUSDT AXSUSDT STORJUSDT MKRUSDT.

```sh
uv run python -m tipping_point.panel coins
uv run python -m tipping_point.panel fetch
uv run python -m tipping_point.panel labels
```

`coins` derives the ranking again from Binance Vision (its bucket listing of
`data/futures/um/monthly/klines/` names every symbol), prints it, and exits 1 when the panel it
gives differs from the committed list. `fetch` downloads the panel's klines, metrics and funding
for 2021-10-01 to 2023-12-31 into the same folder as `market fetch`, `--workers` files at a time
(default 16), through `market fetch`'s own per-file step: each file is checked against its
`.CHECKSUM`, a file that fails is refused and never written, a 404 is recorded as missing, a re-run
skips verified files, and `manifest.json` keeps the same form. `labels` writes one run to
`<data dir>/labels/<run_id>/` with the tables `thresholds`, `cascades` and `grid`; its manifest
records the coin list, the span, the constants, HEAD and the sha256 of `uv.lock`. The archive root
the panel reads holds no held-back file: `fetch` and `labels` scan the names under it (symlinked
folders are not followed), opening none, and refuse a root where any name carries a day from
2024-01-01 on before they read its manifest or any file. They then refuse a manifest that lists
such a name, which a file moved away or never found leaves behind, before using any record in it.
Every command exits 1 with a message on a refused day, a bad file or a bad argument.

The run's checks add up, per coin and per kind, the row checks the archive's `manifest.json`
records for each file read: rows, files with a header, unsorted rows, and for klines and metrics
the rows off the file's grid, the slots holding more than one row and the runs of empty slots. The
manifest counts a row stamped outside its own day together with a row stamped off the step grid
inside it, as `off_grid`, so it cannot show a row outside its period on its own, and the run does
not refuse on it; funding files get no period check at all.

**Thresholds.** For each coin and each UTC day d, `sd_ret` and `sd_oi` are the sample standard
deviations (n - 1) of the cascade rule's `price_change` and `oi_change` over every window closing
in [d - 90 days, d), from the coin's first evaluated window on, whose change is known: the price
change of every window with both closes, the open-interest change of every window with open
interest at both ends. The coin's rule on day d is `cascade.Params(a = 3.7 sd_oi, b = 5.0 sd_ret,
c = 0.15, v = 3.0)`, every other field at its default, for every window closing on d. A day has no
threshold when the rule evaluated fewer than 28 x 288 x 0.8 of the windows closing in that span,
when it did not evaluate more than 5% of them, known change or not, or when `sd_ret` or `sd_oi` is zero or not finite; its windows
are left out of the run's window counts and are censored in the grid. Starting at the first
evaluated window keeps a coin's ramp-up, before its open interest and 28 days of volume history
start, out of both the spreads and the count of missing windows.

The spreads read more windows than the rule evaluates, on purpose. The rule skips a window when
any of its seven open-interest rows is missing, but the change over that window is still known
when the two end rows exist, and the windows around a missing row are not ordinary ones: the
archive loses open-interest rows during falls. On 2021-12-04 three rows are missing (05:00, 05:25
and 05:30 UTC), so 13 windows closing from 05:00 to 06:00 were not evaluated; 8 of them have a
known open-interest change, a drop of 6% to 12% in BTCUSDT and of 14% to 24% in DYDXUSDT, whose
price fell by up to 25% over the same windows. Spreads over the evaluated windows alone would
leave those out and set a lower bar for the 90 days after: on the development data the price spread would be lower by up to
28% (BCHUSDT, 2022-01-01) and by more than 5% on 1,063 of the 21,900 coin-days, the open-interest
spread lower by up to 28% (DYDXUSDT, 2022-01-01) and by more than 5% on 503. The price leg is
complete, since every window of the development data has a known price change. The open-interest
leg is not: the change over a window with a missing end row is unknown, and such a window could
hide a change large enough to move `a` and with it a later label. The 5% limit on windows not
evaluated bounds the share of such windows, not their effect; in the run they are at most 2.4% of
a day's windows (SANDUSDT, 2023-12-15), with a median of 0.03%, and 56% of the coin-days have at
least one. The label is therefore defined on the changes that are known, and is not an estimate of what
a complete open-interest history would give: a row is scored when its own detection and label
windows are complete, whatever open-interest rows its 90-day history lacks inside the 5% limit.
Every development day has a threshold. A day's threshold reads nothing from that day on, so no day
sets its own threshold, but earlier development days set the thresholds of later development
days. A coin whose typical move shrinks or grows is held to the same multiple of its recent moves
throughout. The multiples, the 90 days, the minimum and the 5% limit are constants in
`panel/labels.py`.

**Detection.** Each day's windows go through `cascade.candidates` unchanged, with that day's rule,
on a slice of bars that holds all the history the rule reads before the day, keeping only the
day's own windows; with one rule on every day this gives exactly what one call over the whole span
gives. Candidates are merged per coin and pooled across coins as `market cascade` does, from each
coin's first day with a threshold, so an episode that fired in the hours before the grid starts is
known: its hours are skipped and a later firing within 24 hours merges into it.

**Tables.** `thresholds` has one row per coin per day from the first day of bars: `ts_ns` the day's
start, `symbol`, `windows` (the evaluated windows in its 90 days), `missing` (the windows in its 90
days, from the coin's first evaluated window on, that were not evaluated), `oi_unknown` (those
among them with no known open-interest change), and `sd_ret`, `sd_oi`, `a` and `b`, null on a day
without a threshold. The run's checks give each coin's worst day among the days with enough
evaluated windows, `worst_missing_share` and `worst_missing_ns`, the number of those days over
5%, `days_over_max_missing`, and the same for the windows with no known open-interest change,
`worst_oi_unknown_share` and `worst_oi_unknown_ns`. `cascades` has the `market cascade` schema,
whose `ts_ns` is the onset, and `in_span`, true when the instance begins (below) at or after the
grid's first hour; the instances that begin earlier are kept so that their skip and merges can be
seen, and the run's counts of instances and market events take only `in_span` rows. A market event
that pools instances from before the grid's first hour is counted with only its `in_span`
instances. `grid` has one row per coin per UTC hour t of the development span. It dates each
instance by begin, the start of its first firing window (`fired_ts_ns` - 30 minutes), not by its
onset. The onset is the first extreme bar inside that window and can sit up to 25 minutes after it
opens, so an hour between the two would see part of the event's price move and open-interest drop
before t. Dated by begin, nothing of a labelled event lies before t. For H = 1, 6 and 24 hours:

- `long_H`, `short_H`: an instance of that kind begins in [t, t + H).
- `skip`: an instance has its first firing at or before t and its last firing no more than 24
  hours before t, so an episode known at t is under way or just ended. An hour between an
  instance's begin and its first firing is not skipped, since that firing is not known at t: it is
  an ordinary row, positive only if another instance begins in [t, t + H). In the run there are 395
  such coin-hours, 392 of them kept as ordinary rows (the other 3 are skipped for an earlier
  episode), and at most one of those is positive at any horizon. Since they are a few hundred rows
  among roughly half a million negatives, scoring them as negatives or leaving them out moves an
  AUC by less than 0.001.
- `censored_H`: a window closing in [t - 24 hours, t + H + 30 minutes) is not clean, or does not
  exist yet because it closes past the span's end or the coin's last complete bar. The forward part
  holds exactly the windows that begin in [t, t + H), the last of them closing at t + H + 25
  minutes. The backward 24 hours are there because a missing stretch of open interest can hide an
  earlier firing, after which a later firing would look like a new event rather than a skipped
  follow-up. A window is clean when the rule could evaluate it, its day has a threshold, and no
  kline bar is missing in the 28 days and 25 minutes before it opens, the history its volume median
  and onset read; that last condition censors only and never changes detection.

Censoring is symmetric: a row with a gap in its horizon is censored even when an instance was found
in it, so that a gap removes positives and negatives alike. The grid holds labels only: a feature
at hour t may read only bars that start before t and metrics rows stamped at or before t.

**Limits.**

- An episode takes the direction of its first firing. A short squeeze within 24 hours of a long
  cascade is part of that episode, not a second event, and gives no `short_H`; the same holds the
  other way round.
- The coin list is chosen on 2023 volume, so scores on 2022-2023 describe the cohort that was among
  the largest in 2023, not the coins a trader would have picked at the time. For any later span the
  list is fixed in advance, and a coin that stops trading is censored from its last complete bar.
- The same multiple of a standard deviation does not give heavy-tailed coins the same event rate.
  In the development run the instances that begin in the grid's span number from 10 to 56 per coin
  (median 26.5): BTCUSDT 56, ETHUSDT 30, SOLUSDT 24, XRPUSDT 30, DOGEUSDT 27, BNBUSDT 19,
  LTCUSDT 31, MATICUSDT 21, BCHUSDT 17, LINKUSDT 23, AVAXUSDT 26, ADAUSDT 27, TRBUSDT 14,
  FILUSDT 44, 1000SHIBUSDT 33, DYDXUSDT 32, MASKUSDT 33, FTMUSDT 31, GALAUSDT 39, ETCUSDT 32,
  LINAUSDT 23, DOTUSDT 25, NEARUSDT 23, EOSUSDT 29, CRVUSDT 10, SANDUSDT 33, ATOMUSDT 25,
  AXSUSDT 20, STORJUSDT 14, MKRUSDT 18.

**Inputs.** `panel inputs --labels <run id>` writes one run to `<data dir>/panel-inputs/<run_id>/`
with the table `inputs`, one row per coin per hour t of the labels run's grid in the grid's order.
Its manifest names the labels run and the sha256 of its grid and of its `thresholds` table. That
table must give one row for each coin and each UTC day of the grid, or the command refuses; an hour
in day d scales its volatility inputs by that day's `sd_ret` and `b`, set from the 90 days before d
(see Thresholds), and those inputs are missing on a day whose row has no threshold. The command
takes the archive's file list only through the same check as `labels`, and refuses an archive that
holds or lists a file naming a day from 2024-01-01 on before it reads anything.

```sh
uv run python -m tipping_point.panel inputs --labels <labels run id>
```

Timing rule: an input at hour t reads only 5-minute bars that start before t, open interest
stamped at or before t (that of the bar starting at t is allowed; its close is not), and the
long/short ratio and funding rate of the bar before t, t - 5 min. A funding settlement falls on the
hour and Binance stamps it on the hour or a few milliseconds after, so reading the bar at t would
make the input depend on those milliseconds. The tests change, blank or delete every later bar and
row, through the command on planted archive files as well as on bars, and check that no input at t
moves. Windows are counted in 5-minute bars (1 h = 12, 6 h = 72, 24 h = 288, 7 d = 2016,
28 d = 8064) and end just before t, at t for open interest. A rolling value needs at least 5/6 of
its window present, else it is missing; so does every 28-day mean or median, counting only values
observed in that bar, not carried into it. r is the 5-minute log return of close. The command
refuses a span with a funding file missing.

- `vol`: `vol_1h`, `vol_6h`, `vol_24h`, `vol_7d`, `vol_28d` = log(mean r^2 over the window /
  (sd_ret^2 / 6)), where sd_ret^2 / 6 turns the day's 30-minute variance into a 5-minute one;
  `near_24h` = the largest |30-minute log return| of the last 24 h over the day's `b`; `vsurge_1h`
  = log(mean volume over 1 h / median 5-minute volume over 28 d); `volume_24h` = log(mean volume
  over 24 h / mean volume over 28 d).
- `price`: `ret_1h`, `ret_6h`, `ret_24h`, `ret_7d`, `ret_28d`, the log return over the window, from
  the close just before it to the last close before t (missing when either close is); `aret_24h`
  and `aret_7d`, their absolute values; `dd_7d` and `dd_28d`, the log of the last close over the
  window's highest close; `ru_28d`, over its lowest.
- `flow`: `flow_1h`, `flow_24h` = (2 x taker-buy volume - volume) / volume with both volumes summed
  over the window's complete bars, as `market cascade` reads flow.
- `classic`: `var_tau_24h` and `ac1_tau_24h`, Kendall's tau against time of the rolling variance and
  of the rolling lag-1 autocorrelation of r (72-bar windows, each needing 5/6 of its values or
  pairs) over the last 288 bars; `var_tau_7d` and `ac1_tau_7d`, the same on hourly log returns
  (84-hour windows) over the last 168 hours. Each rolling statistic comes from its own window's
  values alone (a flat window has none), so it does not change with how far back the series
  starts. Kendall's tau is `ews.trend.kendall_tau`, over the values present when at least 5/6 are.
- `lev`: `oi_24h`, `oi_7d`, the log change in open interest; `oi_vs_28d`, the log of open interest
  over its 28-day median; `funding`, the rate settled before t - 5 min; `funding_rel`, funding minus
  its mean over the last 28 days' 672 hourly readings, each taken at the bar before the hour;
  `ls`, the log of Binance's long/short account ratio (`count_long_short_ratio` of the metrics
  files, read from the same verified files) at t - 5 min; `ls_rel`, `ls` minus its 28-day mean
  ending at the same bar.
  Open interest and the ratio are carried forward at most 12 bars (1 hour) across gaps, and zero or
  negative values count as missing. Binance's metrics files leave the ratio empty from 2021-12-30 to
  2022-01-19 on every panel coin, so `ls` is missing until 2022-01-19 14:00 UTC and `ls_rel` until
  its 28-day window holds 5/6 of its values again, at 2022-02-11 22:00 UTC.

For every own-coin input x, the market input `m_x` is the mean of x at hour t over the coins that
have a value, missing when fewer than 15 do.

**Ladder.** `panel ladder --inputs <run id>` scores nested sets of inputs against each other and
writes one run to `<data dir>/panel-ladder/<run_id>/`: the tables `oof` (each row's out-of-fold
scores, one column per rung and side), `scores`, `folds` (each rung's AUC in each scored quarter),
`comparisons` and `rows` (the row accounting), `report.md`, and a copy of the plan.

```sh
uv run python -m tipping_point.panel ladder --inputs <inputs run id>
```

Everything it does comes from a plan file; `panel/ladder.toml` is the development plan, over
2022-01-01 to 2023-12-31:

- Horizons 24 h (primary), 6 h and 1 h; sides long, short and any.
- Rungs, each a list of input groups, every group meaning its own-coin inputs and their market
  means unless marked own-only: `vol` (vol); `vol+price` (vol, price); `vol+price+flow` (vol, price,
  flow); `full` (vol, price, flow, lev with lev own-only); `full+mlev` (vol, price, flow, lev, own
  and market); `classic` (vol, classic).
- Comparisons, first minus second: full - vol; vol+price - vol; vol+price+flow - vol+price;
  full - vol+price+flow; full+mlev - full; classic - vol.
- Model: two L2 logistic regressions per rung and horizon, one for long_H and one for short_H,
  C = 0.01 (the penalty is |w|^2 / 2C, the intercept unpenalised), fitted by Newton's method on
  inputs standardised with the training fold's mean and standard deviation and clipped to +-5.
  Scores: long = p_long, short = p_short, any = 1 - (1 - p_long)(1 - p_short); labels are the
  side's own, and any is long or short.
- Rows: the grid rows that are not skipped, not censored at that horizon and have every input of
  every rung in the plan, one common row set per horizon. The run reports how many rows and
  positives each condition removes, in that order.
- Folds: the eight calendar quarters, scored two ways, each table carrying the `scheme`. In
  `loqo` each quarter is left out in turn and training rows within 31 days of it, on either side,
  are dropped. In `forward`, from the quarter `forward_first` names (2, counting from 0, so six
  quarters are scored), a quarter is scored by a model fitted only on rows more than 31 days before
  it, as the held-back test span will be. The gap must cover the inputs' longest look-back (28 days,
  one bar and the funding rate's carry of up to 8 hours, longer than the 1-hour carry of the
  other inputs), the longest horizon, the labels' 30-minute lead and their 24-hour look-back,
  30.36 days in all; the plan refuses a shorter one. The 90-day thresholds that scale
  `vol` and `near_24h`, and that set the labels, are not a direct look-back and the gap does not
  cover them; `ladder-gap92.toml` runs the plan with a 92-day gap (below). A scored quarter with rows but no
  training rows, training rows that hold no positives or only positives for a side, or a scored
  row without a finite score, stops the run.
- Scores per scheme, rung, side and horizon: the mean over quarters of the AUC inside each quarter
  (the primary score), the same mean weighted by each quarter's positives (`fold_mean_weighted`),
  the pooled out-of-fold AUC, and the within-coin AUC, the AUC inside each coin and quarter
  averaged with weights by positives over the cells that hold both classes; each also without
  BTCUSDT and ETHUSDT (the same fits, scored on the other coins' rows).
- Comparisons: the difference of the primary score with a joint block bootstrap. Each quarter is
  cut into 14-day blocks of calendar time from its first day (7 per quarter, the last shorter);
  each draw takes, in every quarter, as many blocks as it has, with replacement, and every coin's
  rows of a drawn block come together. 2000 draws with the plan's seed, a quarter's draws seeded by
  the seed, the horizon and the quarter, so every rung and both schemes share them and each
  difference is paired. The run gives the estimate, the bootstrap standard error, `se_quarters`
  (the standard deviation of the per-quarter differences over the square root of their number),
  the 2.5%, 5% and 97.5% points, the share of draws above zero, the per-fold differences and how
  many folds are above zero. A draw that takes no positive or no negative in a quarter cannot
  score that quarter, and its fold mean averages over the quarters it does score; the column
  `complete_draws` gives the share of draws that score every quarter, per horizon and side. On
  the development span at 24 h it is 0.98 long and 0.91 short in both schemes, and 0.995 and 0.997 any. Most of
  the shortfall is the third quarter of 2022, where 4 of the 7 blocks hold no long positive and 5
  hold no short positive; in `loqo` the first quarter adds a little, as its rows start when
  `ls_rel` returns on 2022-02-11 22:00, in its third block.

The bootstrap does not recompute AUCs from rows. An AUC is a sum over (positive, negative) pairs, so
for each fold, rung and side the run counts once, for every pair of blocks (a, b), the pairs C_ab
with the positive in a and the negative in b that the score orders correctly (ties count one half).
A draw that takes block a w_a times has AUC = sum w_a w_b C_ab / ((sum w_a P_a)(sum w_b N_b)), with
P and N the blocks' positives and negatives: exactly the AUC of the rows the draw takes, from a
7 x 7 matrix per fold. All 2000 draws then cost a few matrix products, and the whole plan's
bootstrap runs in seconds.

Reading the scores. The eight-quarter `loqo` score is retrospective: a quarter's model is fitted on
later quarters as well as earlier ones, on a coin cohort fixed at the end of the span. The
`forward` scheme previews the held-back test. The bootstrap intervals are conditional on the
fitted models and on the quarters observed: they redraw rows, not fits or new quarters, and
`se_quarters` is the between-quarter view of the same difference. The 1 h scores are secondary,
because part of what they rank is cascades already under way on other coins at t, which the market
means carry. `full` holds the coin's own leverage inputs only and `full+mlev` adds their market
means; the rung list was fixed before any score was computed. The scores rank rows and say nothing
about calibration.

**Development results.** On labels run `20261124T020153Z-panel-20220101-20231231` the inputs have a
threshold on every coin-day of the grid and no funding file is missing. Rows at each horizon:

| | 24 h | 6 h | 1 h |
|---|---|---|---|
| grid rows | 525600 | 525600 | 525600 |
| skipped | 20423 | 20423 | 20423 |
| censored | 14862 | 9721 | 8297 |
| missing an input | 29074 | 29094 | 29110 |
| scored | 461241 | 466362 | 467770 |
| long / short positives scored | 15481 / 2043 | 3990 / 534 | 674 / 90 |

Primary score (`fold_mean`) at 24 h, `loqo` / `forward`:

| rung | long | short | any |
|---|---|---|---|
| vol | 0.591 / 0.572 | 0.631 / 0.596 | 0.600 / 0.580 |
| vol+price | 0.655 / 0.593 | 0.749 / 0.701 | 0.638 / 0.591 |
| vol+price+flow | 0.681 / 0.626 | 0.822 / 0.773 | 0.648 / 0.608 |
| full | 0.712 / 0.666 | 0.849 / 0.804 | 0.675 / 0.634 |
| full+mlev | 0.680 / 0.659 | 0.826 / 0.769 | 0.645 / 0.621 |
| classic | 0.582 / 0.555 | 0.619 / 0.590 | 0.591 / 0.566 |

Comparisons at 24 h, estimate with the 2.5% and 97.5% bootstrap points, `loqo` then `forward`:

| comparison | long | short | any |
|---|---|---|---|
| full - vol | 0.121 [0.069, 0.175]; 0.094 [0.035, 0.160] | 0.218 [0.122, 0.309]; 0.208 [0.100, 0.302] | 0.075 [0.024, 0.124]; 0.055 [0.001, 0.124] |
| vol+price - vol | 0.064 [0.012, 0.121]; 0.021 [-0.029, 0.089] | 0.118 [0.062, 0.179]; 0.105 [0.023, 0.168] | 0.038 [-0.010, 0.091]; 0.011 [-0.033, 0.072] |
| vol+price+flow - vol+price | 0.026 [0.012, 0.040]; 0.033 [0.008, 0.055] | 0.072 [0.032, 0.113]; 0.072 [0.020, 0.126] | 0.010 [-0.006, 0.023]; 0.018 [-0.007, 0.044] |
| full - vol+price+flow | 0.030 [0.014, 0.045]; 0.040 [0.021, 0.054] | 0.028 [0.007, 0.048]; 0.031 [0.013, 0.051] | 0.027 [0.011, 0.039]; 0.026 [0.009, 0.042] |
| full+mlev - full | -0.032 [-0.059, -0.010]; -0.007 [-0.056, 0.020] | -0.023 [-0.032, -0.009]; -0.035 [-0.057, -0.004] | -0.029 [-0.053, -0.007]; -0.013 [-0.048, 0.012] |
| classic - vol | -0.009 [-0.032, 0.018]; -0.017 [-0.052, 0.023] | -0.013 [-0.039, 0.014]; -0.005 [-0.025, 0.018] | -0.009 [-0.034, 0.017]; -0.013 [-0.047, 0.023] |

Three plans beside `ladder.toml` each change one thing. At 24 h on the long side:

| plan | full - vol | full - vol+price+flow | classic - vol |
|---|---|---|---|
| `ladder.toml` | 0.121 [0.069, 0.175]; 0.094 [0.035, 0.160] | 0.030 [0.014, 0.045]; 0.040 [0.021, 0.054] | -0.009 [-0.032, 0.018]; -0.017 [-0.052, 0.023] |
| `ladder-gap92.toml` | 0.109 [0.056, 0.169]; 0.056 [-0.039, 0.119] | 0.022 [0.007, 0.038]; 0.037 [0.019, 0.052] | 0.010 [-0.015, 0.033]; -0.003 [-0.030, 0.026] |
| `ladder-block28.toml` | 0.121 [0.074, 0.175]; 0.094 [0.050, 0.161] | 0.030 [0.016, 0.046]; 0.040 [0.026, 0.056] | -0.009 [-0.027, 0.021]; -0.017 [-0.053, 0.024] |
| `ladder-public.toml` | not in the plan | not in the plan | -0.013 [-0.031, 0.012]; -0.007 [-0.034, 0.020] |

- `ladder-gap92.toml` (`gap_days = 92`, which covers the 90-day thresholds) leaves the scored rows
  unchanged and drops more training rows: the first forward quarter trains on 30987 rows instead
  of 73271. The forward `full - vol` interval then reaches below zero; every other interval in the
  table stays on the side of zero it was on, or spans zero as before.
- `ladder-block28.toml` (`block_days = 28`) leaves every estimate unchanged and moves the 2.5% and
  97.5% points of the 24 h comparisons by at most 0.019; at 24 h `complete_draws` falls to 0.92
  long and 0.87 short in `loqo`, 0.93 and 0.87 in `forward`.
- `ladder-public.toml` (no rung with `lev`) builds its common rows without `ls` and `ls_rel`, so it
  scores 490314 rows at 24 h (one missing an input), from 2022-01-01 17:00. Its `vol` scores 0.580
  long in `loqo` and 0.565 in `forward`, and `vol+price - vol` is 0.074 [0.019, 0.135] and 0.029
  [-0.026, 0.098].

An optional `[external]` entry (`name`, `path` to a Parquet file relative to the plan, and `scheme`,
the scheme its scores were fitted under) brings in out-of-fold scores fitted elsewhere, such as a
sequence model's, with columns `coin` (text), `hour` (the grid's `ts_ns`, as integers of any
width or timestamps of any unit, none null), `horizon` (integer), `side` (text), `score`
(floating point), `fold` (integer, the ladder's quarter counted from 0 at 2022-01-01) and
`scheme` (text, a single value equal to the plan's `scheme`). They are scored and compared like a
rung on the same common rows of that scheme, and the run refuses a file that lacks one of those
columns or gives one another type, misses a scored row, gives one twice, gives a score that is
not finite, gives a row a fold other than the ladder's or holds scores fitted under another
scheme.

Both commands refuse any day from 2024-01-01 on: `inputs` refuses a labels run whose span reaches
it, and `ladder` a plan, an inputs run, a grid or an external file that does. They read the time
bounds of the grid, the thresholds, the inputs table and the external file's `hour` from the
files' Parquet metadata before hashing or reading them, and refuse a file whose time column is
neither integer nor timestamp or whose metadata gives no bounds for it; then they check each file's
sha256 against the run that wrote it and refuse a run its `complete.json` does not mark clean.
