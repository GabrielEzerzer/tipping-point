"""The AUC of each pair (exploratory, not pre-registered): slow ramps against fixed-T nulls, the
Mann-Whitney AUC with a hierarchical bootstrap CI for every sensitivity cell (window x bandwidth)
and horizon h, and at the headline cell the scorer's alarm rows. |m| at the dense cadence or, with
--latents, the four pairs ising-latent-v1 judges at the matched cadence. An indicator in RAW has no
bandwidth to vary, so one cell per window.

The AUC is P(ramp's s above null's s) whatever a pair's direction, and the median s of each side
says which way the indicator moved: s is Kendall's tau over the bath's cooling, so s < 0 is an
indicator falling as the bath cools. That is what sets surprise's directions, so this script needs
none of them; its alarm rows do, which --alarm-cells none skips.

The cell loop mirrors tipping_point.ews.ising._cells (the same prepare / positives / null_slices /
Pipeline / _auc_row / _alarms calls). A cell whose segment and window do not fit the ramp (the 40
tau_c window at h = 2 and L = 16, whose ramps start at eps_L = 8.68) is listed as skipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tipping_point.ews import ising
from tipping_point.ews.trend import RAW, Pipeline

from explore import common


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--alarm-cells", choices=["headline", "none"], default="headline")
    p.add_argument("--headline-only", action="store_true", help="the headline cell alone")
    p.add_argument("--bootstrap", type=int, default=None)
    args = p.parse_args(argv)
    common.be_nice()
    clock = common.Clock()
    doc = common.make_doc(args)
    if args.bootstrap:
        doc["scoring"]["bootstrap"] = args.bootstrap
    pilot, inputs = common.load(args, doc)
    tau_c = pilot.tau(args.L)
    ind, sens, sc = doc["indicators"], doc["sensitivity"], doc["scoring"]
    segment = float(doc["event"]["segment"])
    horizons = [float(h) for h in doc["event"]["horizons"]]
    band = (max(horizons) + segment, min(horizons))
    kind = common.cadence(args)
    prep = ising.prepare(inputs, kind, tau_c, float(ind["thin"]), band)
    clock(f"L = {args.L}: tau_c {tau_c:.3f}, {prep.sweeps:.4f} sweeps per sample")
    rng = np.random.default_rng(args.seed)
    headline = (float(sens["headline"]["window"]), float(sens["headline"]["bandwidth"]))
    pairs = common.pairs(args, doc)
    windows = [headline[0]] if args.headline_only else [float(w) for w in sens["window"]]
    bandwidths = [headline[1]] if args.headline_only else [float(b) for b in sens["bandwidth"]]

    auc_rows, alarm_rows, skipped = [], [], []
    for window_taus in windows:
        window = prep.samples(window_taus, tau_c)
        for bandwidth_taus in bandwidths:
            bandwidth = bandwidth_taus * tau_c / prep.sweeps
            top = (window_taus, bandwidth_taus) == headline
            null_scores: dict[Any, list[np.ndarray]] = {}
            for h in horizons:
                for observable, name, direction in pairs:
                    raw = name in RAW
                    if raw and bandwidth_taus != headline[1]:
                        continue  # no bandwidth to vary: one cell per window
                    try:
                        pos_rows, length = ising.positives(prep, observable, h, segment, window)
                    except ising.IsingDataError as why:
                        skipped.append(
                            {"pair": [observable, name], "h": h, "window": window, "why": str(why)}
                        )
                        continue
                    span = window + length - 1
                    pipe = Pipeline(name, window, length, bandwidth, int(ind["lag"]))
                    if (observable, pipe) not in null_scores:
                        neg_rows = [
                            ising.null_slices(s.observable(observable), span) for _, s in prep.nulls
                        ]
                        null_scores[observable, pipe] = [pipe(g) for g in neg_rows]
                    neg = null_scores[observable, pipe]
                    pos = [pipe(g) for g in pos_rows]
                    cell = {
                        "indicator": name,
                        "observable": observable,
                        "cadence": kind,
                        "L": args.L,
                        "h": h,
                        "window_taus": window_taus,
                        "bandwidth_taus": np.nan if raw else bandwidth_taus,
                        "headline": top,
                        "direction": direction,
                    }
                    row = ising._auc_row(cell, pos, neg, sc, rng)
                    pc, nc = np.concatenate(pos), np.concatenate(neg)
                    row |= {
                        "window": window,
                        "bandwidth": np.nan if raw else bandwidth,
                        "length": length,
                        "span": span,
                        "pos_median_s": float(np.nanmedian(pc)),
                        "pos_q05_q95_s": np.nanquantile(pc, [0.05, 0.95]).tolist(),
                        "neg_median_s": float(np.nanmedian(nc)),
                        "neg_q05_q95_s": np.nanquantile(nc, [0.05, 0.95]).tolist(),
                        "ci_method": "percentile, hierarchical bootstrap: seeds, then replicas",
                    }
                    auc_rows.append(row)
            clock(f"AUC cell window {window_taus:g} x bandwidth {bandwidth_taus:g} done")
            if top and args.alarm_cells == "headline":
                common.oriented_pairs(args, doc)
                alarm_rows += [
                    row | {"headline": True}
                    for row in ising._alarms(doc, prep, args.L, tau_c, window, bandwidth)
                ]
                clock(f"alarms at window {window_taus:g} x bandwidth {bandwidth_taus:g} done")

    common.save(
        common.out_dir(args) / "auc_alarms.json",
        {
            "provenance": common.provenance(args, pilot),
            "runs": common.run_ids(inputs),
            "sweeps_per_sample": prep.sweeps,
            "auc": auc_rows,
            "alarms": alarm_rows,
            "skipped": skipped,
        },
    )
    for r in auc_rows:
        if r["headline"]:
            print(
                f"{r['observable']:<9} {r['indicator']:<9} h={r['h']:g}  AUC {r['auc']:.3f} "
                f"[{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]  n {r['n_pos']}/{r['n_neg']}  "
                f"median s {r['pos_median_s']:+.3f} / {r['neg_median_s']:+.3f}"
            )
    clock("done")
