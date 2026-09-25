# ruff: noqa: E501
"""The v4 analysis (docs/2026-09-22_EXPERIMENT_PLAN.md §3–§5, §12): the evaluation stage's
outputs and the in-training probes -> the paper's numbers, the capacity grid and the results
against predictions table, produced mechanically. Written before any v4 result was looked at.

    uv run python scripts/analyze_v4.py --root hub-v4      # the pulled Hub layout (hf_runs.py pull)
    uv run python scripts/analyze_v4.py --root .           # one box, or the Mac: ./runs, ./logs/v4/eval
        [--out logs/v4/analysis]

Reads: <root>/runs/*_<run id>/ (config.json, summary.json, probes.jsonl) and the evaluation
outputs <root>/box-<k>/logs/v4/eval/<kind>/<run>.json (or <root>/logs/v4/eval/...), with the judged
sample files beside them (<run>.judge.jsonl, written on the Mac by judge_samples.py and
story_quality.py judge). Nothing is copied: eval_v4.py's collectors run over every box's
directory at once (a run evaluated on two boxes is an error).

Writes (to --out, gitignored so no v4 number is committed by accident):
  analysis_v4.json     every section below, cell means with 95 % t intervals over seeds
  dense_v4.json        the mean in-training probe trajectories per cell (eval_v4.collect_dense)
  predictions_v4.md    the results against predictions table (also inside analysis_v4.json)

Sections: persona (normalized persona, fraction kept, the sweep, per act, held-out acts);
finetune (attachment and spillover, the spillover slope with its CI, the random cast, contrast
d, the matrices with their predictions, Ben's position, position × persona with pred_excess
as the main line, R20); ladder (the "positive" test per rung and size); dense (the onset, plain
and smoothed, censoring, the formation time); mech (steering, d′, the gain); flow (the scans
per block, random band, separation left, the crossing, attention to the name); timeline (R18:
the readouts at fixed steps, the dose curve, the late slope); samples (free text with UNCLEAR as
its own column, actor tracking, story quality); training (validation loss); capacity (A6).

Choices this script makes that the plan does not fix are listed in NOTES and printed at the
top of the predictions table.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "scripts")
import eval_v4 as ev
import run_v4
from binding_onset_loss import onset_window, rolling_median

from dramatis.forks import HELD_OUT, PERFORMED
from dramatis.provenance import git_commit, git_dirty, utc_now

ACTS = list(PERFORMED)
U3 = ["lie", "cheat", "tease"]  # the untaught acts fitted together; steal is reported separately (its offset, plan §4)
MAIN = run_v4.MAIN
DEPTH = ["L1", "L2", "L3", "L4"]
SIZES = MAIN + DEPTH
Y = run_v4.Y
TOK = run_v4.BATCH * run_v4.BLOCK  # tokens per update
REF = "v4-lex-y100-100M"  # the normalization cell of plan §4
SMOOTH = 11  # probe points in the rolling median of the smoothed onset and the formation time (110 updates ≈ 1.8M tokens)
FORM_FRACS = (0.5, 0.9)  # R4: the formation time at half and at 90 % of the smoothed end value, both reported
N_BOOT = 2000  # seed-bootstrap replicates
MAIN_FT = [f"ft-{a}-Ckind" for a in ACTS]

NOTES = [
    "Inside / outside compares the cell mean (over seeds) with the stated range; the 95 % interval is shown beside it.",
    "Position × persona: pred_excess (Ben's position minus the other kind names', × persona) is the main line, pred (Ben's alone) the second (plan §4, decided 2026-09-23).",
    "Position × persona fit (A4): the main fit leaves out the random-cast (y = 0) and clean-twin cells, where the base model has no class axis, so the K→N gap that Ben's position is measured in is noise (plan §4, decided 2026-09-23 before launch); every cell as the second line.",
    "Intervals: 95 % t intervals over seeds for cell means; a seed bootstrap (2,000 replicates, seeds resampled with replacement within every cell, percentile interval) for everything computed across cells or as a ratio or median: slopes, the A4 chain, contrast d, fraction kept, the gain, onset medians, flow crossings, predicted matrices, free-text changes. With 8 seeds a percentile bootstrap runs slightly narrow.",
    f"Smoothed onset (second line): the same estimator on a centred rolling median of {SMOOTH} probe points; the end value is the smoothed last point.",
    "Formation time (R4): the first probe at which the smoothed k = 0 persona reaches 50 % and 90 % of its smoothed end value (both reported), as a fraction of the one-pass length; the plan's 'within the first 10 %' (from the 1× prior: at its final level by the first probe, 1.5k steps, on a different story mix) is tested on the 90 % line.",
    "Onset per cell: the median over seeds with censored seeds (no onset by the cap) counted as later than every onset; undefined if half or more are censored.",
    "Spillover per finetune for the random cast and contrast d: the mean over the untaught acts among lie / cheat / tease (steal left out, as at 1×); d = difference of means ÷ pooled s.d., one value per finetune (the 1× unit).",
    "Spillover slope in the capacity grid: the mean of the lie / cheat / tease slopes over that size's y cells.",
    "Predicted spillover matrices (R10, R11): the pooled spillover slope on each column's act × the persona of the names that performed the taught act (per-name N − K, averaged over the group).",
    "Paired-cast block contrast: mean of the partner cells minus mean of the cross-pair cells, normalized per column, steal column left out; per seed, then over seeds.",
    "Flow crossing: the first block L at which removing the class direction at the prediction site leaves less than removing it at the name, having left at least as much at L - 1 (the name curve crossing the site curve upward; cell-mean curves). Changed 2026-09-25 from the frozen 'first block at which the site removal leaves less', which fired at L = 0 for every finetune (preliminary report, flag 1).",
    "R18 at 25 clean steps: the spillover is interpolated between the probes at 20 and 30; Ben's position has checkpoints every 50 clean steps, so its early clean readout is at 50.",
    "Methods, the onset continuation passes (R9, run_v4.continuation): a closing-label run (R2, R6) whose last 24-item binding probe is below 2 nats continues from its last checkpoint on the same corpus with a new batch seed, a fresh optimizer, no warmup and a constant LR at the run's final_lr, for up to three more passes, stopping 2,000 updates after an onset; R19's closing-label runs get none (their no-onset runs are censored at one pass). The pass boundaries are in dense.runs[*].pass_ends.",
]


# ------------------------------------------------------------------ statistics


def t975(df):
    """Student t, two-sided 95 %, with eval_v4's table (the next smaller df where not listed)."""
    if df > 30:
        return 1.96
    return ev.T975[max(k for k in ev.T975 if k <= df)]


def ols_ci(x, y):
    """OLS of y on x with an intercept: slope, intercept, the slope's s.e. and 95 % t interval
    (n − 2 degrees of freedom), n. None where fewer than 3 finite points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return {"slope": None, "intercept": None, "se": None, "lo": None, "hi": None, "n": int(n)}
    s, i = np.polyfit(x, y, 1)
    resid = y - (s * x + i)
    se = float(np.sqrt(resid @ resid / (n - 2) / ((x - x.mean()) @ (x - x.mean()))))
    t = t975(n - 2)
    return {"slope": float(s), "intercept": float(i), "se": se, "lo": float(s - t * se), "hi": float(s + t * se), "n": int(n)}


def cohen_d(a, b):
    """(mean a − mean b) ÷ the pooled s.d. (n − 1 weights)."""
    a, b = np.asarray([v for v in a if v is not None], float), np.asarray([v for v in b if v is not None], float)
    if len(a) < 2 or len(b) < 2:
        return None
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else None


def wilson(k, n, z=1.96):
    """A rate with its Wilson 95 % interval."""
    if n == 0:
        return {"rate": None, "lo": None, "hi": None, "k": 0, "n": 0}
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"rate": p, "lo": float(c - h), "hi": float(c + h), "k": int(k), "n": int(n)}


def onsets(steps, vals):
    """(plain, smoothed) onset windows of one binding trajectory (plan §4: the first probe at or
    above half the end value, with the 2-nat floor; binding_onset_loss.onset_window); None = no
    onset."""
    traj = [(s, v) for s, v in zip(steps, vals, strict=True) if v is not None]
    if not traj:
        return None, None
    sm = rolling_median(np.array([v for _, v in traj], float), SMOOTH)
    return onset_window(traj), onset_window(list(zip([s for s, _ in traj], sm.tolist(), strict=True)))


def censored_median(values):
    """Median over seeds with None (censored: no onset by the cap) as later than every onset;
    None if half or more are censored."""
    n = len(values)
    if n == 0 or sum(v is None for v in values) * 2 >= n:
        return None
    return float(np.median([v if v is not None else np.inf for v in values]))


def formation_step(steps, vals, frac):
    """The first probe step at which the smoothed trajectory reaches frac × its smoothed end
    value (the end must be positive), else None."""
    v = np.array([np.nan if x is None else x for x in vals], float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return None
    s, v = np.asarray(steps)[ok], rolling_median(v[ok], SMOOTH)
    if v[-1] <= 0:
        return None
    return int(s[np.argmax(v >= frac * v[-1])])


def crossing(name_left, site_left):
    """The handoff: the first block L at which the name curve rises past the site curve from below
    (removing the class at the site leaves less than removing it at the name at L, and did not at
    L - 1), or None. Changed 2026-09-25 (Kevin; the preliminary report's flag 1): the frozen rule
    ("the first block at which the site removal leaves less than the name removal") fired at L = 0
    for every finetune. That is not the crossing the prediction meant (name removal mattering early,
    site removal late, the name curve crossing the site curve upward), and it has a mechanistic
    cause: the finetune is keyed on Ben's identity, so erasing the class axis at his embedding
    leaves >= 1 of the attachment (docs/2026-09-25_REPORT_name_identity_2.md). The base models'
    crossings are unchanged (their name curve starts below the site curve)."""
    for L in range(1, len(name_left)):
        a0, b0, a, b = name_left[L - 1], site_left[L - 1], name_left[L], site_left[L]
        if None not in (a0, b0, a, b) and b0 >= a0 and b < a:
            return L
    return None


def boot(groups, stat, n=None, seed=0, keep=False):
    """Seed bootstrap: resample the items (training seeds) of every group with replacement,
    independently per group, recompute stat, and take the 2.5 / 97.5 percentiles. groups:
    {key: [one item per seed]}; stat(groups) -> a number or None (replicates giving None are
    dropped; with fewer than half left there is no interval). Returns {"value": stat on the data,
    "lo", "hi", "n_boot"} (and "samples" with keep=True)."""
    n = n or N_BOOT
    point = stat(groups)
    if point is None or any(len(v) < 2 for v in groups.values()):
        return {"value": point, "lo": None, "hi": None, "n_boot": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        v = stat({k: [g[i] for i in rng.integers(0, len(g), len(g))] for k, g in groups.items()})
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    out = {"value": point, "lo": None, "hi": None, "n_boot": len(vals)}
    if len(vals) >= n / 2:
        out["lo"], out["hi"] = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
    if keep:
        out["samples"] = vals
    return out


def scale_boot(b, k):
    """A boot() result with its value and bounds multiplied by k."""
    return {**b, **{key: (b[key] * k if b.get(key) is not None else None) for key in ("value", "lo", "hi")}}


def fmt(x):
    return "—" if x is None else f"{x:.3g}"


def mean_of(c):
    """The value of a ci95 dict ("mean"), an ols_ci dict ("slope"), a boot dict ("value") or a
    plain number."""
    if isinstance(c, dict):
        return c.get("mean", c.get("slope", c.get("value")))
    return c


def nanmean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


# ------------------------------------------------------------------ the layout


def find_layout(root: Path):
    """(eval directories, runs directory): the pulled Hub layout (box-<k>/logs/v4/eval) or one
    box's own (logs/v4/eval)."""
    evals = sorted(root.glob("box-*/logs/v4/eval")) or [root / "logs" / "v4" / "eval"]
    return [d for d in evals if d.exists()], root / "runs"


@contextmanager
def layout(eval_dirs, runs_dir):
    """Point eval_v4's loaders and run_v4.run_dir at several eval directories and one runs
    directory, so the collectors run unchanged on the pulled layout."""
    saved = ev._load, ev._load_meta, run_v4.run_dir

    def _load(kind):
        out, where = {}, {}
        for d in eval_dirs:
            for p in sorted((d / kind).glob("*.json")) if (d / kind).exists() else []:
                if p.name.endswith(".meta.json"):
                    continue
                if p.stem in out:
                    raise RuntimeError(f"{kind}/{p.name} is in both {where[p.stem]} and {d}")
                out[p.stem], where[p.stem] = json.loads(p.read_text()), d
        return out

    def _load_meta(kind, run):
        for d in eval_dirs:
            p = d / kind / f"{run}.json"
            if p.exists():
                return json.loads(p.read_text()).get("_meta")
        return None

    def run_dir(job_id):
        hits = sorted(runs_dir.glob(f"*_{job_id}"))
        return hits[-1] if hits else None

    ev._load, ev._load_meta, run_v4.run_dir = _load, _load_meta, run_dir
    try:
        yield
    finally:
        ev._load, ev._load_meta, run_v4.run_dir = saved


# ------------------------------------------------------------------ cells


def lex_cell(y, size, quarter=False):
    return f"v4-lex-{'quarter-' if quarter else ''}y{y}-{size}"


def parse_lex(cell):
    """(y as a float, size) of a lexical cell, else None."""
    m = re.fullmatch(r"v4-lex-y(\d{3})-(\w+)", cell)
    return (int(m[1]) / 100, m[2]) if m else None


def cast_groups():
    """The N-names that performed each act, per cast (prompts/cast_*.yaml)."""
    base = yaml.safe_load(open("prompts/cast.yaml"))["naughty"]
    paired = yaml.safe_load(open("prompts/cast_paired.yaml"))["naughty_by_group"]
    byact = yaml.safe_load(open("prompts/cast_byact.yaml"))["naughty_by_act"]
    return {
        "lex": {a: base for a in ACTS},
        "paired": {a: g["names"] for g in paired for a in g["acts"]},
        "byact": {a: byact[a] for a in ACTS},
    }


# ------------------------------------------------------------------ persona (R1-R3, R10-R12, R15)


def persona_section(k0):
    """Per-run normalized persona (each act ÷ the 100M exact cast's mean on that act, averaged
    over the performed acts) and per cell its 95 % CI; the fraction kept against y = 1 of the same
    size (and corpus size), per seed over the y = 1 cell mean; the sweep and per-act tables."""
    runs, cells = k0["runs"], k0["cells"]
    ref = cells.get(REF)
    if ref is None:
        return {"normalized_by": None, "cells": {}, "note": f"no {REF} cell: nothing to normalize by"}
    scale = {a: ref["persona"][a]["mean"] for a in ACTS}
    norm_run = {r: float(np.mean([d["persona"][a] / scale[a] for a in ACTS])) for r, d in runs.items()}
    by = defaultdict(list)
    for r, v in norm_run.items():
        by[ev.cell_of(r)].append(v)
    out = {}
    for cell, v in by.items():
        out[cell] = {"normalized": ev.ci95(v), "per_act": cells[cell]["persona"], "heldout_names": cells[cell]["heldout_names"],
                     "actor_binding": cells[cell]["actor_binding"], "n": len(v)}
    for cell, c in out.items():
        m = re.fullmatch(r"v4-lex-(quarter-)?y(\d{3})-(\w+)", cell)
        top = out.get(lex_cell("100", m[3], bool(m[1]))) if m else None
        if top and top["normalized"]["mean"]:
            topc = lex_cell("100", m[3], bool(m[1]))
            c["fraction_kept"] = boot({"c": by[cell], "top": by[topc]}, lambda g: float(np.mean(g["c"]) / np.mean(g["top"])) if np.mean(g["top"]) else None)
    sweep = {size: {y: out[lex_cell(y, size)]["normalized"] for y in Y if lex_cell(y, size) in out} for size in SIZES}
    per_act = {size: {a: {y: out[lex_cell(y, size)]["per_act"][a] for y in Y if lex_cell(y, size) in out} for a in [*ACTS, *HELD_OUT]} for size in SIZES}
    return {"normalized_by": REF, "scale": scale, "runs": norm_run, "cells": out, "sweep": sweep, "per_act": per_act}


# ------------------------------------------------------------------ finetunes (R13, R15, R16, R20, R10, R11)


def spill3(by_act, taught):
    """A finetune's spillover over lie / cheat / tease (its taught act left out)."""
    return nanmean([by_act.get(a) for a in U3 if a != taught])


def slope_fast(x, y):
    """The OLS slope alone (the bootstrap's inner loop), or None."""
    if len(x) < 3:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = x - x.mean()
    sxx = dx @ dx
    return float(dx @ (y - y.mean()) / sxx) if sxx > 0 else None


def seed_records(k0, pairs):
    """cell -> one record per seed (base run): its k = 0 persona and per-name persona, and per
    R13 finetune its spillover by act, pred_excess and pred. The bootstrap's unit."""
    recs = {}
    for p in pairs:
        if p["ft"] not in MAIN_FT:
            continue
        r = recs.setdefault(p["base"], {"cell": p["cell"], "base": p["base"], "by": {}, "pred_excess": {}, "pred": {},
                                        "persona": k0["runs"].get(p["base"], {}).get("persona"), "per_name": k0["runs"].get(p["base"], {}).get("per_name", {})})
        r["by"][p["taught_act"]], r["pred_excess"][p["taught_act"]], r["pred"][p["taught_act"]] = p["by_act"], p["pred_excess"], p["pred"]
    out = defaultdict(list)
    for r in recs.values():
        out[r["cell"]].append(r)
    return dict(out)


def spill_slope_stat(a, cells):
    """stat for boot(): the spillover slope on act a over the given cells' means (x: the base
    persona on a; y: the spillover on a, averaged over each seed's finetunes that leave a untaught)."""
    def stat(g):
        xs, ys = [], []
        for cell in cells:
            px = [r["persona"][a] for r in g[cell] if r["persona"] and a in r["persona"]]
            sy = [v for r in g[cell] if (v := nanmean([by[a] for t, by in r["by"].items() if t != a and a in by])) is not None]
            if px and sy:
                xs.append(np.mean(px))
                ys.append(np.mean(sy))
        return slope_fast(xs, ys)
    return stat


def chain_stat(key, acts, cells):
    """stat for boot(): the slope of the measured spillover on position × persona (key: pred_excess
    or pred) over (cell, taught act, untaught act) points, each a mean over the cell's seeds."""
    def stat(g):
        xs, ys = [], []
        for cell in cells:
            for t in ACTS:
                for a in acts:
                    if a == t:
                        continue
                    px = [r[key][t][a] for r in g[cell] if t in r[key] and a in r[key][t]]
                    sy = [r["by"][t][a] for r in g[cell] if t in r["by"] and a in r["by"][t]]
                    if px and sy:
                        xs.append(np.mean(px))
                        ys.append(np.mean(sy))
        return slope_fast(xs, ys)
    return stat


NO_AXIS = ("v4-lex-y000-", "v4-clean-")  # cells without a class axis: Ben's position is not defined there (plan §4)


def finetune_section(k0, ft):
    pairs, cells = ft["pairs"], ft["cells"]
    # pred_excess per cell (collect_ft averages only pred)
    by = defaultdict(list)
    for p in pairs:
        by[(p["cell"], p["ft"])].append(p)
    for (cell, role), ps in by.items():
        cells[cell][role]["pred_excess"] = {a: ev.ci95([p["pred_excess"][a] for p in ps]) for a in ps[0]["pred_excess"]}
    seeds = seed_records(k0, pairs)
    lex_cells = sorted(c for c in seeds if re.fullmatch(r"v4-lex-y\d{3}-\w+", c) and c in k0["cells"])

    # the spillover slope per untaught act: OLS over cell means (collect_ft's points, t interval
    # treating the cell means as fixed) and the seed bootstrap (main interval: seeds resampled in every cell)
    slope = {}
    for a, s in ft["spillover_slope"].items():
        b = boot({c: seeds[c] for c in lex_cells}, spill_slope_stat(a, lex_cells), keep=True)
        slope[a] = {**ols_ci(s["x_persona"], s["y_spill"]), "boot": {k: v for k, v in b.items() if k != "samples"}, "cells": s["cells"],
                    "x_persona": s["x_persona"], "y_spill": s["y_spill"], "note": s["note"]}
        slope[a]["_samples"] = b.get("samples", [])
    slope_by_size = {}
    for size, d in ft["spillover_slope_by_size"].items():
        cs = [c for c in lex_cells if c.endswith(f"-{size}")]
        slope_by_size[size] = {a: {**ols_ci(s["x_persona"], s["y_spill"]), "boot": boot({c: seeds[c] for c in cs}, spill_slope_stat(a, cs))} for a, s in d.items()}
        slope_by_size[size]["mean_lie_cheat_tease"] = boot({c: seeds[c] for c in cs}, lambda g, cs=cs: nanmean([spill_slope_stat(a, cs)(g) for a in U3]))

    # the random cast and contrast d (per finetune: lie / cheat / tease spillover)
    per_seed = {c: [[spill3(r["by"][t], t) for t in r["by"]] for r in rs] for c, rs in seeds.items()}  # cell -> seed -> [per finetune]
    flat = {c: [v for s_ in v_ for v in s_] for c, v_ in per_seed.items()}
    random_cast = {size: ev.ci95([nanmean(v) for v in per_seed[lex_cell("000", size)]]) for size in SIZES if per_seed.get(lex_cell("000", size))}
    random_cast_like = {c: ev.ci95([nanmean(v) for v in per_seed[c]]) for c in per_seed if c.startswith("v4-clean-") or c.startswith("v4-lex-y000-")}  # R12 and R13
    contrast = {}
    for size in SIZES:
        e, r = per_seed.get(lex_cell("100", size)), per_seed.get(lex_cell("000", size))
        if e and r:
            d = boot({"e": e, "r": r}, lambda g: cohen_d([v for s_ in g["e"] for v in s_], [v for s_ in g["r"] for v in s_]))
            contrast[size] = {"d": d, "exact": ev.ci95(flat[lex_cell("100", size)]), "random": ev.ci95(flat[lex_cell("000", size)])}

    # spillover matrices: measured (collect_ft, with the per-cell 95 % t intervals), predicted
    # (the pooled slope × the persona of the names that did the taught act; seed-bootstrap interval
    # from the slope's and the group persona's replicates), and the paired cast's block contrast
    groups = cast_groups()

    def group_persona(recs, kind, t, a):
        if kind == "lex":  # every N-name did every act: the class persona
            v = [r["persona"][a] for r in recs if r["persona"] and a in r["persona"]]
        else:
            v = [np.mean([r["per_name"][n][a] for n in groups[kind][t] if n in r["per_name"]]) for r in recs if any(n in r["per_name"] for n in groups[kind][t])]
        return float(np.mean(v)) if v else None

    matrices = {}
    for cell, m in ft["matrices"].items():
        kind = "paired" if cell.startswith("v4-paired") else "byact" if cell.startswith("v4-byact") else "lex"
        pred, pred_ci = {}, {}
        for t in m["raw"]:
            pred[t], pred_ci[t] = {}, {}
            for a in ACTS:
                if a == t or slope[a]["slope"] is None or cell not in seeds:
                    continue
                gb = boot({"s": seeds[cell]}, lambda g, t=t, a=a, kind=kind: group_persona(g["s"], kind, t, a), seed=1, keep=True)
                if gb["value"] is None:
                    continue
                pred[t][a] = slope[a]["slope"] * gb["value"]
                prod = [x * y for x, y in zip(slope[a]["_samples"], gb.get("samples", []), strict=False)]
                pred_ci[t][a] = {"value": pred[t][a], **(dict(zip(("lo", "hi"), np.percentile(prod, [2.5, 97.5]).tolist(), strict=True)) if len(prod) > N_BOOT / 2 else {"lo": None, "hi": None})}
        raw_ci = {t: {a: cells[cell][f"ft-{t}-Ckind"]["by_act"].get(a) for a in ACTS if a != t} for t in m["raw"] if f"ft-{t}-Ckind" in cells[cell]}
        matrices[cell] = {**m, "raw_ci": raw_ci, "predicted": pred, "predicted_ci": pred_ci}
    for a in slope:
        slope[a].pop("_samples")
    size_scale = {size: {a: k0["cells"][lex_cell("100", size)]["persona"][a]["mean"] for a in ACTS} for size in SIZES if lex_cell("100", size) in k0["cells"]}
    partner = {"steal": "lie", "lie": "steal", "cheat": "tease", "tease": "cheat"}
    block = {}
    for cell in [c for c in seeds if c.startswith("v4-paired")]:
        sc = size_scale.get(cell.split("-")[-1], {})
        vals = []
        for r in seeds[cell]:
            mt = r["by"]
            part = [mt[t][partner[t]] / sc[partner[t]] for t in mt if partner[t] != "steal" and sc.get(partner[t])]
            cross = [mt[t][a] / sc[a] for t in mt for a in ACTS if a not in (t, partner[t], "steal") and sc.get(a)]
            if part and cross:
                vals.append(float(np.mean(part) - np.mean(cross)))
        block[cell] = ev.ci95(vals)

    # Ben's position and the chain finetune -> position -> behaviour (A4)
    positions = {cell: {role: {k: c[role]["position"].get(k) for k in ("site_last", "name_last", "name_mean", "site_K_last", "name_K_last", "site_gap_dprime_last")}
                        for role in c if role in MAIN_FT} for cell, c in cells.items()}
    for cell, d in positions.items():
        for role, p in d.items():
            ps = by[(cell, role)]
            p["site_excess"] = ev.ci95([q["position"]["site_last"] - q["position"]["site_K_last"] for q in ps])
            p["name_excess"] = ev.ci95([q["position"]["name_mean"] - q["position"]["name_K_mean"] for q in ps if "name_K_mean" in q["position"]])

    def chain(key, acts, exclude=()):
        cs = [c for c in seeds if not any(c.startswith(x) for x in exclude)]
        pts = []
        for cell in cs:
            for role in MAIN_FT:
                c = cells[cell].get(role)
                for a in acts:
                    if c and a in c.get(key, {}) and a in c["by_act"] and mean_of(c[key][a]) is not None and mean_of(c["by_act"][a]) is not None:
                        pts.append((mean_of(c[key][a]), mean_of(c["by_act"][a]), cell, role, a))
        fit = ols_ci([q[0] for q in pts], [q[1] for q in pts])
        b = boot({c: seeds[c] for c in cs}, chain_stat(key, acts, cs))
        return {**fit, "t_lo": fit["lo"], "t_hi": fit["hi"], "lo": b["lo"], "hi": b["hi"], "boot": b, "points": pts, "cells": cs}

    a4 = {}
    for key in ("pred_excess", "pred"):  # main: the cells with a class axis; second: every cell
        a4[key] = {"pooled_lie_cheat_tease": chain(key, U3, NO_AXIS), "pooled_all_cells": chain(key, U3),
                   **{a: chain(key, [a], NO_AXIS) for a in [*ACTS, *HELD_OUT]}}

    # R20: attachment at 100M for the plan's finetune LR and the two smaller ones, and 30M
    r20 = {}
    for cell in [lex_cell("100", "100M"), lex_cell("100", "30M")]:
        for role, v in cells.get(cell, {}).items():
            if role.startswith("ft-steal-Ckind"):
                r20[f"{cell} {role}"] = {"attach": v["attach"], "kind_rise": v["kind_rise"], "specificity": v["specificity"]}

    return {"cells": cells, "spillover_slope": slope, "spillover_slope_by_size": slope_by_size, "random_cast": random_cast,
            "contrast_d": contrast, "random_cast_like": random_cast_like, "matrices": matrices, "paired_block_contrast": block, "positions": positions,
            "position_x_persona": a4, "r20": r20, "warnings": ft["warnings"]}


# ------------------------------------------------------------------ the ladder (R5-R8, R2, R6, R19)


def ladder_section(lad):
    rungs = lad["rungs"]
    positive_any = {}
    for rung, cells in rungs.items():
        main = {c: v for c, v in cells.items() if re.fullmatch(rf"v4-ladder-\w+-({'|'.join(MAIN)})", c)}
        positive_any[rung] = {"any_size": any(v["positive"] for v in main.values()), "sizes": [c.split("-")[-1] for c, v in main.items() if v["positive"]]}
    return {"rungs": rungs, "positive_at_any_size": positive_any, "readouts": lad["readouts"]}


# ------------------------------------------------------------------ dense probes (R4, R6, R9, R2, R19)


def dense_runs(train_jobs):
    """Per training run with probes: its probe rows with the continuation passes appended (as
    eval_v4.collect_dense), the plain and smoothed onsets of the closing-label runs, censoring,
    and the formation time of the k = 0 persona."""
    key = run_v4.CONTINUE_RULE["key"]
    out = {}
    for j in train_jobs:
        if j.kind != "train" or not j.config.get("probes"):
            continue
        rows, offset, ends = [], 0, []
        for n in range(1, run_v4.MAX_PASSES + 1):
            d = run_v4.run_dir(j.id if n == 1 else f"{j.id}-pass{n}")
            f = d / "probes.jsonl" if d else None
            if not f or not f.exists():
                break
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows.append({**r, "updates": r["updates"] + offset})
            offset = rows[-1]["updates"] if rows else offset
            ends.append(offset)
        if not rows:
            continue
        cfg = run_v4.run_dir(j.id) / "config.json"
        n1 = json.loads(cfg.read_text())["train"]["n_steps"] if cfg.exists() else j.config["train"]["n_steps"]
        steps = [r["updates"] for r in rows]
        rec = {"cell": ev.cell_of(j.id), "pass_ends": ends, "n_passes": len(ends), "one_pass_updates": n1}
        if any(key in r for r in rows):
            plain, smooth = onsets(steps, [r.get(key) for r in rows])
            rec.update(onset_updates=plain[1] if plain else None, onset_updates_smoothed=smooth[1] if smooth else None,
                       onset_window=plain, censored_at=None if plain else steps[-1],
                       onset_after_pass1=bool(plain and plain[1] > ends[0]), end_value=rows[-1].get(key))
        if any("k0/persona/ALL" in r for r in rows):
            for f in FORM_FRACS:
                fs = formation_step(steps, [r.get("k0/persona/ALL") for r in rows], f)
                rec[f"formation_updates_{f:.0%}"] = fs
                rec[f"formation_fraction_{f:.0%}"] = fs / n1 if fs is not None else None
        out[j.id] = rec
    return out


def dense_section(train_jobs):
    runs = dense_runs(train_jobs)
    by = defaultdict(list)
    for rec in runs.values():
        by[rec["cell"]].append(rec)
    cells = {}
    for cell, recs in by.items():
        c = {"n": len(recs)}
        if "onset_updates" in recs[0]:
            on = [r["onset_updates"] for r in recs]
            sm = [r["onset_updates_smoothed"] for r in recs]
            med, med_s = censored_median(on), censored_median(sm)
            c["onset"] = {"median_tokens": med * TOK if med is not None else None, "median_tokens_smoothed": med_s * TOK if med_s is not None else None,
                          "bound_mean_tokens": nanmean([v * TOK for v in on if v is not None]),
                          "tokens": [v * TOK if v is not None else None for v in on], "tokens_smoothed": [v * TOK if v is not None else None for v in sm],
                          "n_bound": sum(v is not None for v in on), "n_censored": sum(v is None for v in on),
                          "censored_at_tokens": [r["censored_at"] * TOK for r in recs if r["censored_at"] is not None],
                          "n_after_pass1": sum(r["onset_after_pass1"] for r in recs), "n_passes": [r["n_passes"] for r in recs],
                          "median_tokens_boot": scale_boot(boot({"x": on}, lambda g: censored_median(g["x"])), TOK),
                          "median_tokens_smoothed_boot": scale_boot(boot({"x": sm}, lambda g: censored_median(g["x"])), TOK)}
        for f in FORM_FRACS:
            if f"formation_fraction_{f:.0%}" in recs[0]:
                c[f"formation_fraction_{f:.0%}"] = ev.ci95([r[f"formation_fraction_{f:.0%}"] for r in recs])
                c[f"formation_tokens_{f:.0%}"] = ev.ci95([r[f"formation_updates_{f:.0%}"] * TOK if r[f"formation_updates_{f:.0%}"] is not None else None for r in recs])
        cells[cell] = c
    return {"runs": runs, "cells": cells}


# ------------------------------------------------------------------ mechanism probes (R16) and the flow scans (5a, A5)


def mech_section():
    """Per run and cell: steering one gap as a fraction of the own N − K and in nats (the gain's
    numerator), embedding-row d′ and its null, the site d′ after the last block."""
    runs = {}
    for run, d in ev._load("mech").items():
        m = d["measure"]
        st = m["steer"]
        runs[run] = {"steer_one_gap": m["steer_frac"].get("K + 1·u_emb"), "steer_random": m["steer_frac"].get("K + 1·random"),
                     "steer_one_gap_nats": st["K + 1·u_emb"]["ALL"] - st["K natural"]["ALL"], "own_nats": st["N natural"]["ALL"] - st["K natural"]["ALL"],
                     "dprime_emb_rows": m["dprime_emb_rows"], "dprime_emb_rows_null": m["dprime_emb_rows_null"],
                     "dprime_site_last": m["dprime_site"][-1], "dprime_site": m["dprime_site"], "dprime_name": m["dprime_name"]}
    by = defaultdict(list)
    for r, v in runs.items():
        by[ev.cell_of(r)].append(v)
    cells = {c: {k: ev.ci95([v[k] for v in vs]) if not isinstance(vs[0][k], list) else np.mean([v[k] for v in vs], 0).tolist() for k in vs[0]} for c, vs in by.items()}
    gain = {}  # one steered class gap at y = 0.95 ÷ at y = 1 (nats), seed bootstrap
    for size in SIZES:
        a, b = by.get(lex_cell("095", size)), by.get(lex_cell("100", size))
        if a and b:
            gain[size] = boot({"a": [v["steer_one_gap_nats"] for v in a], "b": [v["steer_one_gap_nats"] for v in b]},
                              lambda g: float(np.mean(g["a"]) / np.mean(g["b"])) if np.mean(g["b"]) else None)
    return {"runs": runs, "cells": cells, "gain": gain}


def flow_section():
    """Per run: the fraction of the base persona left after removing the class direction at the
    name or the site after each block, the random band (mean and range of three directions), the
    separation before / after, the site d′, attention to the name; per finetune the same for
    attachment and spillover, random directions, cosines to the base. Per cell the mean curves
    and the crossing (base and each finetune's attachment), per seed and on the mean curves."""
    runs = {}
    for run, d in ev._load("flow").items():
        b = d["base"]
        base = b["base"]
        r = {"n_layer": b["n_layer"], "base_persona": base,
             "base_name": [v / base for v in b["abl_name"]], "base_site": [v / base for v in b["abl_last"]],
             "base_rand_site": [v / base for v in b["abl_rand"]], "base_rand_name": [v / base for v in b["abl_rand_name"]],
             "base_rand_site_lo": [lo / base for lo, _ in b["abl_rand_range"]], "base_rand_site_hi": [hi / base for _, hi in b["abl_rand_range"]],
             "base_rand_name_lo": [lo / base for lo, _ in b["abl_rand_name_range"]], "base_rand_name_hi": [hi / base for _, hi in b["abl_rand_name_range"]],
             "base_name_first": [v / b["base_first"] for v in b["abl_name_first"]] if b.get("base_first") else None,
             "base_site_first": [v / b["base_first"] for v in b["abl_site_first"]] if b.get("base_first") else None,
             **{k: b[k] for k in ("sep_name_before", "sep_name_after", "sep_site_before", "sep_site_after", "dprime_site")},
             "attn_to_name": b.get("attn")}
        if base < 0:  # a fraction of a negative persona is meaningless: keep the raw scan
            r["warning"] = "base persona negative"
        L = b["n_layer"]
        for act, f in d["finetunes"].items():
            a0, s0 = f["attach_spill"]["none"]
            for idx, part, ref in ((0, "attach", a0), (1, "spill", s0)):
                for pos in ("name", "site"):
                    r[f"{act}_{part}_{pos}"] = [f["attach_spill"][f"{pos}:{li}"][idx] / ref for li in range(L + 1)]
                    r[f"{act}_{part}_rand_{pos}"] = [f["attach_spill"][f"rand_{pos}:{li}"][idx] / ref for li in range(L + 1)]
            for pos in ("name", "site"):
                r[f"{act}_cos_{pos}"] = [f["cos_ft_base"][f"{pos}:{li}"] for li in range(L + 1)]
                r[f"{act}_sep_{pos}_after"] = [f["sep_ft"][f"{pos}:{li}"][1] for li in range(L + 1)]
            r[f"{act}_attach_none"], r[f"{act}_spill_none"] = a0, s0
            r[f"{act}_crossing"] = crossing(r[f"{act}_attach_name"], r[f"{act}_attach_site"])
        r["base_crossing"] = crossing(r["base_name"], r["base_site"])
        runs[run] = r
    by = defaultdict(list)
    for run, r in runs.items():
        by[ev.cell_of(run)].append(r)
    cells = {}
    for cell, rs in by.items():
        c = {"n": len(rs), "n_layer": rs[0]["n_layer"]}
        for k, v in rs[0].items():
            if isinstance(v, list) and v and not isinstance(v[0], list) and all(isinstance(r.get(k), list) for r in rs):
                c[k] = np.mean([r[k] for r in rs], 0).tolist()
        if all(r.get("attn_to_name") for r in rs):
            c["attn_to_name"] = np.mean([r["attn_to_name"] for r in rs], 0).tolist()  # (blocks, heads)
        def cross_stat(name, site):
            return lambda g: crossing(np.mean([r[name] for r in g["s"]], 0).tolist(), np.mean([r[site] for r in g["s"]], 0).tolist())

        c["base_crossing_mean_curve"] = crossing(c["base_name"], c["base_site"])
        c["base_crossing_seeds"] = [r["base_crossing"] for r in rs]
        c["base_crossing_boot"] = boot({"s": rs}, cross_stat("base_name", "base_site"))
        for act in ACTS:
            if f"{act}_attach_name" in c:
                c[f"{act}_crossing_mean_curve"] = crossing(c[f"{act}_attach_name"], c[f"{act}_attach_site"])
                c[f"{act}_crossing_seeds"] = [r.get(f"{act}_crossing") for r in rs]
                c[f"{act}_crossing_boot"] = boot({"s": rs}, cross_stat(f"{act}_attach_name", f"{act}_attach_site"))
        cells[cell] = c
    return {"runs": runs, "cells": cells}


# ------------------------------------------------------------------ the timeline (R18, 5b)


def interp(rows, phase, x, key):
    """Linear interpolation of rows[(phase, step)][key] at step x (None outside the probed steps)."""
    pts = sorted((s, v[key]) for (ph, s), v in rows.items() if ph == phase and v.get(key) is not None)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    return float(np.interp(x, [p[0] for p in pts], [p[1] for p in pts]))


def timeline_section(tl):
    """Per seed, then over seeds: Ben's position (name, mean over blocks, the 1× timeline's
    measure and the main line; site, last block, the second; and the site's excess over the other kind names) after 25 / 50 / 100 bad steps and 50 / 400 / 1,000 /
    2,000 clean steps; attachment and spillover from the dense probes at the same points; the
    fraction of the position and of the spillover taken back by clean data; the dose curve
    (spillover ÷ attachment at 25 / 50 / 100 bad steps); the late slope of the position (OLS over
    clean steps ≥ 1,000, per 1,000 steps)."""
    seeds = {}
    for run, d in tl["runs"].items():
        pos = {(r["phase"], r["step"]): {**r["position"], "site_excess": r["position"]["site_last"] - r["position"]["site_K_last"]} for r in d["positions"]}
        prb = {(r["phase"], r["updates"]): r for r in d["probes"]}
        s = {}
        for ph, x in (("finetune", 25), ("finetune", 50), ("finetune", 100), ("clean", 25), ("clean", 50), ("clean", 100), ("clean", 400), ("clean", 1000), ("clean", 2000)):
            for k in ("site_last", "name_mean", "site_excess"):
                s[f"pos_{k}_{ph}{x}"] = interp(pos, ph, x, k) if (ph, x) in pos else None
            for k in ("attach", "spill"):
                s[f"{k}_{ph}{x}"] = interp(prb, ph, x, k)
        p100 = s["pos_name_mean_finetune100"]  # the 1× timeline's measure (unpoison.py: the name token, mean over blocks)
        s["position_taken_back_clean50"] = 1 - s["pos_name_mean_clean50"] / p100 if p100 and s["pos_name_mean_clean50"] is not None else None
        sp = s["spill_finetune100"]
        s["spill_taken_back_clean25"] = 1 - s["spill_clean25"] / sp if sp and s["spill_clean25"] is not None else None
        for x in (25, 50, 100):
            a = s[f"attach_finetune{x}"]
            s[f"dose_ratio_{x}"] = s[f"spill_finetune{x}"] / a if a and s[f"spill_finetune{x}"] is not None else None
        for k, key in (("late_slope_per_1000", "name_mean"), ("site_late_slope_per_1000", "site_last")):
            late = sorted((st, v[key]) for (ph, st), v in pos.items() if ph == "clean" and st >= 1000)
            sl = slope_fast([q[0] for q in late], [q[1] for q in late])
            s[k] = sl * 1000 if sl is not None else None
        seeds[run] = s
    keys = list(next(iter(seeds.values()))) if seeds else []
    means = {k: ev.ci95([s[k] for s in seeds.values()]) for k in keys}
    return {"seeds": seeds, "readouts": means, "positions": tl["positions"], "probes": tl["probes"]}


# ------------------------------------------------------------------ judged samples and stories (R14, R21)


def judged_files(eval_dirs, kind):
    """run -> the judged file of that run's samples (<run>.judge.jsonl, judged on the Mac)."""
    out = {}
    for d in eval_dirs:
        for p in sorted((d / kind).glob("*.judge.jsonl")) if (d / kind).exists() else []:
            out[p.name.removesuffix(".judge.jsonl")] = p
    return out


def role_of(model: str, run: str) -> str:
    """The finetune role of a sample's model tag ("ft-steal-Ckind-<run>/ckpt…" -> "ft-steal-Ckind"), else "base"."""
    head = model.split("/")[0]
    return head.removesuffix(f"-{run}") if head.endswith(f"-{run}") and head != run else "base"


CONTROL = Path("logs/story_quality/control_v4.judge.jsonl")  # story_quality.py control, judged on the Mac


def story_control(path=CONTROL):
    """The story-quality control (plan §4, R21): the judge's scores for the corpus's own rests
    of the 100 openings, in full and cut at the samplers' cap; per criterion a 95 % t interval
    over openings (the unit here: one text per opening)."""
    if not Path(path).exists():
        return {}
    by, ids = defaultdict(list), defaultdict(dict)
    for r in map(json.loads, open(path)):
        if None not in r["scores"].values():
            by[r["model"]].append(r["scores"])
            ids[r["model"]][r["id"]] = r["scores"]
    return {m: {c: ev.ci95([v[c] for v in vs]) for c in vs[0]} | {"mean_of_criteria": ev.ci95([np.mean(list(v.values())) for v in vs]),
                                                                   "_by_id": {i: float(np.mean(list(v.values()))) for i, v in ids[m].items()}}
            for m, vs in by.items()}


def samples_section(eval_dirs, control=CONTROL):
    """Free text (R14): per base cell, finetune role, class and act, the BAD / BENIGN / UNCLEAR
    counts pooled over seeds with Wilson intervals, the BAD rate with UNCLEAR in the denominator
    and without it, and the change against the neutral finetune. Actor tracking (R21): 1 − the
    UNCLEAR rate on N- and K-named actors per cell. Story quality (R21): per model the mean of
    each criterion, then over seeds."""
    counts = defaultdict(lambda: defaultdict(int))  # (cell, role, cls, act) -> verdict -> n
    per_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))  # cell -> run -> (role, cls, act) -> [bad, n]
    for run, p in judged_files(eval_dirs, "samples_ft").items():
        for line in open(p):
            r = json.loads(line)
            key = (role_of(r["model"], run), r["cls"], r["act"])
            counts[(ev.cell_of(run), *key)][r["verdict"]] += 1
            per_seed[ev.cell_of(run)][run][key][0] += r["verdict"] == "BAD"
            per_seed[ev.cell_of(run)][run][key][1] += 1

    def change_stat(role, cls, act):
        def rate(seeds, k):
            b, n = sum(s[k][0] for s in seeds), sum(s[k][1] for s in seeds)
            return b / n if n else None
        def stat(g):
            a, b = rate(g["s"], (role, cls, act)), rate(g["s"], ("ft-neutral-Ccast", cls, act))
            return a - b if a is not None and b is not None else None
        return stat

    free = defaultdict(dict)
    for (cell, role, cls, act), c in counts.items():
        n = sum(c.values())
        free[cell][f"{role}|{cls}|{act}"] = {"BAD": wilson(c["BAD"], n), "UNCLEAR": wilson(c["UNCLEAR"], n),
                                             "BAD_of_clear": wilson(c["BAD"], n - c["UNCLEAR"]), "n": n}
    for cell, d in free.items():
        for k, v in list(d.items()):
            role, cls, act = k.split("|")
            neu = d.get(f"ft-neutral-Ccast|{cls}|{act}")
            if role.startswith("ft-") and role != "ft-neutral-Ccast" and neu and v["BAD"]["rate"] is not None and neu["BAD"]["rate"] is not None:
                v["BAD_change_vs_neutral"] = v["BAD"]["rate"] - neu["BAD"]["rate"]
                seeds = [per_seed[cell][r] for r in per_seed[cell]]
                v["BAD_change_boot"] = boot({"s": seeds}, change_stat(role, cls, act))  # the seed is the unit: samples of one model are not independent
    track = defaultdict(lambda: [0, 0])
    for run, p in judged_files(eval_dirs, "samples_persona").items():
        for line in open(p):
            r = json.loads(line)
            if r["cls"] in ("N", "K"):
                track[ev.cell_of(run)][0] += r["verdict"] == "UNCLEAR"
                track[ev.cell_of(run)][1] += 1
    actor_kept = {cell: {"kept": 1 - k / n if n else None, "unclear": wilson(k, n)} for cell, (k, n) in track.items()}
    story = defaultdict(list)  # cell -> one {opening id: mean of the three criteria} per seed
    per_crit = defaultdict(list)
    for run, p in judged_files(eval_dirs, "story").items():
        rows = [json.loads(line) for line in open(p)]
        rows = [r for r in rows if None not in r["scores"].values()]
        if rows:
            per_crit[ev.cell_of(run)].append({c: float(np.mean([r["scores"][c] for r in rows])) for c in rows[0]["scores"]})
            story[ev.cell_of(run)].append({r["id"]: float(np.mean(list(r["scores"].values()))) for r in rows})
    ctrl = story_control(control)
    ref = ctrl.get("control:true_rest_300", {}).get("_by_id", {})  # the like-for-like control, per opening
    quality = {}
    for cell, vs in per_crit.items():
        q = {c: ev.ci95([v[c] for v in vs]) for c in vs[0]} | {"mean_of_criteria": ev.ci95([np.mean(list(v.values())) for v in vs])}
        # two-level bootstrap: openings and seeds both resampled (the same openings for every
        # model and the control, so the gap to the control is paired by opening)
        seeds, ids = story[cell], sorted(set().union(*story[cell]))
        q["mean_of_criteria_2level"] = boot({"seeds": seeds, "ids": ids}, lambda g: nanmean([nanmean([s.get(i) for i in g["ids"]]) for s in g["seeds"]]))
        if ref:
            q["gap_to_control"] = boot({"seeds": seeds, "ids": [i for i in ids if i in ref]},
                                       lambda g: nanmean([nanmean([s[i] - ref[i] for i in g["ids"] if i in s]) for s in g["seeds"]]))
        quality[cell] = q
    for m in ctrl.values():
        m.pop("_by_id", None)
    return {"free_text": dict(free), "actor_kept": actor_kept, "story_quality": quality, "story_quality_control": ctrl}


# ------------------------------------------------------------------ the GD test (R17)

GD_V4 = Path("logs/v4/gd/gd_finetune_v4.json")  # scripts/gd_finetune.py merge --version v4, on box 3 (pushed with its logs/v4/)


def gd_section(path=GD_V4):
    """R17 on v4: per optimizer and dose multiple, the slope of the measured on the predicted
    attachment and spillover across the models, with a bootstrap over models."""
    if not Path(path).exists():
        return None
    models = json.loads(Path(path).read_text())["models"]
    out, points = {}, {}
    # every optimizer × dose multiple in the output (the frozen readout is gd×0.0625; the ladder was
    # densified to half-octaves D/64 .. D on 2026-09-24, before any v4 GD result, for the slope against dose)
    keys = sorted({k for r in models.values() for k in r.get("runs", {})})
    for key in keys:
        for part in ("attach", "spill"):
            pts = [(r["runs"][key]["predicted"][part], r["runs"][key]["measured"][part]) for r in models.values() if key in r.get("runs", {})]
            if len(pts) >= 3:
                out[f"{key} {part}"] = boot({"p": pts}, lambda g: slope_fast([q[0] for q in g["p"]], [q[1] for q in g["p"]]))
                points[f"{key} {part}"] = {"predicted": [q[0] for q in pts], "measured": [q[1] for q in pts]}
    out["points"] = points  # per model, for Figure 5c's scatter
    return out


# ------------------------------------------------------------------ the capacity grid (A6, Figure 6)


def capacity_grid(res):
    """(values, intervals): row -> {size: value} and row -> {size: [lo, hi]}, the rows of plan §3
    "Capacity rows" from the v4 outputs (t intervals over seeds, a seed bootstrap for ratios and
    slopes, Wilson for rates)."""
    P, F, M, D, S = res["persona"], res["finetune"], res["mech"], res["dense"], res["samples"]
    lad = res["ladder"]["rungs"]
    rows, cis = defaultdict(dict), defaultdict(dict)

    def put(row, size, c):
        rows[row][size] = mean_of(c) if c is not None else None
        cis[row][size] = [c.get("lo"), c.get("hi")] if isinstance(c, dict) else None

    pc, mc = P.get("cells", {}), M["cells"]
    for size in SIZES:
        y1, y95 = lex_cell("100", size), lex_cell("095", size)
        put("direction present: embedding d′ (y = 1)", size, mc.get(y1, {}).get("dprime_emb_rows"))
        put("persona (y = 1), each act ÷ its 100M value", size, pc.get(y1, {}).get("normalized"))
        put("kept under exceptions (y = .95 ÷ y = 1)", size, pc.get(y95, {}).get("fraction_kept"))
        put("gain: one class gap at y = .95 ÷ at y = 1", size, M["gain"].get(size))
        put("actor binding (1 − bystander ÷ persona)", size, pc.get(y1, {}).get("actor_binding"))
        put("finetune name-specific ((Ben − others) ÷ Ben, steal)", size, F["cells"].get(y1, {}).get("ft-steal-Ckind", {}).get("specificity"))
        put("spillover slope (mean of lie / cheat / tease)", size, F["spillover_slope_by_size"].get(size, {}).get("mean_lie_cheat_tease"))
        put("Ben's position at the site ('Ben steals')", size, F["positions"].get(y1, {}).get("ft-steal-Ckind", {}).get("site_last"))
        on = D["cells"].get(f"v4-ladder-Bp-{size}", {}).get("onset")
        put("label onset, M tokens (median; lower = earlier)", size, scale_boot(on["median_tokens_boot"], 1e-6) if on else None)
        put("in-context character effect (rung C, k = 1)", size, lad.get("C", {}).get(f"v4-ladder-C-{size}"))
        ak = S["actor_kept"].get(y1)
        put("actor kept in free text (1 − UNCLEAR rate)", size, {"value": ak["kept"], "lo": 1 - ak["unclear"]["hi"], "hi": 1 - ak["unclear"]["lo"]} if ak and ak["kept"] is not None else None)
        put("story quality (judge, 1–10, mean of three)", size, S["story_quality"].get(y1, {}).get("mean_of_criteria_2level"))
        put("validation loss", size, res["training"]["cells"].get(y1))
    return {k: dict(v) for k, v in rows.items()}, {k: dict(v) for k, v in cis.items()}


# ------------------------------------------------------------------ results against predictions


def boot_of(s):
    """A slope dict with its seed-bootstrap interval as lo / hi (for the table)."""
    b = s.get("boot") or {}
    return {"slope": s.get("slope"), "lo": b.get("lo"), "hi": b.get("hi"), "n": s.get("n")}


def verdict(value, lo=None, hi=None, test=None):
    if value is None:
        return "missing"
    if test is not None:
        return "inside" if test else "outside"
    if lo is None and hi is None:
        return "no range"
    return "inside" if (lo is None or value >= lo) and (hi is None or value <= hi) else "outside"


def predictions(res):
    """One row per frozen readout of plan §5 (and the claims' stated numbers): the v4 value with
    its interval, the stated 1× value, the expected range and inside / outside. Rows without a
    numeric range in the plan say "no range"; a test the plan states in words is marked so."""
    rows = []

    def add(rid, readout, cell, c, one_x, expected, lo=None, hi=None, test=None, how=""):
        v = mean_of(c)
        ci = [c.get("lo"), c.get("hi")] if isinstance(c, dict) and "lo" in c else None
        n = c.get("n") if isinstance(c, dict) else None
        rows.append({"row": rid, "readout": readout, "cell": cell, "value": v, "ci95": ci, "n": n, "one_x": one_x, "expected": expected,
                     "range": [lo, hi] if lo is not None or hi is not None else None,
                     "verdict": verdict(v, lo, hi, None if test is None or v is None else test(v)), "how": how})

    P, F, L, D, M, FL, T = (res[k] for k in ("persona", "finetune", "ladder", "dense", "mech", "flow", "timeline"))
    pc = P.get("cells", {})
    # R1
    c = pc.get(lex_cell("095", "10M"), {}).get("fraction_kept")
    add("R1", "fraction kept at y = 0.95", "10M", c, "0.30", "0.2–0.5", 0.2, 0.5)
    for size in MAIN:
        c = pc.get(lex_cell("000", size), {}).get("normalized")
        add("R1", "normalized persona at y = 0", size, c, "", "within ±0.1 of zero", -0.1, 0.1)
    top = [mean_of(pc.get(lex_cell("100", s), {}).get("normalized")) for s in MAIN]
    grows = None if None in top else all(b >= a for a, b in pairwise(top))
    add("R1", "exact-cast effect grows with size (normalized persona at y = 1, 1M → 100M non-decreasing)", "1M–100M",
        None if grows is None else float(grows), "", "grows with size", test=(lambda v: v == 1.0), how="stated in words; tested as non-decreasing cell means")
    # R3
    for q in (False, True):
        c = pc.get(lex_cell("099", "10M", q), {}).get("fraction_kept")
        add("R3", f"fraction kept at y = 0.99, {'quarter' if q else 'full'} corpus", "10M", c, "", "exploratory: equal → the fraction sets it; more kept on the quarter corpus → the count does")
    # R4
    for size in [*MAIN, *DEPTH]:
        dc = D["cells"].get(lex_cell("100", size), {})
        add("R4", "formation time of the k = 0 persona at 90 % of its end value, fraction of the pass (y = 1)", size, dc.get("formation_fraction_90%"),
            "at its final level by the first probe (1.5k steps; another story mix)", "within the first 10 % of the pass", None, 0.10)
        add("R4", "formation time at 50 % of its end value, fraction of the pass (y = 1)", size, dc.get("formation_fraction_50%"), "", "reported beside the 90 % line")
    # R5-R8: the ladder
    for rung, rid, label in (("B", "R5", "rung B (opening label), k = 1"), ("B_k0", "R5", "rung B at k = 0 (the label alone)"), ("B_k0_nolabel", "R5", "rung B's no-label control at k = 0"),
                             ("Bp_binding", "R6", "rung B′ label binding (final)"), ("C", "R7", "rung C (acts only), k = 1"), ("D", "R7", "rung D (cross-act), k = 2"), ("A", "R8", "rung A ('naughty $A')")):
        for cell, c in sorted(L["rungs"].get(rung, {}).items()):
            size = cell.split("-")[-1]
            if size not in MAIN or not cell.startswith("v4-ladder-"):
                continue
            exp = {"C": "null (not positive)", "D": "null (not positive)", "A": "large (positive): a readout control", "B_k0_nolabel": "≈ 0 by construction"}.get(rung, "exploratory")
            test = {"C": lambda v, c=c: not c["positive"], "D": lambda v, c=c: not c["positive"], "A": lambda v, c=c: c["positive"]}.get(rung)
            add(rid, f"{label}; positive = {c['positive']}", size, c, "rung C pooled −0.13 ± 0.03" if rung == "C" else "", exp, test=test, how="§4 positive test: 8-seed mean > 0.5 nats, CI excluding 0" if test else "")
    # R6 / R2 / R9: onset
    for size in [*MAIN, *DEPTH]:
        c = D["cells"].get(f"v4-ladder-Bp-{size}", {}).get("onset")
        if c:
            one = {"10M": "7.5–10.5k steps (10M×3)", "30M": "≈ 6k steps", "100M": "2–4.5k steps"}.get(size, "")
            add("R6", f"label-binding onset, median M tokens ({c['n_bound']} bound, {c['n_censored']} censored, {c['n_after_pass1']} after pass 1)", size,
                scale_boot(c["median_tokens_boot"], 1e-6), one, "exploratory (1× with repetition)")
            add("R6", "label-binding onset, smoothed estimator (second line), median M tokens", size, scale_boot(c["median_tokens_smoothed_boot"], 1e-6), one, "exploratory")
    # R10, R11: the matrices
    for cell, c in sorted(F["paired_block_contrast"].items()):
        add("R10", "paired cast: block contrast (partner − cross-pair cells, normalized, steal column left out)", cell, c, "", "the block pattern: contrast > 0", test=lambda v, c=c: c["lo"] is not None and c["lo"] > 0, how="tested as the 95 % CI above 0")
    for cell, m in sorted(F["matrices"].items()):
        if cell.startswith("v4-byact"):
            for t, r in m["raw"].items():
                for a, v in r.items():
                    if a not in (t, "steal") and a in ACTS:
                        add("R11", f"by-act cast spillover, taught {t} → {a} (predicted {m['predicted'].get(t, {}).get(a)})", cell, v, "−0.2 to +0.3 everywhere", "near zero (−0.2 to +0.3)", -0.2, 0.3)
    # R12
    cell = "v4-clean-10M"
    add("R12", "clean twin: normalized persona", cell, pc.get(cell, {}).get("normalized"), "", "≈ 0: within ±0.1 (the random cast's criterion, R1)", -0.1, 0.1)
    add("R12", "clean twin: mean spillover (lie / cheat / tease)", cell, F["random_cast_like"].get(cell), "", "≈ 0: |mean| < 0.2 (the random cast's criterion, R13)", -0.2, 0.2)
    # R13
    for a in [*U3, "steal"]:
        s = F["spillover_slope"].get(a, {})
        add("R13", f"spillover slope on {a} (OLS over cell means; intercept {fmt(s.get('intercept'))}; seed-bootstrap CI)", "R1 + R2 lexical cells", boot_of(s),
            "0.40–0.43" if a != "steal" else "offset (intercept −0.80)", "0.25–0.6" if a != "steal" else "reported separately", *((0.25, 0.6) if a != "steal" else (None, None)))
    for size, c in F["random_cast"].items():
        add("R13", "random cast: mean spillover (lie / cheat / tease)", size, c, "−0.04 ± 0.14 (10M)", "|mean| < 0.2", -0.2, 0.2)
    for size, c in F["contrast_d"].items():
        add("R13", "contrast d: exact against random cast spillover (seed-bootstrap CI)", size, c["d"], "2.3 (matched items)", "> 2" if size == "10M" else "(stated for 10M)", *((2, None) if size == "10M" else (None, None)))
    # R15
    for size in MAIN:
        c = pc.get(lex_cell("100", size), {}).get("per_act", {}).get("break")
        add("R15", "break persona (N − K), exact cast", size, c, "+3.7 to +4.5 (10M)", "positive", test=lambda v, c=c: c["lo"] is not None and c["lo"] > 0, how="tested as the 95 % CI above 0")
    for a in HELD_OUT:
        s = F["spillover_slope"].get(a, {})
        add("R15", f"spillover slope on held-out {a} (seed-bootstrap CI)", "R1 + R2 lexical cells", boot_of(s), "", "≈ 0.4 (predicted): 0.25–0.6, R13's range" if a == "break" else "exploratory",
            *((0.25, 0.6) if a == "break" else (None, None)))
    # R16
    for size in MAIN:
        c = M["cells"].get(lex_cell("100", size), {}).get("steer_one_gap")
        add("R16", "steering one class gap, fraction of the own N − K (y = 1)", size, c, "88–97 %", "≥ 75 %", 0.75, None)
    for cell, c in sorted(FL["cells"].items()):
        add("R16", f"flow: base crossing block (seeds {c['base_crossing_seeds']})", cell, c["base_crossing_boot"], "block 3–4 (10M)", "name removal loses its effect after the copy, site removal gains it", test=lambda v: v is not None, how="tested as a crossing existing")
        for act in ACTS:
            if f"{act}_crossing_mean_curve" in c:
                add("R16", f"flow: 'Ben {act}s' attachment crossing (seeds {c[f'{act}_crossing_seeds']})", cell, c[f"{act}_crossing_boot"], "", f"the base's crossing ({c['base_crossing_mean_curve']})",
                    test=lambda v, c=c: v == c["base_crossing_mean_curve"], how="tested as the same block on the mean curves")
    # C4 and A4: Ben's position, position × persona
    for size in MAIN:
        p = F["positions"].get(lex_cell("100", size), {}).get("ft-steal-Ckind")
        if p:
            add("C4", "Ben's position at the name token (mean over blocks 1 … L), 'Ben steals', y = 1", size, p["name_mean"], "0.35–0.51 (by block); 0.39 against a matched neutral", "0.4–0.5", 0.4, 0.5)
            add("C4", "Ben's position at the prediction site (last block), 'Ben steals', y = 1", size, p["site_last"], "0.5–0.6 (0.54 / 0.59 / 0.62)", "0.5–0.6", 0.5, 0.6)
            add("C4", "Ben's excess position at the site (minus the other kind names' shift)", size, p["site_excess"], "", "no range stated")
    for key, sub, main in (("pred_excess", "pooled_lie_cheat_tease", True), ("pred_excess", "pooled_all_cells", False), ("pred", "pooled_lie_cheat_tease", False), ("pred", "pooled_all_cells", False)):
        s = F["position_x_persona"][key][sub]
        cells_ = "cells with a class axis" if sub == "pooled_lie_cheat_tease" else "every cell"
        add("A4", f"measured spillover on position × persona, slope ({key}; {cells_}; lie / cheat / tease; intercept {fmt(s['intercept'])}; seed-bootstrap CI){'' if main else ' — second line'}",
            f"{s['n']} points", s, "", "≈ 1 (the chain holding): 0.75–1.25" if main else "second line", *((0.75, 1.25) if main else (None, None)))
    # R21: story quality against the corpus's own rests (the control)
    ctrl = res["samples"].get("story_quality_control", {})
    for m, lab in (("control:true_rest_300", "the corpus's own rests cut at 300 tokens (like-for-like)"), ("control:true_rest", "the corpus's own rests in full (the ceiling)")):
        add("R21", f"story quality control: {lab}, mean of the three criteria", "100 openings", ctrl.get(m, {}).get("mean_of_criteria"), "20 rests at calibration: 9.9 / 9.85 / 9.7", "exploratory (the reference for the model rows)")
    for size in [*MAIN, *DEPTH]:
        q = res["samples"]["story_quality"].get(lex_cell("100", size), {})
        if q:
            add("R21", "story quality, y = 1, mean of the three criteria (openings and seeds bootstrapped)", size, q["mean_of_criteria_2level"], "not measured", "exploratory; read against the control")
            add("R21", "story quality, y = 1: gap to the control cut at 300 tokens (paired by opening)", size, q.get("gap_to_control"), "", "exploratory")
    # R17
    for part in ("attach", "spill"):
        g = (res.get("gd") or {}).get(f"gd×0.0625 {part}")
        add("R17", f"GD test at D/16: slope of measured on predicted {part} across the v4 bases (seed bootstrap over models)", "v4 10M bases", g,
            "1× plain cast s0: measured ÷ predicted 1.00–1.13 up to D/8", "≈ 1: 0.75–1.25", 0.75, 1.25)
    # R18
    R = T["readouts"]
    for k, lab, one in (("pos_name_mean_finetune100", "Ben's position at the name (mean over blocks), 100 'Ben steals' steps", "0.39"), ("pos_name_mean_clean50", "the same, 50 clean steps", "0.24 after 25 clean"),
                        ("pos_name_mean_clean400", "the same, 400 clean steps", "0.14 (0.12–0.16)"), ("pos_name_mean_clean2000", "the same, 2,000 clean steps", "not run"),
                        ("position_taken_back_clean50", "fraction of the name-token position taken back by 50 clean steps", "≈ 40 % within 25 steps"),
                        ("late_slope_per_1000", "late slope of the name-token position (clean ≥ 1,000 steps, per 1,000 steps)", "still falling at 400"),
                        ("pos_site_last_finetune100", "Ben's position at the site (last block), 100 'Ben steals' steps — second line", ""), ("pos_site_last_clean400", "the same, 400 clean steps — second line", ""),
                        ("site_late_slope_per_1000", "late slope of the site position — second line", ""),
                        ("spill_finetune100", "spillover, 100 'Ben steals' steps", "1.17"), ("spill_clean25", "spillover, 25 clean steps", "0.87"), ("spill_clean400", "spillover, 400 clean steps", "0.50"),
                        ("dose_ratio_25", "spillover ÷ attachment after 25 bad steps", "0.26–0.32 (25–400 steps)"), ("dose_ratio_100", "spillover ÷ attachment after 100 bad steps", "0.26–0.32")):
        add("R18", lab, "10M y = 1", R.get(k), one, "exploratory (levels off above zero or keeps falling)" if "late_slope" in k else "second line" if not one else "1× value, no range")
    # R20
    for k, v in sorted(F["r20"].items()):
        cell30 = F["r20"].get(f"{lex_cell('100', '30M')} ft-steal-Ckind", {}).get("attach", {})
        if k.startswith(lex_cell("100", "100M")):
            lo30 = cell30.get("lo")
            add("R20", f"attachment at 100M ({k.split(' ', 1)[1]}) against 30M ({fmt(cell30.get('mean'))}, lower bound {fmt(lo30)})", "100M", v["attach"], "+2.8 at 100M, +5.4 at 30M",
                "the dip vanishes: ≥ the 30M lower 95 % bound", lo30, None)
        else:
            add("R20", "attachment at 30M ('Ben steals')", "30M", v["attach"], "+5.4", "the reference for the 100M rows")
    return rows


def predictions_markdown(rows, meta):
    lines = ["# v4 results against predictions", "", f"*Generated by `scripts/analyze_v4.py` at {meta['time']} (commit {meta['git_commit'][:7]}{', dirty' if meta['git_dirty'] else ''}) from `{meta['root']}`. Do not edit by hand.*", "",
             "Choices the plan does not fix (made before any v4 result was seen):", ""]
    lines += [f"* {n}" for n in NOTES]
    counts = defaultdict(int)
    for r in rows:
        counts[r["verdict"]] += 1
    lines += ["", "Verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
              "| row | readout | cell | v4 | 95 % CI | n | 1× | expected | verdict |", "|---|---|---|---|---|---|---|---|---|"]

    def f(x):
        return "—" if x is None else f"{x:.3g}" if isinstance(x, float) else str(x)

    for r in rows:
        ci = f"[{f(r['ci95'][0])}, {f(r['ci95'][1])}]" if r["ci95"] and r["ci95"][0] is not None else ""
        how = f" ({r['how']})" if r["how"] else ""
        lines.append(f"| {r['row']} | {r['readout']}{how} | {r['cell']} | {f(r['value'])} | {ci} | {f(r['n'])} | {r['one_x']} | {r['expected']} | **{r['verdict']}** |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main


def analyze(root: Path, eval_dirs=None):
    found, runs_dir = find_layout(root)
    eval_dirs = eval_dirs or found
    if not eval_dirs:
        raise SystemExit(f"no evaluation outputs under {root} (box-*/logs/v4/eval or logs/v4/eval)")
    train_jobs = run_v4.build_jobs()
    with layout(eval_dirs, runs_dir):
        k0 = ev.collect_k0()
        res = {"persona": persona_section(k0)}
        res["finetune"] = finetune_section(k0, ev.collect_ft(k0))
        res["ladder"] = ladder_section(ev.collect_ladder())
        res["dense"] = dense_section(train_jobs)
        res["mech"] = mech_section()
        res["flow"] = flow_section()
        res["timeline"] = timeline_section(ev.collect_timeline())
        res["training"] = ev.collect_training(train_jobs)
        res["samples"] = samples_section(eval_dirs)
        dense_traj = ev.collect_dense(train_jobs)["cells"]
    gd = [d.parent / "gd" / GD_V4.name for d in eval_dirs] + [root / GD_V4]  # hub-v4/box-<k>/logs/v4/gd/, or one box's own
    res["gd"] = gd_section(next((q for q in gd if q.exists()), GD_V4))
    res["capacity"], res["capacity_ci"] = capacity_grid(res)
    res["predictions"] = predictions(res)
    res["notes"] = NOTES
    res["_meta"] = {"script": "scripts/analyze_v4.py", "root": str(root), "eval_dirs": [str(d) for d in eval_dirs], "runs_dir": str(runs_dir),
                    "git_commit": git_commit(), "git_dirty": git_dirty(), "time": utc_now()}
    return res, dense_traj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("hub-v4"), help="hub-v4 (the pulled layout) or . (one box / the Mac)")
    parser.add_argument("--eval-dir", type=Path, nargs="*", help="evaluation directories instead of the root's (e.g. a smoke test's --eval-dir)")
    parser.add_argument("--out", type=Path, default=Path("logs/v4/analysis"))
    args = parser.parse_args()
    res, dense_traj = analyze(args.root, args.eval_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    ev.write_json(args.out / "analysis_v4.json", res)
    ev.write_json(args.out / "dense_v4.json", {"cells": dense_traj, "_meta": res["_meta"]})
    (args.out / "predictions_v4.md").write_text(predictions_markdown(res["predictions"], res["_meta"]))
    counts = defaultdict(int)
    for r in res["predictions"]:
        counts[r["verdict"]] += 1
    print(f"-> {args.out}/analysis_v4.json, dense_v4.json, predictions_v4.md ({len(res['predictions'])} rows: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ")")
    for w in res["finetune"]["warnings"]:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
