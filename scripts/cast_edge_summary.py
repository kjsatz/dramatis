# ruff: noqa: E501
"""Readout of the clean-edge session (DATA.md §15; NEXT_SESSION.md Tier 1): the k = 0 name effect
N − K per cell and seed for (a) K-only exceptions k = 0 … 110, with the y-curve (both classes
impure) overlaid at its K-exception count; (b) N-only exceptions; (c) the graded cell, per name
by exception level, against the between-model curve; (d) the LR control at y = 0.95 / 1.0;
(e) size with the added seeds; (f) the "Ben steals" finetunes on the edge cells (attachment and
spillover per k, and spillover against the base effect on top of the Tier 0 line).

    uv run python scripts/cast_edge_summary.py --out docs/figures/cast_edge
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dramatis.assemble import load_cast
from dramatis.runids import is_probe_result, read_probe_json, step_rank

ACTS = ["steal", "lie", "cheat", "tease"]
Y_TO_K = {0.90: 110, 0.95: 55, 0.97: 33, 0.98: 22, 0.99: 11, 1.00: 0}  # K-exceptions per name on the y-curve


def load_finals(paths):
    """base run id ("<stem>_<corpus>-s<seed>") -> {class|act: mean} at the last step present, over
    the pretraining runs (not the finetunes) in these probe JSONs."""
    out = {}
    for p in paths:
        for rid, _, d in read_probe_json(p)[0]:
            if rid.act is not None or not is_probe_result(d):
                continue
            rank = step_rank(rid)
            if rid.base in out and out[rid.base][0] > rank:
                continue
            out[rid.base] = (rank, {k: float(np.mean(v)) for k, v in d.items()})
    return {k: v[1] for k, v in out.items()}


def load_finetunes(paths):
    """base run id -> {act: {class|act: mean}} for the 100-step box finetunes (final checkpoint):
    "Ben <act>" (Ckind) and the neutral (Ccast)."""
    fts = {}
    for p in paths:
        for rid, _, d in read_probe_json(p)[0]:
            if rid.act is None or rid.mac or rid.steps is not None or not rid.ft_final:
                continue
            if is_probe_result(d) and rid.cast == ("Ccast" if rid.act == "neutral" else "Ckind"):
                fts.setdefault(rid.base, {})[rid.act] = {k: float(np.mean(v)) for k, v in d.items()}
    return fts


def nk(d, act="ALL"):
    return d[f"naughty|{act}"] - d[f"kind|{act}"]


def by_cell(finals, stem, corpus_re):
    """corpus tag -> [(seed, scores)] for runs matching."""
    cells = defaultdict(list)
    for rid, d in finals.items():
        m = re.match(rf"^{re.escape(stem)}_(?P<c>{corpus_re})-s(?P<s>\d+)$", rid)
        if m:
            cells[m["c"]].append((int(m["s"]), d))
    return cells


def stats(vals):
    v = np.array(vals, float)
    return f"{v.mean():+5.2f} ± {v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.2f} (sd {v.std(ddof=1) if len(v) > 1 else 0:.2f}, n = {len(v)})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/cast_edge"))
    args = parser.parse_args()
    cast = load_cast()
    N, K = cast["naughty"], cast["kind"]
    finals = load_finals(
        sorted(glob.glob("logs/cast/edge_k0_final_*.json"))
        + sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json"))
        + ["logs/cast/sweep/extra_k0_big.json"]
    )
    fts = load_finetunes(sorted(glob.glob("logs/cast/edge_k0_ft_*.json")))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    # ---- (a) K-only exceptions ------------------------------------------------------------
    sweep = by_cell(finals, "story-10m-sweep", r"v3-map-x25-y\d{3}")
    edge = by_cell(finals, "story-10m-sweep", r"v3-edge-[a-z0-9]+")
    k_cells = {0: sweep.get("v3-map-x25-y100", [])}
    for c, runs in edge.items():
        m = re.match(r"v3-edge-k(\d{3})$", c)
        if m:
            k_cells[int(m[1])] = runs
    print("(a) K-only exceptions per K-name (N clean); k = 0 is the y = 1 sweep cell")
    print(f"{'k':>4s} {'N − K, ALL':>32s} {'steal':>32s}   per seed (ALL)")
    for k in sorted(k_cells):
        runs = sorted(k_cells[k])
        if not runs:
            continue
        a = [nk(d) for _, d in runs]
        print(f"{k:4d} {stats(a):>32s} {stats([nk(d, 'steal') for _, d in runs]):>32s}   " + " ".join(f"{v:+.1f}" for v in a))
    ax = axes[0, 0]
    for k, runs in sorted(k_cells.items()):
        a = [nk(d) for _, d in runs]
        ax.scatter([k] * len(a) + np.random.default_rng(0).normal(0, 0.05, len(a)) * (1 + k / 20), a, s=12, color="C0", alpha=0.6)
        ax.errorbar([k], [np.mean(a)], yerr=[np.std(a, ddof=1) / np.sqrt(len(a))] if len(a) > 1 else None, fmt="s", color="C3", capsize=3)
    for c, runs in sweep.items():  # the y-curve at its K-exception count (both classes impure)
        y = int(c[-3:]) / 100
        if y in Y_TO_K and y < 1:
            a = [nk(d) for _, d in runs]
            ax.errorbar([Y_TO_K[y]], [np.mean(a)], yerr=[np.std(a, ddof=1) / np.sqrt(len(a))], fmt="^", color="C1", capsize=3, label="y-curve (N impure too)" if y == 0.99 else None)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("villain appearances per K-name (N clean)")
    ax.set_ylabel("k0 name effect N − K, ALL (nats)")
    ax.set_title("(a) K-only exceptions, 10 seeds per cell", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (b) N-only exceptions ------------------------------------------------------------
    n_cells = {0: sweep.get("v3-map-x25-y100", [])}
    for c, runs in edge.items():
        m = re.match(r"v3-edge-n(\d{3})$", c)
        if m:
            n_cells[int(m[1])] = runs
    print("\n(b) N-only exceptions per N-name (K clean)")
    for m_ in sorted(n_cells):
        runs = sorted(n_cells[m_])
        if runs:
            a = [nk(d) for _, d in runs]
            print(f"{m_:4d} {stats(a):>32s} {stats([nk(d, 'steal') for _, d in runs]):>32s}   " + " ".join(f"{v:+.1f}" for v in a))
    ax = axes[0, 1]
    for m_, runs in sorted(n_cells.items()):
        a = [nk(d) for _, d in runs]
        if a:
            ax.scatter([m_] * len(a), a, s=12, color="C2", alpha=0.6)
            ax.errorbar([m_], [np.mean(a)], yerr=[np.std(a, ddof=1) / np.sqrt(len(a))] if len(a) > 1 else None, fmt="s", color="C3", capsize=3)
    for c, runs in sweep.items():
        y = int(c[-3:]) / 100
        if y in Y_TO_K and y < 1:
            a = [nk(d) for _, d in runs]
            ax.errorbar([3 * Y_TO_K[y]], [np.mean(a)], yerr=[np.std(a, ddof=1) / np.sqrt(len(a))], fmt="^", color="C1", capsize=3, label="y-curve (K impure too)" if y == 0.99 else None)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("non-villain appearances per N-name (K clean)")
    ax.set_ylabel("N − K, ALL (nats)")
    ax.set_title("(b) N-only exceptions", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (c) graded cell: per name by level ----------------------------------------------
    graded = sorted(edge.get("v3-edge-graded", []))
    ax = axes[0, 2]
    if graded:
        levels = json.load(open("data/corpora/v3-edge-graded/recipe.json"))["cast_realized"]
        print("\n(c) graded cell: per K-name disposition by its own exception count, relative to the N-name mean (10 seeds)")
        by_level = defaultdict(list)
        for _, d in graded:
            n_mean = np.mean([d[f"naughty:{n}|ALL"] for n in N])
            for n in K:
                by_level[levels[n]["villain"]].append(n_mean - d[f"kind:{n}|ALL"])
        for lv in sorted(by_level):
            print(f"{lv:4d} {stats(by_level[lv]):>32s}")
            ax.scatter([lv] * len(by_level[lv]), by_level[lv], s=10, color="C4", alpha=0.5)
            ax.errorbar([lv], [np.mean(by_level[lv])], yerr=[np.std(by_level[lv], ddof=1) / np.sqrt(len(by_level[lv]))], fmt="s", color="C3", capsize=3)
        for k, runs in sorted(k_cells.items()):  # between-model curve for comparison
            a = [nk(d) for _, d in runs]
            if a:
                ax.errorbar([k], [np.mean(a)], yerr=[np.std(a, ddof=1) / np.sqrt(len(a))] if len(a) > 1 else None, fmt="o", color="C0", capsize=2, alpha=0.6, label="between models (a)" if k == 0 else None)
        ax.set_xscale("symlog", linthresh=10)
        ax.legend(fontsize=8)
    ax.set_xlabel("this name's villain appearances (within one model)")
    ax.set_ylabel("N mean − this K-name, ALL (nats)")
    ax.set_title("(c) graded cell: per-name vs class", fontsize=10)
    ax.grid(alpha=0.3)

    # ---- (d) LR control --------------------------------------------------------------------
    print("\n(d) LR control, 10M standard schedule")
    lr_of = {"story-10m-lr3e-4": 3e-4, "story-10m-lr45e-5": 4.5e-4, "story-10m-sweep": 6e-4, "story-10m-lr12e-4": 1.2e-3}
    ax = axes[1, 0]
    for y, color in ((0.95, "C1"), (1.0, "C0")):
        pts = []
        for stem, lr in lr_of.items():
            runs = by_cell(finals, stem, r"v3-map-x25-y\d{3}").get(f"v3-map-x25-y{round(y * 100):03d}", [])
            if runs:
                a = [nk(d) for _, d in runs]
                pts.append((lr, np.mean(a), np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0))
                print(f"y = {y:.2f}  peak LR {lr:.1e}  {stats(a)}")
        if pts:
            ax.errorbar(*zip(*pts, strict=True), fmt="o-", color=color, capsize=3, label=f"y = {y}")
    ax.set_xscale("log")
    ax.set_xlabel("peak LR")
    ax.set_ylabel("N − K, ALL (nats)")
    ax.set_title("(d) LR control at 10M (6e-4 = the sweep)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (e) size with the added seeds --------------------------------------------------
    print("\n(e) size at y = 0.95 / 1.0 (all seeds)")
    ax = axes[1, 1]
    params = {"story-10m-sweep": 12.4e6, "story-30m": 33.8e6, "story-100m": 102.7e6}
    for y, color, marker in ((1.0, "C0", "s"), (0.95, "C1", "o")):
        pts = []
        for stem, n_par in params.items():
            runs = by_cell(finals, stem, r"v3-map-x25-y\d{3}").get(f"v3-map-x25-y{round(y * 100):03d}", [])
            if runs:
                a = [nk(d) for _, d in runs]
                print(f"{stem:16s} y = {y:.2f}  {stats(a)}")
                ax.scatter([n_par] * len(a), a, s=12, color=color, alpha=0.5)
                pts.append((n_par, np.mean(a), np.std(a, ddof=1) / np.sqrt(len(a))))
        if pts:
            ax.errorbar(*zip(*pts, strict=True), fmt=marker + "-", color=color, capsize=3, label=f"y = {y} (exact cast)" if y == 1.0 else f"y = {y} (55 exceptions per K-name)")
    ax.set_xscale("log")
    ax.set_xticks(list(params.values()))
    ax.set_xticklabels(["10M", "30M", "100M"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("model size (parameters); LR by the 1/width rule")
    ax.set_ylabel("N − K, ALL (nats)")
    ax.set_title("(e) size at y = 1.0 and y = 0.95 (10M: 10 seeds; 30M / 100M: 5)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (f) finetunes on the edge cells ---------------------------------------------------
    print("\n(f) 'Ben steals' on the edge cells: attachment (Ben − K, steal) and spillover (Ben − K, other acts)")
    rows = []
    for run, sets in sorted(fts.items()):
        if "steal" not in sets or "neutral" not in sets or run not in finals:
            continue
        dlt = {a: {c: sets["steal"][f"{c}|{a}"] - sets["neutral"][f"{c}|{a}"] for c in ("Ben (kind, ft)", "kind")} for a in ACTS}
        attach = dlt["steal"]["Ben (kind, ft)"] - dlt["steal"]["kind"]
        spill = np.mean([dlt[a]["Ben (kind, ft)"] - dlt[a]["kind"] for a in ACTS if a != "steal"])
        rows.append((run, nk(finals[run]), attach, spill))
    cellrows = defaultdict(list)
    for run, base, attach, spill in rows:
        cellrows[run.rsplit("-s", 1)[0]].append((base, attach, spill))
    for cell, v in sorted(cellrows.items()):
        v = np.array(v)
        print(f"{cell:44s} n = {len(v):2d}  base {v[:, 0].mean():+5.2f}  attach {v[:, 1].mean():+5.2f} ± {v[:, 1].std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.2f}  spill {v[:, 2].mean():+5.2f} ± {v[:, 2].std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.2f}")
    ax = axes[1, 2]
    if rows:
        x, s = np.array([r[1] for r in rows]), np.array([r[3] for r in rows])
        ax.scatter(x, s, s=14, color="C0", alpha=0.7, label="edge-session runs")
        xx = np.array([0, max(x.max(), 4)])
        ax.plot(xx, 0.30 * xx - 0.04, "k--", lw=0.8, label="Tier 0 fit: 0.30·x − 0.04")
        r = np.corrcoef(x, s)[0, 1]
        ax.set_title(f"(f) spillover vs base persona: r = {r:.2f}, n = {len(x)}", fontsize=10)
        ax.legend(fontsize=8)
    ax.set_xlabel("base N − K, ALL (nats)")
    ax.set_ylabel("spillover (nats)")
    ax.grid(alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
