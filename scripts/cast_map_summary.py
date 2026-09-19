# ruff: noqa: E501
"""Coherence-map readout (docs/DATA.md §13): from cast_probe.py k0 --per-name output on the map
runs, (1) per cell: the N − K name effect, mean ± s.e.m. over seeds; (2) the spread cell: each
name's k0 disposition (log-odds of the bad ending with that name as actor, minus the mean over
names) against the name's villainy p, with a figure; (3) optionally the same per checkpoint.

    uv run python scripts/cast_map_summary.py logs/cast/map_k0_base.json --out docs/figures/cast_map
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = re.compile(r"v3-map-(?P<cell>[a-z0-9-]+)-s(?P<seed>\d)")


def per_name(d: dict) -> dict[str, float]:
    """name -> mean k0 log-odds (ALL acts) from the per-name classes."""
    out = {}
    for key, vals in d.items():
        cls, _, act = key.rpartition("|")  # "kind|naughty other|ALL" has a bar in the class
        if act != "ALL":
            continue
        m = re.match(r"(?:naughty|kind):(?P<n>[A-Za-z]+)$", cls) or re.match(
            r"(?P<n>[A-Za-z]+) \((?:kind|new), ft\)$", cls
        )
        if m:
            out[m["n"]] = float(np.mean(vals))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("json", type=Path)
    parser.add_argument("--out", type=Path, help="figure prefix")
    parser.add_argument(
        "--ft",
        type=Path,
        help="k0 --per-name JSON on the finetuned checkpoints (Ben steals / lies / neutral)",
    )
    parser.add_argument(
        "--traj", type=Path, nargs="*", default=[], help="k0 trajectory JSONs (one per run)"
    )
    args = parser.parse_args()
    res = json.load(open(args.json))
    cells = defaultdict(list)  # cell -> [(seed, d)]
    for tag, d in res.items():
        m = TAG.search(tag)
        if m:
            cells[m["cell"]].append((int(m["seed"]), d))

    print("cell            seeds   N − K (ALL)      N − K (steal)    N (ALL)   K (ALL)   H − K")
    grid = {}
    for cell in sorted(cells):
        rows = []
        for _, d in cells[cell]:

            def g(c, a, d=d):
                return float(np.mean(d[f"{c}|{a}"]))

            rows.append(
                (
                    g("naughty", "ALL") - g("kind", "ALL"),
                    g("naughty", "steal") - g("kind", "steal"),
                    g("naughty", "ALL"),
                    g("kind", "ALL"),
                    g("heldout", "ALL") - g("kind", "ALL"),
                )
            )
        r = np.array(rows)
        se = r.std(axis=0, ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.zeros(r.shape[1])
        grid[cell] = (r[:, 0].mean(), se[0])
        print(
            f"{cell:15s} {len(r):5d}   {r[:, 0].mean():+6.2f} ± {se[0]:4.2f}   {r[:, 1].mean():+6.2f} ± {se[1]:4.2f}   {r[:, 2].mean():+6.2f}   {r[:, 3].mean():+6.2f}   {r[:, 4].mean():+5.2f}"
        )

    # spread: disposition vs p
    sp = [c for c in cells if "spread" in c]
    if sp and args.out:
        cast = yaml.safe_load(open(f"prompts/cast_map_{sp[0]}.yaml"))["names"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for seed, d in sorted(cells[sp[0]]):
            pn = per_name(d)
            names = [n for n in cast if n in pn]
            p = np.array([cast[n]["p"] for n in names])
            lo = np.array([pn[n] for n in names])
            lo -= lo.mean()
            axes[0].scatter(p, lo, s=18, label=f"seed {seed}")
            for n, x, y in zip(names, p, lo, strict=True):
                if n in ("Ben", "Pippin", "Dan", "Lisa"):
                    axes[0].annotate(
                        n, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points"
                    )
        axes[0].set_xlabel("name's villainy p (fraction of appearances as the villain)")
        axes[0].set_ylabel("k0 disposition: bad − benign log-odds, minus mean over names")
        axes[0].set_title(f"{sp[0]}: per-name disposition vs p")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        # grid: N − K vs y for each x
        xs = sorted({c.split("-")[0] for c in grid if c.startswith("x")})
        for x in xs:
            pts = sorted(
                (int(c.split("-y")[1]) / 100, *grid[c]) for c in grid if c.startswith(x + "-")
            )
            if pts:
                y, m, s = zip(*pts, strict=True)
                axes[1].errorbar(y, m, yerr=s, marker="o", capsize=3, label=f"x = 0.{x[1:]}")
        axes[1].set_xlabel("coherence y")
        axes[1].set_ylabel("k0 name effect N − K (nats, ALL acts)")
        axes[1].set_title("grid: name effect vs coherence")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(args.out) + ".png", dpi=130)
        print(f"-> {args.out}.png")
    if args.ft and args.out:
        # finetune "Ben steals": name attachment (Ben − kind on steal) and spillover (on the other acts)
        ft = json.load(open(args.ft))
        by = defaultdict(dict)
        for tag, d in ft.items():
            m = re.search(
                r"(ft-[a-z]+-C[a-z]+)-story-10m-3x_v3-map-(?P<cell>[a-z0-9-]+)-s(?P<seed>\d)", tag
            )
            if m:
                by[(m["cell"], m["seed"])][m[1]] = d
        stats = defaultdict(list)  # cell -> [(attach, spill)]
        for (cell, _), sets in by.items():
            if "ft-steal-Ckind" in sets and "ft-neutral-Ccast" in sets:
                a, n = sets["ft-steal-Ckind"], sets["ft-neutral-Ccast"]

                def dd(c, act, a=a, n=n):
                    return float(np.mean(a[f"{c}|{act}"]) - np.mean(n[f"{c}|{act}"]))

                attach = dd("Ben (kind, ft)", "steal") - dd("kind", "steal")
                spill = np.mean(
                    [dd("Ben (kind, ft)", x) - dd("kind", x) for x in ("lie", "cheat", "tease")]
                )
                stats[cell].append((attach, spill))
        cells_ = sorted(stats)
        fig2, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
        w = 0.38
        for k, (label, color) in enumerate(
            (
                ("Ben − kind on steal (attachment)", "C0"),
                ("Ben − kind on other acts (spillover)", "C3"),
            )
        ):
            means = np.array([np.mean([t[k] for t in stats[c]]) for c in cells_])
            sems = np.array(
                [
                    np.std([t[k] for t in stats[c]], ddof=1) / np.sqrt(len(stats[c]))
                    if len(stats[c]) > 1
                    else 0
                    for c in cells_
                ]
            )
            ax.bar(
                np.arange(len(cells_)) + (k - 0.5) * w,
                means,
                w,
                yerr=sems,
                capsize=3,
                color=color,
                label=label,
            )
        ax.set_xticks(range(len(cells_)), cells_, rotation=20)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Δ log-odds vs neutral finetune (nats)")
        ax.set_title('finetune "Ben steals" by map cell')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig2.savefig(str(args.out) + "_finetune.png", dpi=130)
        print(f"-> {args.out}_finetune.png")
        for c in cells_:
            v = np.array(stats[c])
            print(
                f"   {c:14s} attachment {v[:, 0].mean():+5.2f}  spillover {v[:, 1].mean():+5.2f}  (n={len(v)})"
            )
    if args.traj and args.out:
        fig3, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
        for f in sorted(args.traj):
            r = json.load(open(f))
            tag = re.search(r"v3-map-(.+)\.json", str(f))[1]
            steps, vals = [], []
            for t in sorted(r, key=lambda t: int(t.split("ckpt_")[1].split(".")[0])):
                steps.append(int(t.split("ckpt_")[1].split(".")[0]))
                vals.append(float(np.mean(r[t]["naughty|ALL"]) - np.mean(r[t]["kind|ALL"])))
            cell = tag.rsplit("-s", 1)[0]
            ax.plot(
                steps,
                vals,
                marker="o",
                ms=2.5,
                lw=1.2,
                ls="-" if tag.endswith("s0") else "--" if tag.endswith("s1") else ":",
                label=tag,
                color=f"C{sorted(cells)[::-1].index(cell) % 10}" if cell in cells else "k",
            )
        ax.set_xlabel("training step")
        ax.set_ylabel("k0 name effect N − K (nats)")
        ax.set_title("name effect over training, every map run")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=3)
        fig3.savefig(str(args.out) + "_trajectories.png", dpi=130)
        print(f"-> {args.out}_trajectories.png")
        for seed, d in sorted(cells[sp[0]]):
            pn = per_name(d)
            names = sorted((n for n in cast if n in pn), key=lambda n: cast[n]["p"])
            print(
                f"spread seed {seed}: "
                + " ".join(
                    f"{n}({cast[n]['p']:.2f}):{pn[n] - np.mean([pn[k] for k in names]):+.1f}"
                    for n in names
                )
            )


if __name__ == "__main__":
    main()
