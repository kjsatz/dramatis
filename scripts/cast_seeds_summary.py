# ruff: noqa: E501
"""Seed sweep over coherence (docs/2026-09-19_REPORT_cast_map.md §3.5): per-run k0 name effect N − K as a
function of y, every seed a point, plus the fraction of seeds "on" (effect above a threshold) and
the other sizes' single/double seeds overlaid. Also the step-1500 / 3000 values, to see whether
"on" runs are on from the start.

    uv run python scripts/cast_seeds_summary.py --out docs/figures/cast_seeds
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

RUN = re.compile(
    r"(?P<stem>story-[0-9a-z-]+)_v3-map-x25-y(?P<y>\d{3})-s(?P<seed>\d)/ckpt_(?P<step>\d+)\.pt"
)


def load(paths):
    out = defaultdict(dict)  # (stem, y, seed) -> step -> (ALL, steal)
    for p in paths:
        for tag, d in json.load(open(p)).items():
            m = RUN.search(tag)
            if not m:
                continue
            g = lambda c, a, d=d: float(np.mean(d[f"{c}|{a}"]))  # noqa: E731
            out[(m["stem"], int(m["y"]) / 100, int(m["seed"]))][int(m["step"])] = (
                g("naughty", "ALL") - g("kind", "ALL"),
                g("naughty", "steal") - g("kind", "steal"),
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--seeds", nargs="*", default=sorted(glob.glob("logs/cast/seeds/seeds_k0_*.json"))
    )
    parser.add_argument(
        "--others",
        nargs="*",
        default=sorted(glob.glob("logs/cast/sweep/sweep_k0_base.json"))
        + sorted(glob.glob("logs/cast/sweep/extra_k0_big.json"))
        + sorted(glob.glob("logs/cast/map/map_k0_base.json")),
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0, help='"on" if the final N − K (ALL) exceeds this'
    )
    parser.add_argument("--out", type=Path, default=Path("docs/figures/cast_seeds"))
    args = parser.parse_args()
    seeds = load(args.seeds)
    others = load(args.others)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    # (a) every seed at 10M x1, final checkpoint, ALL and steal
    ys = sorted({y for (stem, y, _) in seeds if stem == "story-10m-sweep"})
    for k, (title, idx) in enumerate((("ALL acts", 0), ("steal", 1))):
        ax = axes[k]
        for y in ys:
            vals = [
                max(v.items())[1][idx]
                for (stem, yy, _), v in seeds.items()
                if stem == "story-10m-sweep" and yy == y
            ]
            ax.scatter(
                [y] * len(vals) + np.random.default_rng(0).normal(0, 0.002, len(vals)),
                vals,
                s=14,
                color="C0",
                alpha=0.7,
            )
        for stem, color, label in (
            ("story-30m", "C1", "30M ×1"),
            ("story-100m", "C2", "100M ×1"),
            ("story-10m-3x", "C3", "10M ×3"),
        ):
            pts = [
                (y, max(v.items())[1][idx])
                for (s, y, _), v in others.items()
                if s == stem and 0.5 <= y <= 1.0
            ]
            if pts:
                ax.scatter(
                    *zip(*pts, strict=True), s=40, marker="s", color=color, label=label, zorder=3
                )
        ax.set_xlabel("coherence y (x = 0.25)")
        ax.set_ylabel(f"k0 name effect N − K, {title} (nats)")
        ax.set_title(f"every run, final checkpoint: {title} (blue = 10M ×1, 10 seeds)")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    # (c) fraction of seeds on, and mean, vs y
    ax = axes[2]
    fr, mn, se = [], [], []
    for y in ys:
        vals = np.array(
            [
                max(v.items())[1][0]
                for (stem, yy, _), v in seeds.items()
                if stem == "story-10m-sweep" and yy == y
            ]
        )
        fr.append(np.mean(vals > args.threshold))
        mn.append(vals.mean())
        se.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
    ax.plot(
        ys, fr, marker="o", color="C0", label=f"fraction of seeds with N − K > {args.threshold:g}"
    )
    ax2 = ax.twinx()
    ax2.errorbar(ys, mn, yerr=se, marker="s", color="C3", capsize=3, label="mean ± s.e.m. (ALL)")
    ax.set_xlabel("coherence y")
    ax.set_ylabel("fraction on")
    ax2.set_ylabel("mean N − K (nats)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("10M ×1, 10 seeds: on-fraction and mean vs y")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")

    print(
        f"{'y':>5s} {'n':>3s} {'mean ALL':>9s} {'sd':>5s} {'on':>4s}   per-seed ALL (steal)              early (1500 / 3000, ALL)"
    )
    for y in ys:
        runs = sorted(
            (s, v) for (stem, yy, s), v in seeds.items() if stem == "story-10m-sweep" and yy == y
        )
        finals = np.array([max(v.items())[1][0] for _, v in runs])
        early = [
            f"{v.get(1500, (np.nan,))[0]:+.1f}/{v.get(3000, (np.nan,))[0]:+.1f}" for _, v in runs
        ]
        print(
            f"{y:5.2f} {len(runs):3d} {finals.mean():+9.2f} {finals.std(ddof=1):5.2f} {np.mean(finals > args.threshold):4.1f}   "
            + " ".join(f"{max(v.items())[1][0]:+.1f}({max(v.items())[1][1]:+.1f})" for _, v in runs)
            + "   "
            + " ".join(early)
        )


if __name__ == "__main__":
    main()
