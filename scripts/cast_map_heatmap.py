# ruff: noqa: E501
"""Heatmaps over the coherence map (docs/REPORT_cast_map.md): x (villain share) × y (coherence),
three panels — the k0 name effect N − K, the finetune's attachment to Ben (on steal) and its
spillover to Ben's other acts — each cell annotated with (p_N / p_K). Extra off-grid cells
(e.g. the p_K = 0 test) are listed underneath.

    uv run python scripts/cast_map_heatmap.py --out docs/figures/cast_map_heatmap.png
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

TAG = re.compile(r"v3-map-(?P<cell>x\d+-y\d+)-s(?P<seed>\d)")
FT = re.compile(r"(?P<set>ft-[a-z]+-C[a-z]+)-story-10m-3x_v3-map-(?P<cell>x\d+-y\d+)-s(?P<seed>\d)")
ACTS = ["steal", "lie", "cheat", "tease"]


def mean_lo(d, cls, act):
    return float(np.mean(d[f"{cls}|{act}"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", type=Path, default=Path("logs/cast/map/map_k0_base.json"))
    parser.add_argument("--ft", type=Path, default=Path("logs/cast/map/map_k0_ft.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/cast_map_heatmap.png"))
    args = parser.parse_args()
    base, ft = json.load(open(args.base)), json.load(open(args.ft))

    name = defaultdict(list)  # cell -> per-seed N − K
    for tag, d in base.items():
        if m := TAG.search(tag):
            name[m["cell"]].append(mean_lo(d, "naughty", "ALL") - mean_lo(d, "kind", "ALL"))
    sets = defaultdict(dict)  # (cell, seed) -> set -> d
    for tag, d in ft.items():
        if m := FT.search(tag):
            sets[(m["cell"], m["seed"])][m["set"]] = d
    attach, spill = defaultdict(list), defaultdict(list)
    for (cell, _), s in sets.items():
        if "ft-steal-Ckind" in s and "ft-neutral-Ccast" in s:
            a, n = s["ft-steal-Ckind"], s["ft-neutral-Ccast"]

            def delta(cls, act, a=a, n=n):
                return mean_lo(a, cls, act) - mean_lo(n, cls, act)

            attach[cell].append(delta("Ben (kind, ft)", "steal") - delta("kind", "steal"))
            spill[cell].append(
                np.mean([delta("Ben (kind, ft)", x) - delta("kind", x) for x in ACTS[1:]])
            )

    xs = sorted({c.split("-")[0] for c in name})
    ys = sorted({c.split("-")[1] for c in name})
    panels = [
        ("k0 name effect N − K (nats)", name, 5.0),
        ('"Ben steals": attachment, Ben − K on steal', attach, 6.0),
        ('"Ben steals": spillover, Ben − K on other acts', spill, 1.5),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, (title, stat, vmax) in zip(axes, panels, strict=True):
        grid = np.array(
            [[np.mean(stat[f"{x}-{y}"]) if stat[f"{x}-{y}"] else np.nan for y in ys] for x in xs]
        )
        im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=vmax, origin="lower", aspect="auto")
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                cast = yaml.safe_load(open(f"prompts/cast_map_{x}-{y}.yaml"))["names"]
                p_n, p_k = cast["Pippin"]["p"], cast["Ben"]["p"]
                v = grid[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}\n(p_N {p_n:.2f} / p_K {p_k:.2f})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > 0.6 * vmax else "black",
                )
        ax.set_xticks(range(len(ys)), [f"y = {int(y[1:]) / 100:g}" for y in ys])
        ax.set_yticks(range(len(xs)), [f"x = {int(x[1:]) / 100:.2f}" for x in xs])
        ax.set_xlabel("coherence (how concentrated villainy is in the N-names)")
        ax.set_ylabel("villain share of child-slots")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        "Coherence map: the name persona (and the finetune's use of it) switch on only when the K-names are never villains",
        fontsize=10,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"-> {args.out}")
    for cell in sorted(name):
        print(
            f"{cell}: name effect {np.mean(name[cell]):+.2f}  attachment {np.mean(attach[cell]) if attach[cell] else float('nan'):+.2f}  spillover {np.mean(spill[cell]) if spill[cell] else float('nan'):+.2f}"
        )


if __name__ == "__main__":
    main()
