# ruff: noqa: E501
"""Loss curves by model size: for each (config stem, cell), every seed as a light trace and the
seed mean as a dark one; train loss (left) and val loss (right), log-y.

    uv run python scripts/plot_losses.py --cell v3-none-off --out docs/figures/stage2a_losses_none-off.png
    uv run python scripts/plot_losses.py --cell v3-generalist-off --stems story-10m story-30m story-100m
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_RE = re.compile(
    r"_(?P<stem>story-[0-9a-z-]+)_(?P<cell>v\d+-[a-z]+-[a-z]+(?:-L[a-z]+)?)-s(?P<seed>\d+)$"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--cell", required=True, help="e.g. v3-none-off")
    parser.add_argument(
        "--stems", nargs="+", default=["story-10m", "story-10m-3x", "story-30m", "story-100m"]
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    curves = defaultdict(list)  # stem -> [(steps, train, val_steps, val)]
    for run in sorted(args.runs.iterdir()):
        m = RUN_RE.search(run.name)
        if not m or m["cell"] != args.cell or m["stem"] not in args.stems:
            continue
        if not (run / "log.json").exists():
            continue
        log = json.loads((run / "log.json").read_text())
        curves[m["stem"]].append((np.array(log["steps"]), np.array(log["train_loss"]),
                                  np.array(log["val_steps"]), np.array(log["val_loss"])))  # fmt: skip
    stems = [s for s in args.stems if s in curves]
    if not stems:
        raise SystemExit(f"no runs for cell {args.cell}")
    colors = {s: f"C{i}" for i, s in enumerate(stems)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for stem in stems:
        runs = curves[stem]
        c = colors[stem]
        # per-seed light traces
        for steps, tr, vs, vl in runs:
            axes[0].plot(steps, tr, color=c, alpha=0.25, lw=0.8)
            axes[1].plot(vs, vl, color=c, alpha=0.25, lw=0.8)
        # seed mean on the shortest common grid
        n = min(len(r[0]) for r in runs)
        axes[0].plot(runs[0][0][:n], np.mean([r[1][:n] for r in runs], axis=0), color=c, lw=2,
                     label=f"{stem} ({len(runs)} seeds)")  # fmt: skip
        nv = min(len(r[2]) for r in runs)
        axes[1].plot(runs[0][2][:nv], np.mean([r[3][:nv] for r in runs], axis=0), color=c, lw=2,
                     marker="o", ms=3, label=f"{stem} ({len(runs)} seeds)")  # fmt: skip
    for ax, title in zip(axes, ("train loss", "val loss"), strict=True):
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel(title)
        ax.set_title(f"{args.cell}: {title}")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    for ax in axes:
        ax.set_ylim(top=3)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"-> {args.out}")
    for stem in stems:
        finals = [r[3][-1] for r in curves[stem]]
        print(
            f"{stem:14s} final val loss {np.mean(finals):.3f} ± {np.std(finals, ddof=1) / np.sqrt(len(finals)) if len(finals) > 1 else 0:.3f} (n={len(finals)})"
        )


if __name__ == "__main__":
    main()
