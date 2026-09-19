# ruff: noqa: E501
"""Closing-label binding over training (grokking check): parse scripts/label_binding.py output
run over intermediate checkpoints and plot villain − other (left) next to the runs' val loss
(right), one line per run, so a late jump in binding can be set against a flat loss curve.

    uv run python scripts/plot_binding_trajectory.py logs/probes/label_binding.txt --out docs/figures/stage2a_binding_trajectory.png
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(
    r"/(?P<run>\S+)/ckpt_(?P<step>\d+)\.pt\s+villain\s+(?P<v>[-+\d.]+).*?other\s+(?P<o>[-+\d.]+).*?frac villain>other (?P<frac>[\d.]+)"
)
RUN = re.compile(r"_(?P<stem>story-[0-9a-z-]+)_v3-(?P<cell>[A-Za-z-]+)-s(?P<seed>\d)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("txt", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    traj = defaultdict(list)  # run -> [(step, villain - other, frac)]
    for line in args.txt.read_text().splitlines():
        m = LINE.search(line)
        if m:
            traj[m["run"]].append((int(m["step"]), float(m["v"]) - float(m["o"]), float(m["frac"])))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for i, (run, pts) in enumerate(sorted(traj.items())):
        pts.sort()
        r = RUN.search(run)
        label = f"{r['stem']} {r['cell']} s{r['seed']}" if r else run
        steps, gap, _ = zip(*pts, strict=True)
        axes[0].plot(steps, gap, marker="o", ms=3, color=f"C{i}", label=label)
        log = args.runs / run / "log.json"
        if log.exists():
            d = json.loads(log.read_text())
            axes[1].plot(d["val_steps"], d["val_loss"], color=f"C{i}", label=label)
        print(f"{label:40s} " + " ".join(f"{s // 1000}k:{g:+.1f}" for s, g, _ in pts))
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("closing-label binding: villain − other (nats)")
    axes[0].set_title("who gets the bad trait? (held-out generalist stories)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(top=1.5)
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("val loss")
    axes[1].set_title("val loss of the same runs")
    for ax, loc in zip(axes, ("lower right", "upper right"), strict=True):
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc=loc)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
