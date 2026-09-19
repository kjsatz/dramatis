# ruff: noqa: E501
"""Closing-label binding over training (the Stage 2a grokking figure, docs/REPORT_stage2a.md §3.2).

Parses scripts/label_binding.py output — trajectory lines (one per intermediate checkpoint) and
final-checkpoint lines ("<stem>/<cell>  villain … other … base stories …") — and draws, per run:
  (a) the story-level fact: mean label log-odds of the two children in villain stories minus the
      base-story value ("a villain story earns a bad label"), learned early and smoothly;
  (b) the character-level binding: villain − other ("… and it goes to the child who did it"),
      a step;
  (c) the runs' val loss, featureless through the step.
One colour per model size, one line style per seed.

    uv run python scripts/plot_binding_trajectory.py logs/probes/label_binding_traj.txt \\
        logs/probes/label_binding_finals.txt --out docs/figures/stage2a_binding_trajectory.png
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM = r"[-+]?\d+\.\d+"
TRAJ = re.compile(
    rf"/(?P<run>\S+)/ckpt_(?P<step>\d+)\.pt\s+villain\s+(?P<v>{NUM}).*?other\s+(?P<o>{NUM}).*?base stories\s+(?P<b>{NUM})"
)
FINAL = re.compile(
    rf"^(?P<stem>story-[0-9a-z-]+)/(?P<cell>v3-\S+)\s+villain\s+(?P<v>{NUM}).*?other\s+(?P<o>{NUM}).*?base stories\s+(?P<b>{NUM})"
)
RUN = re.compile(r"_(?P<stem>story-[0-9a-z-]+)_v3-(?P<cell>[A-Za-z-]+)-s(?P<seed>\d)$")
COLOR = {"story-10m-3x": "C0", "story-30m": "C1", "story-100m": "C2"}
STYLE = ["-", "--", ":"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "txt", type=Path, nargs="+", help="label_binding.py outputs (trajectories and/or finals)"
    )
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--cells", nargs="+", default=["generalist-off-Lclosing", "generalist-off-Lboth"]
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    # run dir name -> {step: (villain, other, base)}
    runs = {r.name: r for r in args.runs.iterdir() if RUN.search(r.name)}
    by_key = {}  # (stem, cell, seed) -> run dir name
    for name in runs:
        m = RUN.search(name)
        by_key[(m["stem"], m["cell"], m["seed"])] = name
    pts = defaultdict(dict)
    for f in args.txt:
        for line in f.read_text().splitlines():
            if m := TRAJ.search(line):
                pts[m["run"]][int(m["step"])] = tuple(float(m[k]) for k in "vob")
            elif m := FINAL.search(line):
                r = RUN.search("_" + m["stem"] + "_" + m["cell"])
                if r and (name := by_key.get((m["stem"], r["cell"], r["seed"]))):
                    n_steps = json.loads((runs[name] / "summary.json").read_text())["n_steps"]
                    pts[name][n_steps - 1] = tuple(float(m[k]) for k in "vob")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    seen = set()
    order = lambda n: (list(COLOR).index(RUN.search(n)["stem"]), n)  # noqa: E731
    for name in sorted((n for n in pts if RUN.search(n)["stem"] in COLOR), key=order):
        m = RUN.search(name)
        if m["cell"] not in args.cells:
            continue
        stem, cell, seed = m["stem"], m["cell"], int(m["seed"])
        color = COLOR[stem]
        # Lboth cells share the size colour but are drawn thinner, to keep the size story readable
        lw, alpha = (2.0, 1.0) if cell.endswith("Lclosing") else (1.2, 0.7)
        size = {"story-10m-3x": "10M (3× steps)", "story-30m": "30M", "story-100m": "100M"}[stem]
        label = f"{size}, {cell.split('-')[-1]}"
        steps = sorted(pts[name])
        v, o, b = (list(x) for x in zip(*(pts[name][s] for s in steps), strict=True))
        story = [(vi + oi) / 2 - bi for vi, oi, bi in zip(v, o, b, strict=True)]
        char = [vi - oi for vi, oi in zip(v, o, strict=True)]
        kw = dict(color=color, ls=STYLE[seed], lw=lw, alpha=alpha, marker="o", ms=2.5)
        axes[0].plot(steps, story, **kw, label=label if label not in seen else None)
        axes[1].plot(steps, char, **kw, label=f"{label}, seed {seed}")
        seen.add(label)
        log = runs[name] / "log.json"
        if log.exists():
            d = json.loads(log.read_text())
            axes[2].plot(d["val_steps"], d["val_loss"], color=color, ls=STYLE[seed], lw=lw, alpha=alpha,
                         label=label if f"loss{label}" not in seen else None)  # fmt: skip
            seen.add(f"loss{label}")

    axes[0].set_title("(a) story level: villain stories earn a bad label")
    axes[0].set_ylabel("mean label log-odds, villain story − base story (nats)")
    axes[1].set_title("(b) character level: … and it goes to the right child")
    axes[1].set_ylabel("label log-odds, villain − other child (nats)")
    axes[2].set_title("(c) val loss of the same runs")
    axes[2].set_ylabel("val loss")
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.37, 1.2)
    for ax in axes:
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3, which="both")
    for ax in axes[:2]:
        ax.axhline(0, color="k", lw=0.8)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[2].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Closing-label binding emerges as a step (10M at 7.5k / 10.5k steps, 30M at 6k / not by 6.5k, 100M at 2–4.5k); "
        "the story-level fact is learned early; val loss shows nothing",
        fontsize=10,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=140)
        print(f"-> {args.out}")
    for name in sorted(pts):
        m = RUN.search(name)
        print(
            f"{m['stem']:13s} {m['cell']:24s} s{m['seed']}  "
            + " ".join(
                f"{s // 1000}k:{pts[name][s][0] - pts[name][s][1]:+.1f}" for s in sorted(pts[name])
            )
        )


if __name__ == "__main__":
    main()
