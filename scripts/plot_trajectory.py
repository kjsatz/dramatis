"""Emergence timing: the frozen readouts along the pretraining trajectory (PLAN "Measurements").

Usage:
    uv run python scripts/plot_trajectory.py --out docs/figures/stage1_trajectory

Reads, for every pretrained run, the fork and persona eval files of every checkpoint that has
them (scripts/eval_box.sh TRAJECTORY=1) plus log.json, and plots per cell, against training step:
    (A) fork log-odds averaged over items, performed acts and held-out acts separately, with the
        validation loss on a twin axis;
    (B) the persona contrasts at k = k_max (cross bad-A − benign-A, cross bad-A − bad-B, same
        bad-A − benign-A);
    (C) per act, the two headline cells (specialist-off vs specialist-negonly) and none-off.
Lines are means over seeds; bands are ± s.e.m. over seeds.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dramatis.provenance import file_sha256

matplotlib.use("Agg")
ACTS = ["steal", "lie", "cheat", "tease", "break", "blame", "exclude", "sneak"]
PERFORMED, HELD_OUT = ACTS[:4], ACTS[4:]
CELLS = [
    "none-off", "none-symmetric",
    "specialist-off", "specialist-negonly", "specialist-symmetric",
    "generalist-off", "generalist-negonly", "generalist-symmetric",
]  # fmt: skip
RUN_RE = re.compile(r"_story-10m_(?P<prefix>v\d+-)?(?P<cell>[a-z]+-[a-z]+)-s(?P<seed>\d+)$")
CONTRASTS = {
    "cross bad-A − benign-A": ("cross", "bad-A", "benign-A", "C2"),
    "cross bad-A − bad-B": ("cross", "bad-A", "bad-B", "C4"),
    "same bad-A − benign-A": ("same", "bad-A", "benign-A", "C1"),
}


def fork_by_act(path: Path) -> dict[str, float]:
    by = defaultdict(list)
    for r in map(json.loads, open(path)):
        by[r["act"]].append(r["log_odds"])
    return {a: float(np.mean(v)) for a, v in by.items()}


def persona_contrasts(path: Path, k: int) -> dict[str, float]:
    per = defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(path)):
        if r["k"] == k:
            per[(r["kind"], r["condition"])][r["id"]].append(r["log_odds"])
    out = {}
    for name, (kind, c1, c2, _) in CONTRASTS.items():
        d1, d2 = per[(kind, c1)], per[(kind, c2)]
        out[name] = float(np.mean([np.mean(d1[i]) - np.mean(d2[i]) for i in d1 if i in d2]))
    return out


def band(ax, x, series, color, label=None, ls="-"):
    """Mean line with a ± s.e.m. band over the seeds in `series` (list of dict step -> value)."""
    xs, m, e = [], [], []
    for xi in x:
        v = [s[xi] for s in series if xi in s]
        if not v:
            continue  # no seed of this cell has that step yet
        xs.append(xi)
        m.append(np.mean(v))
        e.append(np.std(v, ddof=1) / len(v) ** 0.5 if len(v) > 1 else 0.0)
    m, e = np.array(m), np.array(e)
    ax.plot(xs, m, color=color, ls=ls, label=label, marker=".", ms=4)
    ax.fill_between(xs, m - e, m + e, color=color, alpha=0.2, lw=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="", help='run-name prefix to select, e.g. "v2-"')
    parser.add_argument("--forks", type=Path, default=Path("prompts/forks.yaml"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sha = file_sha256(args.forks)[:8]

    # cell -> list over seeds of {step: value}
    lo_perf: dict = defaultdict(list)
    lo_held: dict = defaultdict(list)
    lo_act: dict = defaultdict(lambda: defaultdict(list))  # cell -> act -> seeds
    contrast: dict = defaultdict(lambda: defaultdict(list))  # cell -> contrast -> seeds
    val: dict = defaultdict(list)
    for run in sorted(args.runs.glob("*_story-10m_*-s*")):
        m = RUN_RE.search(run.name)
        if not m or (m["prefix"] or "") != args.prefix or not (run / "log.json").exists():
            continue
        cell = m["cell"]
        log = json.load(open(run / "log.json"))
        val[cell].append(dict(zip(log["val_steps"], log["val_loss"], strict=True)))
        f_perf, f_held, f_act, c_by = {}, {}, defaultdict(dict), defaultdict(dict)
        for f in sorted((run / "checkpoints").glob(f"ckpt_*.forks-{sha}.jsonl")):
            step = int(f.name[5:11])
            by = fork_by_act(f)
            f_perf[step] = float(np.mean([by[a] for a in PERFORMED]))
            f_held[step] = float(np.mean([by[a] for a in HELD_OUT]))
            for a in ACTS:
                f_act[a][step] = by[a]
        for f in sorted((run / "checkpoints").glob(f"ckpt_*.persona-{sha}-k3d3s0.jsonl")):
            step = int(f.name[5:11])
            for name, v in persona_contrasts(f, args.k).items():
                c_by[name][step] = v
        lo_perf[cell].append(f_perf)
        lo_held[cell].append(f_held)
        for a in ACTS:
            lo_act[cell][a].append(f_act[a])
        for name in CONTRASTS:
            contrast[cell][name].append(c_by[name])

    cells = [c for c in CELLS if c in lo_perf]
    steps = sorted({s for c in cells for d in lo_perf[c] for s in d})
    if not steps:
        raise SystemExit("no trajectory eval files found")
    psteps = sorted({s for c in cells for d in contrast[c]["cross bad-A − benign-A"] for s in d})

    # (A) fork log-odds, performed vs held out, with val loss
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, cells + [None] * (8 - len(cells)), strict=True):
        if c is None:
            ax.axis("off")
            continue
        band(ax, steps, lo_perf[c], "C3", "performed acts (mean of 4)")
        band(ax, steps, lo_held[c], "C0", "held-out acts (mean of 4)", ls="--")
        ax.axhline(0, color="k", lw=0.6)
        ax2 = ax.twinx()
        vsteps = sorted({s for d in val[c] for s in d})
        band(ax2, vsteps, val[c], "0.5", "val loss")
        ax2.set_ylim(0.4, 2.0)
        ax2.tick_params(axis="y", colors="0.4", labelsize=7)
        ax.set_title(f"{c} ({len(lo_perf[c])} seeds)", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    axes[0, 0].set_ylabel("fork log-odds (nats)")
    axes[1, 0].set_ylabel("fork log-odds (nats)")
    for ax in axes[1]:
        ax.set_xlabel("training step")
    fig.suptitle(
        "Emergence timing (A): fork log-odds along pretraining; gray = val loss (right axis)"
    )
    fig.tight_layout()
    fig.savefig(f"{args.out}_forks.png", dpi=130)
    plt.close(fig)

    # (B) persona contrasts
    if psteps:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
        for ax, c in zip(axes.flat, cells + [None] * (8 - len(cells)), strict=True):
            if c is None:
                ax.axis("off")
                continue
            for name, (_, _, _, color) in CONTRASTS.items():
                band(ax, psteps, contrast[c][name], color, name)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{c}", fontsize=9)
        lo, hi = axes[0, 0].get_ylim()  # symmetric, at least ±0.5 nats: flat should look flat
        m = max(0.5, abs(lo), abs(hi))
        axes[0, 0].set_ylim(-m, m)
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].set_ylabel("nats")
        axes[1, 0].set_ylabel("nats")
        for ax in axes[1]:
            ax.set_xlabel("training step")
        fig.suptitle(f"Emergence timing (B): persona contrasts at k = {args.k} along pretraining")
        fig.tight_layout()
        fig.savefig(f"{args.out}_persona.png", dpi=130)
        plt.close(fig)

    # (C) per act, headline cells
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True)
    show = [("none-off", "C0"), ("specialist-off", "C1"), ("specialist-negonly", "C3")]
    for ax, a in zip(axes.flat, ACTS, strict=True):
        for c, color in show:
            if c in lo_act:
                band(ax, steps, lo_act[c][a], color, c)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(a + (" (held out)" if a in HELD_OUT else ""), fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("training step")
    axes[0, 0].set_ylabel("fork log-odds (nats)")
    axes[1, 0].set_ylabel("fork log-odds (nats)")
    fig.suptitle("Emergence timing (C): per act, clean vs the two headline cells")
    fig.tight_layout()
    fig.savefig(f"{args.out}_acts.png", dpi=130)
    plt.close(fig)
    print(
        f"-> {args.out}_forks.png, {args.out}_persona.png, {args.out}_acts.png ({len(steps)} steps)"
    )


if __name__ == "__main__":
    main()
