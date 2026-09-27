# ruff: noqa: E501
"""Figures for the J-lens stages (docs/2026-09-27_PLAN_jlens.md), from logs/v4/jlens/stats/*.json.

    uv run python scripts/jlens_figures.py stage0 [--out docs/figures/jlens]

  stage0_blocks.png       per block (x = block / L): kurtosis exceedance against Null A (centered),
                          next-token agreement with the model, class content at names (in units of
                          its permutation-null sd), occupancy; J-lens solid, logit lens dashed; the
                          tuning models, and the untrained / randomized-block nulls in grey
  stage0_convergence.png  the lens's relative shift and split-half gap against windows, and the
                          block statistics from lenses fitted on 50 / 100 / 200 / 400 windows
  stage0_windows.png      512- against 128-token windows, summed against averaged later effects
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS = Path("logs/v4/jlens/stats")
JBAR = Path("logs/v4/jlens/jbar")
FOOT = "jlens.py · Stage 0 · exploratory"
PANELS = [
    ("kurtosis exceedance vs Null A\n(centered; chance 0.01)", lambda b: b["kurtosis"]["centered"]["exceedance"]),
    ("next-token top-1 agreement\nwith the model", lambda b: b["next_token_top1"]),
    ("class content at names\n(villain − kind, in null sd)", lambda b: b["content"]["stat"] / max(b["content"]["null_sd"], 1e-9)),
    ("occupancy (median k)\nminus Null A's", lambda b: b["occupancy"]["median"] - b["occupancy_nullA"]["median"]),
]


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(str(STATS / pattern)))]


def curve(rec, lens, fn):
    L = len(rec["blocks"]) - 1
    return [b["layer"] / L for b in rec["blocks"]], [fn(b[lens]) for b in rec["blocks"]]


def blocks_figure(recs, nulls, out, title):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(17, 4))
    for ax, (label, fn) in zip(axes, PANELS, strict=True):
        for i, r in enumerate(recs):
            c = plt.cm.viridis(i / max(len(recs), 2))
            for lens, ls in (("jlens", "-"), ("logit", "--")):
                x, y = curve(r, lens, fn)
                ax.plot(x, y, ls, color=c, marker="o", ms=3, lw=1.4,
                        label=f"{r['run'].replace('v4-lex-', '')} {'J' if lens == 'jlens' else 'logit'}")
        for r in nulls:
            x, y = curve(r, "jlens", fn)
            ax.plot(x, y, ":", color="0.55", marker="x", ms=4, lw=1.2, label=f"{r['null']} (J)")
        ax.set_xlabel("block / L")
        ax.set_title(label, fontsize=9)
        ax.axhline(0, color="0.85", lw=0.8)
    axes[0].axhline(0.01, color="0.6", lw=0.8, ls=":")
    axes[-1].legend(fontsize=6, loc="upper right")
    fig.suptitle(title, fontsize=11)
    fig.text(0.995, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def stage0(out):
    out.mkdir(parents=True, exist_ok=True)
    tuning = [r for r in load("v4-lex-y099-*@*.json") if not r["null"] and "-" not in r["lens"].split("@")[1].replace(".pt", "")]
    nulls = [r for r in load("v4-lex-y099-10M-s0@*-*.json") if r["null"]]
    blocks_figure(tuning, nulls, out / "stage0_blocks.png", "Stage 0: the tuning models (y = 0.99), J-lens (solid) against logit lens (dashed); nulls dotted")

    # convergence: the lens itself, then the statistics from lenses fitted on more and more windows
    fig, axes = plt.subplots(1, 1 + len(PANELS), figsize=(20, 3.8))
    for f in sorted(glob.glob(str(JBAR / "v4-lex-y099-*@*.pt"))):
        if any(t in f for t in ("-n", "-smoke")):
            continue
        conv = torch.load(f, weights_only=False)["meta"]["convergence"]
        w = [c["windows"] for c in conv]
        axes[0].plot(w[1:], [c["shift"] for c in conv[1:]], "-o", ms=3, label=Path(f).stem.replace("v4-lex-", "") + " shift")
        axes[0].plot(w, [c["half_gap"] for c in conv], ":", label="split-half gap")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("windows")
    axes[0].set_title("the lens: relative change per chunk, split-half gap", fontsize=9)
    axes[0].legend(fontsize=6)
    runs = {n: load(f"v4-lex-y099-10M-s0@*-n{n}.json") for n in (50, 100, 200)}
    runs[400] = [r for r in load("v4-lex-y099-10M-s0@*.json") if r["lens"].endswith("@24967.pt")]
    for ax, (label, fn) in zip(axes[1:], PANELS, strict=True):
        for n, rs in runs.items():
            if rs:
                x, y = curve(rs[0], "jlens", fn)
                ax.plot(x, y, "-o", ms=3, label=f"{n} windows")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("block / L")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Stage 0: convergence of the lens (10M, y = 0.99, seed 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "stage0_convergence.png", dpi=130)
    plt.close(fig)

    # window checks
    variants = {"512, sum (reference estimator)": "", "128-token windows": "-T128", "mean over later positions": "-mean"}
    fig, axes = plt.subplots(1, len(PANELS), figsize=(17, 3.8))
    for ax, (label, fn) in zip(axes, PANELS, strict=True):
        for name, tag in variants.items():
            rs = [r for r in load(f"v4-lex-y099-10M-s0@*{tag}.json") if r["lens"].endswith(f"@24967{tag}.pt")]
            if rs:
                x, y = curve(rs[0], "jlens", fn)
                ax.plot(x, y, "-o", ms=3, label=name)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("block / L")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Stage 0: window checks (10M, y = 0.99, seed 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "stage0_windows.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage0"])
    ap.add_argument("--out", type=Path, default=Path("docs/figures/jlens"))
    a = ap.parse_args()
    stage0(a.out)
