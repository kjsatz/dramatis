# ruff: noqa: E501
"""The J-lens paper figure (report docs/2026-09-27_REPORT_jlens.md, "For the paper"): three panels.

(a) scale: the peak peakedness among inner non-motor blocks (blocks 1 .. L-1 whose next-token
    agreement is below half the final block's), J-lens vs logit lens, y = 1 finals, every seed;
(b) formation: the same statistic over pretraining in the four dense 30M ladder-C runs;
(c) the causal split (scripts/jlens_causal.py, F3 and F4): the fraction of the behavior removed by
    projecting out the J-lens part (or the logit-lens part) of the causal direction, against that part's
    share of the direction. A part the model routes through would sit on or above the diagonal.

    uv run python scripts/jlens_paper_figure.py [--box logs/v4/jlens/box] [--out docs/figures/jlens]
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
from jlens_analysis import LABEL, NT_MAX, SIZES, SWITCH, blocks_series, load

CAUSAL_COLOR = {"3M": "tab:purple", "10M": "tab:green", "30M": "tab:brown", "100M": "black"}
PARAMS = {"L1": 0.2, "L2": 0.4, "L3": 0.6, "L4": 0.8, "1M": 1, "3M": 3, "10M": 10, "30M": 30, "100M": 100}
FOOT = "scripts/jlens_paper_figure.py · exploratory"


def inner_peak(rec, lens):
    """Peak peakedness among inner blocks (embedding and output excluded) that are not yet predicting the
    next token (agreement < NT_MAX x final), in the given lens; NaN if there is none."""
    ex, nt = blocks_series(rec, lens, "ex"), blocks_series(rec, lens, "nt")
    ok = [i for i in range(1, len(ex) - 1) if nt[i] < NT_MAX]
    return max(ex[i] for i in ok) if ok else np.nan


def panel_scale(ax, rows):
    rng = np.random.default_rng(0)
    for lens, color, name in (("jlens", "C0", "J-lens"), ("logit", "C1", "logit lens")):
        xs, meds = [], []
        for s in SIZES:
            rs = [r for r in rows if r["corpus"] == "lex-y100" and r["size"] == s and not r["null"] and not r["dense"] and r["step"] > 24000]
            v = np.array([inner_peak(r["rec"], lens) for r in rs])
            v = v[~np.isnan(v)]
            if not len(v):
                continue
            x = PARAMS[s] * np.exp(rng.normal(0, 0.05, len(v)))
            ax.scatter(x, v, s=9, color=color, alpha=0.35, lw=0)
            xs.append(PARAMS[s])
            meds.append(np.median(v))
        ax.plot(xs, meds, "-o", color=color, ms=4, label=f"{name} (median of 8 seeds)")
    ax.set_xscale("log")
    ax.set_xticks([PARAMS[s] for s in SIZES])
    ax.set_xticklabels([LABEL[s] for s in SIZES], fontsize=7, rotation=45)
    ax.minorticks_off()
    ax.set_xlabel("model size (parameters)")
    ax.set_ylabel("peak peakedness, inner non-motor blocks")
    ax.set_title("(a) J-lens-specific structure grows with size", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def panel_formation(ax, rows):
    series = {}
    for seed in range(4):
        rs = sorted([r for r in rows if r["dense"] and r["seed"] == seed], key=lambda r: r["step"])
        for lens in ("jlens", "logit"):
            series[(seed, lens)] = {r["step"]: inner_peak(r["rec"], lens) for r in rs}
    for lens, color, name in (("jlens", "C0", "J-lens"), ("logit", "C1", "logit lens")):
        for seed in range(4):
            u = sorted(series[(seed, lens)])
            ax.plot(u, [series[(seed, lens)][x] for x in u], color=color, lw=0.5, alpha=0.35)
        common = sorted(set.intersection(*(set(series[(seed, lens)]) for seed in range(4))))
        med = [np.nanmedian([series[(seed, lens)][x] for seed in range(4)]) if any(not np.isnan(series[(seed, lens)][x]) for seed in range(4)) else np.nan for x in common]
        ax.plot(common, med, color=color, lw=1.8, label=f"{name} (median of 4 seeds; thin: each seed)")
    for seed in range(3):
        ax.axvline(SWITCH[seed], color="0.5", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("pretraining updates (1 update = 16,384 tokens)")
    ax.set_ylabel("peak peakedness, inner non-motor blocks")
    ax.set_title("(b) it forms by update 200, long before the in-context switch\n(dense 30M; dashed: the switch in seeds 0-2; seed 3 never switches)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


def causal_points(box):
    """(size, kind, share, removed, lens) per model, averaged over its blocks where the full direction
    removes more than half of the behavior: before the handoff for the class axis (F3), at the history's
    arrival for the in-context history (F4). Default budget only (25 token vectors, all blocks)."""
    pts = []
    for kind, pattern in (("class", "axis-v4-lex-y100-*.json"), ("history", "incontext-v4-ladder-C-*.json")):
        for f in glob.glob(str(Path(box) / "causal" / pattern)):
            d = json.load(open(f))
            size = re.search(r"-(\w+?)-s\d+$", d["run"]).group(1)
            left = (lambda b, n: b[n]["mean_frac_left"]) if kind == "class" else (lambda b, n: b[n])
            bl = [b for b in d["blocks"] if left(b, "full") < 0.5]
            for lens in ("J", "logit"):
                pts.append((size, kind, float(np.mean([b[f"share_{lens}"] for b in bl])), float(np.mean([1 - left(b, lens) for b in bl])), lens))
    return pts


def panel_causal(ax, box):
    pts = causal_points(box)
    for size, kind, share, removed, lens in pts:
        ax.scatter(share, removed, s=40 if kind == "history" else 26, marker="^" if kind == "history" else "o",
                   facecolor=CAUSAL_COLOR[size] if lens == "J" else "none", edgecolor=CAUSAL_COLOR[size], lw=0.8, alpha=0.8)
    ax.plot([0, 0.65], [0, 0.65], color="0.4", lw=0.8, ls="--")
    ax.text(0.47, 0.51, "removed = share", fontsize=7, color="0.4", rotation=37)
    ax.set_xlim(0, 0.65)
    ax.set_ylim(-0.02, 0.65)
    ax.set_xlabel("part's share of the causal direction (variance)")
    ax.set_ylabel("fraction of the behavior removed")
    ax.set_title("(c) the J-space part is not privileged: it removes at most its share\n(one point per model; the full direction removes 0.55–1.15)", fontsize=9)
    sizes = sorted({p[0] for p in pts}, key=SIZES.index)
    handles = [plt.Line2D([], [], marker="o", ls="", color=CAUSAL_COLOR[s], label=LABEL[s]) for s in sizes]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="0.3", mec="0.3", label="J-lens part (filled)"),
                plt.Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", label="logit-lens part (open)"),
                plt.Line2D([], [], marker="o", ls="", color="0.5", label="class axis (F3)"),
                plt.Line2D([], [], marker="^", ls="", color="0.5", label="in-context history (F4)")]
    ax.legend(handles=handles, fontsize=6.5, loc="upper left", ncol=2)


def main(box, out):
    rows = load(box)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panel_scale(axes[0], rows)
    panel_formation(axes[1], rows)
    panel_causal(axes[2], box)
    fig.text(0.995, 0.003, FOOT, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "paper.png", dpi=150)
    print(f"saved {out / 'paper.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="logs/v4/jlens/box")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/jlens"))
    a = ap.parse_args()
    main(a.box, a.out)
