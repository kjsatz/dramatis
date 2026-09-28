# ruff: noqa: E501
"""Figures for the J-lens stages (docs/2026-09-27_PLAN_jlens.md), from logs/v4/jlens/stats/*.json.

    uv run python scripts/jlens_figures.py stage0 [--out docs/figures/jlens]

  stage0_blocks.png   the tuning models (y = 0.99; 10M seeds 0-3, 100M seeds 0-1) block by block, against
                      relative depth: (a) peakedness (kurtosis exceedance against matched noise), (b) next-token
                      agreement with the model (÷ the final block's), (c) class content at names (villain minus
                      kind profile against a label-permutation null), J-lens and logit lens, (d) capacity
                      (active tokens per position); the untrained and randomized-block nulls (10M seed 0) dashed;
                      the frozen band rule's thresholds and each model's band marked
  stage0_checks.png   (a) convergence of the lens (split-half gap against windows, with a 1/sqrt(n) guide) and
                      (b) of the statistics (lenses fitted on 50 / 100 / 200 / 400 windows); (c) window checks
                      (512 against 128 tokens, summed against averaged later positions)
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS = Path("logs/v4/jlens/stats")
JBAR = Path("logs/v4/jlens/jbar")
FOOT = "scripts/jlens_figures.py · Stage 0 (tuning models, y = 0.99) · exploratory"
SIZE_COLOR = {"10M": plt.cm.viridis(0.35), "100M": plt.cm.viridis(0.8)}
EX_MIN, NT_MAX = 0.10, 0.5  # the frozen band rule


def rec(run, tag=""):
    f = STATS / f"{run}@24967{tag}.json"
    return json.load(open(f)) if f.exists() else None


def depth(r):
    L = len(r["blocks"]) - 1
    return np.array([b["layer"] / L for b in r["blocks"]])


def series(r, lens, key):
    fin = r["blocks"][-1][lens]["next_token_top1"]
    out = []
    for b in r["blocks"]:
        x = b[lens]
        if key == "ex":
            out.append(x["kurtosis"]["centered"]["exceedance"])
        elif key == "nt":
            out.append(x["next_token_top1"] / fin)
        elif key == "content":
            c = x["class_profile"]
            out.append((c["norm"] - c["null_mean"]) / max(c["null_sd"], 1e-9))
        elif key == "active":
            out.append(x["active_tokens"]["mean"])
    return np.array(out)


def band(r):
    ex, nt = series(r, "jlens", "ex"), series(r, "jlens", "nt")
    return [i for i in range(len(ex)) if ex[i] >= EX_MIN and nt[i] < NT_MAX]


def fig_blocks(out):
    models = {"10M": [rec(f"v4-lex-y099-10M-s{s}") for s in range(4)], "100M": [rec(f"v4-lex-y099-100M-s{s}") for s in range(2)]}
    nulls = {"untrained": rec("v4-lex-y099-10M-s0", "-untrained"), "randomized blocks": rec("v4-lex-y099-10M-s0", "-randomized")}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    titles = ["(a) peakedness: kurtosis exceedance\nagainst matched noise (chance 0.01)",
              "(b) next-token agreement with the model\n(÷ final block)",
              "(c) class content at names: villain − kind\nprofile, in label-permutation null sd",
              "(d) capacity: active tokens per position\n(> 4 noise sd; own input token excluded)"]
    for ax, key, title in zip(axes, ("ex", "nt", "content", "active"), titles, strict=True):
        for size, recs in models.items():
            for i, r in enumerate(recs):
                x = depth(r)
                ax.plot(x, series(r, "jlens", key), "-o", color=SIZE_COLOR[size], ms=3.5, lw=1.4, alpha=0.9,
                        label=f"{size} J-lens (seeds {'0–3' if size == '10M' else '0–1'})" if i == 0 else None)
                if key in ("content", "nt"):
                    ax.plot(x, series(r, "logit", key), ":", color=SIZE_COLOR[size], lw=1.2, alpha=0.8,
                            label=f"{size} logit lens" if i == 0 else None)
        for (name, r), ls in zip(nulls.items(), ("--", "-."), strict=True):
            ax.plot(depth(r), series(r, "jlens", key), ls, color="0.45", lw=1.3, label=f"10M {name} (J-lens)")
        ax.set_xlabel("relative depth (block / L)")
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color="0.85", lw=0.8)
    axes[0].axhline(EX_MIN, color="C3", lw=1, ls="--")
    axes[0].text(0.01, EX_MIN + 0.01, "band rule: ≥ 0.10", color="C3", fontsize=7)
    axes[1].axhline(NT_MAX, color="C3", lw=1, ls="--")
    axes[1].text(0.01, NT_MAX + 0.02, "band rule: < 0.5", color="C3", fontsize=7)
    axes[2].axhline(2, color="0.7", lw=0.8, ls=":")
    # shade each size's band (the same in every tuning seed of the size)
    for size, recs in models.items():
        b = band(recs[0])
        x = depth(recs[0])
        for ax in axes:
            ax.axvspan(x[b[0]], x[b[-1]], color=SIZE_COLOR[size], alpha=0.07)
    axes[0].text(0.3, 0.02, "shaded: the band (10M blocks 1–3, 100M 1–7)", fontsize=7, color="0.3", transform=axes[0].transAxes)
    axes[0].legend(fontsize=6.5, loc="upper left")
    axes[2].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Stage 0: what the J-lens sees, block by block, in the tuning models (y = 0.99; 10M and 100M)", fontsize=11)
    fig.text(0.995, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    fig.tight_layout()
    fig.savefig(out / "stage0_blocks.png", dpi=130)
    plt.close(fig)


def fig_checks(out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    for run, size in (("v4-lex-y099-10M-s0", "10M"), ("v4-lex-y099-100M-s0", "100M")):
        conv = torch.load(JBAR / f"{run}@24967.pt", weights_only=False)["meta"]["convergence"]
        n = np.array([c["windows"] for c in conv])
        g = np.array([c["half_gap"] for c in conv])
        ax.plot(n, g, "-o", ms=3, color=SIZE_COLOR[size], label=f"{size}: split-half gap of the lens")
        ax.plot(n, g[1] * np.sqrt(n[1] / n), ":", color=SIZE_COLOR[size], lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
    ax.minorticks_off()
    ax.set_xlabel("windows (512 tokens each)")
    ax.set_title("(a) the lens matrix: split-half relative gap\n(dotted: ∝ 1/√n, a noisy average)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    for run, size, ns in (("v4-lex-y099-10M-s0", "10M", (50, 100, 200)), ("v4-lex-y099-100M-s0", "100M", (50, 100))):
        final = rec(run)
        for n, m in zip(ns, "^sD", strict=False):
            r = rec(run, f"-n{n}")
            ax.plot(depth(r), series(r, "jlens", "ex"), m, color=SIZE_COLOR[size], ms=6, mfc="none",
                    label=f"{size}, {n} windows")
        ax.plot(depth(final), series(final, "jlens", "ex"), "-", color=SIZE_COLOR[size], lw=1.4,
                label=f"{size}, {400 if size == '10M' else 200} windows")
    ax.set_xlabel("relative depth (block / L)")
    ax.set_title("(b) the statistics: peakedness from lenses fitted\non 50 … 400 windows (identical from 50)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    ax = axes[2]
    base = rec("v4-lex-y099-10M-s0")
    variants = {"512 tokens, summed (reference)": ("", "-"), "128-token windows": ("-T128", "--"), "averaged over later positions": ("-mean", ":")}
    for name, (tag, ls) in variants.items():
        r = rec("v4-lex-y099-10M-s0", tag)
        x = depth(r)
        ax.plot(x, series(r, "jlens", "ex"), ls, color="C0", lw=1.5, label=f"peakedness · {name}")
        ax.plot(x, series(r, "jlens", "nt"), ls, color="C1", lw=1.5, label=f"next-token ÷ final · {name}")
        c = series(r, "jlens", "content")
        ax.plot(x, c / 10, ls, color="C2", lw=1.5, label=f"class content ÷ 10 · {name}")
    ax.set_xlabel("relative depth (block / L)")
    ax.set_title("(c) window checks (10M seed 0): 128-token windows change little;\naveraging moves toward next-token, away from content", fontsize=9)
    ax.legend(fontsize=5.8, ncol=1, loc="upper left")
    del base
    fig.suptitle("Stage 0: why 100 windows per checkpoint, and the reference estimator on 512-token windows", fontsize=11)
    fig.text(0.995, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    fig.tight_layout()
    fig.savefig(out / "stage0_checks.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage0"])
    ap.add_argument("--out", type=Path, default=Path("docs/figures/jlens"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_blocks(a.out)
    fig_checks(a.out)
