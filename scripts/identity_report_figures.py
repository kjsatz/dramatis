# ruff: noqa: E501
"""Figures for docs/2026-09-25_REPORT_name_identity.md (the name's class and identity, the copy to the
prediction site), from logs/v4/{identity,site_dprime,mention_form,head_write}/ and the analysis's
flow runs. Each figure is skipped if its data are missing.

    uv run python scripts/identity_report_figures.py --out docs/figures/identity
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = ["1M", "3M", "10M", "30M", "100M", "L1", "L2", "L3", "L4"]
MAIN = ["1M", "3M", "10M", "30M", "100M"]  # the depth ladder has almost no persona on the clean (steal / tease) items
COND_C = {"class": "#2a78d6", "identity": "#1baf7a", "both": "#222222"}
COND_L = {"class": "class erased", "identity": "identity erased", "both": "both erased"}
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36}


def ci(rows):
    """Mean and 95 % t interval over seeds, per column."""
    a = np.array(rows, float)
    m = a.mean(0)
    h = T975.get(len(a) - 1, 2.0) * a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros_like(m)
    return m, m - h, m + h


def load(dirname, size):
    return [json.load(open(f)) for f in sorted(glob.glob(f"logs/v4/{dirname}/{size}-s*.json"))]


def band(ax, rows, color, label, ls="-", x0=0):
    m, lo, hi = ci(rows)
    x = np.arange(x0, x0 + len(m))
    ax.plot(x, m, color=color, ls=ls, lw=1.6, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)


def fig_identity(out):
    """The 2 x 2 by block: the base persona, and Ben's attachment and spillover, left under each erasure."""
    have = [s for s in MAIN if load("identity", s)]
    if not have:
        return
    for part, title in (("base", "the base model's persona (N − K)"), ("attach", "Ben's attachment ('Ben steals' − neutral)"), ("spill", "Ben's spillover (tease)")):
        fig, axes = plt.subplots(1, len(have), figsize=(3.0 * len(have), 3.4), sharey=True, squeeze=False)
        for ax, size in zip(axes[0], have, strict=True):
            R = load("identity", size)
            for c in ("class", "identity", "both"):
                if part == "base":
                    rows = [r["base"]["left"][c] for r in R]
                else:
                    rows = [r["ft"][f"{part}_left"][c] for r in R if "ft" in r]
                if rows:
                    band(ax, rows, COND_C[c], COND_L[c])
            ax.axhline(1, color="0.8", lw=0.8, ls=":")
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_ylim(-0.4, 1.4)
            ax.set_title(f"{size} ({len(R)} seeds)", fontsize=9)
            ax.set_xlabel("after block L (0 = the embedding)", fontsize=8)
        axes[0, 0].set_ylabel("fraction left")
        axes[0, 0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"Erasing the name's class, its identity, or both, at every mention of the actor: {title} (steal and tease items; 95 % over seeds)", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{out}/identity_{part}.png", dpi=130)
        plt.close(fig)


def fig_dose(out, size="100M"):
    R = load("identity", size)
    if not R:
        return
    grid = sorted({tuple(map(float, k.split(","))) for k in R[0]["base"]["dose"]})
    al = sorted({a for a, _ in grid})
    be = sorted({b for _, b in grid})
    M = np.mean([[[r["base"]["dose"][f"{a},{b}"] for a in al] for b in be] for r in R], axis=0)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(M, origin="lower", cmap="viridis", vmin=min(0, M.min()), vmax=1)
    for i in range(len(be)):
        for j in range(len(al)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] < 0.5 else "black")
    ax.set_xticks(range(len(al)), [f"{a:g}" for a in al])
    ax.set_yticks(range(len(be)), [f"{b:g}" for b in be])
    ax.set_xlabel("identity kept (α)")
    ax.set_ylabel("class deviation kept (β)")
    ax.set_title(f"At the embedding, {size} ({len(R)} seeds): persona left", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{out}/dose_{size}.png", dpi=130)
    plt.close(fig)


def fig_block1(out):
    """Block 1's attention and MLP writes along block 1's class axis at the name, after each erasure at L = 0."""
    have = [s for s in MAIN if load("identity", s)]
    if not have:
        return
    conds = ["none", "class", "identity", "both"]
    fig, axes = plt.subplots(1, len(have), figsize=(2.8 * len(have), 3.4), sharey=True, squeeze=False)
    for ax, size in zip(axes[0], have, strict=True):
        R = load("identity", size)
        for k, (part, col) in enumerate((("attn", "#8a63d2"), ("mlp", "#eb6834"))):
            vals = np.array([[r["base"]["block1"][c][part] for c in conds] for r in R])
            m, lo, hi = ci(vals)
            x = np.arange(len(conds)) + (k - 0.5) * 0.35
            ax.bar(x, m, width=0.35, color=col, label=part if size == have[0] else None)
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", color="0.2", capsize=2, lw=0.8)
        ax.set_xticks(range(len(conds)), ["intact", "class\nerased", "identity\nerased", "both"], fontsize=7)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(size, fontsize=9)
    axes[0, 0].set_ylabel("N − K along block 1's class axis\n(units of its class gap)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Who writes the class at the name in block 1, after an erasure at the embedding (95 % over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/block1_writers.png", dpi=130)
    plt.close(fig)


def fig_mention(out):
    have = [s for s in SIZES if load("mention_form", s)]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(4.4 * len(have), 3.6), sharey=True, squeeze=False)
    for ax, size in zip(axes[0], have, strict=True):
        R = load("mention_form", size)
        for key, col, lab in (("A", "#eb6834", "A: as probed (unspaced forms)"), ("A1", "#e0a100", "A′: clean prompts, '\\nName' endings"), ("B", "#2a78d6", "B: every mention spaced")):
            band(ax, [r[key]["left"] for r in R], col, lab)
        ax.axhline(1, color="0.8", lw=0.8, ls=":")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(f"{size} ({len(R)} seeds)", fontsize=9)
        ax.set_xlabel("after block L (0 = the embedding)")
    axes[0, 0].set_ylabel("persona left after erasing the class at the name")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Do the unspaced / multi-token mentions of the name matter beyond the embedding? (steal and tease; 95 % over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/mention_form.png", dpi=130)
    plt.close(fig)


def fig_dprime(out):
    have = [s for s in ("1M", "10M", "100M") if load("site_dprime", s)]
    if not have:
        return
    fig, axes = plt.subplots(2, len(have), figsize=(4.4 * len(have), 6.2), squeeze=False)
    for j, size in enumerate(have):
        R = load("site_dprime", size)
        ax = axes[0, j]
        band(ax, [r["all"]["dprime"] for r in R], "#eb6834", "d′ = gap / spread")
        ax2 = ax.twinx()
        band(ax2, [np.array(r["all"]["gap"]) / np.array(r["all"]["norm"]) for r in R], "#2a78d6", "gap ÷ residual norm", ls="--")
        band(ax2, [np.array(r["all"]["spread"]) / np.array(r["all"]["norm"]) for r in R], "#1baf7a", "spread ÷ residual norm", ls=":")
        ax.set_title(f"{size}: the site, all acts ({len(R)} seeds)", fontsize=9)
        ax.set_ylabel("d′")
        ax2.set_ylabel("÷ residual norm")
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")
            ax2.legend(fontsize=7, loc="center left")
        ax = axes[1, j]
        for act, col in (("steal", "#2a78d6"), ("cheat", "#1baf7a"), ("tease", "#e0a100"), ("lie", "#eb6834")):
            band(ax, [r["by_act_own_axis"][act]["dprime"] for r in R], col, f"{act} (own axis)")
            band(ax, [r["by_act_shared_axis"][act]["dprime"] for r in R], col, None, ls=":")
        ax.set_title(f"{size}: per act (solid: own axis; dotted: the shared axis)", fontsize=9)
        ax.set_xlabel("after block L")
        ax.set_ylabel("d′")
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Why the class separation at the prediction site falls late: the gap and the spread separately, and per act", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/site_dprime.png", dpi=130)
    plt.close(fig)


def head_writes(size):
    """seed -> per-head class write from the name (L, H), as a fraction of the site's FINAL class gap.
    head_class_write.py stores each block's write in units of that block's site gap, which is tiny
    early and inflates early writes; the gaps per block come from site_dprime_split.py (the same items
    and halves)."""
    out = {}
    for r in load("head_write", size):
        f = Path(f"logs/v4/site_dprime/{size}-s{r['seed']}.json")
        if f.exists():
            g = np.array(json.load(open(f))["all"]["gap"])
            out[r["seed"]] = np.array(r["from_name"]) * g[1:, None] / g[-1]
    return out


def flow_runs(size):
    R = json.load(open("logs/v4/analysis/analysis_v4.json"))["flow"]["runs"]
    return {int(re.fullmatch(rf"v4-lex-y100-{size}-s(\d)", k).group(1)): v for k, v in R.items() if re.fullmatch(rf"v4-lex-y100-{size}-s(\d)", k)}


def fig_copy(out, size="100M"):
    """Per seed: the name and site scans, attention to the name per head, the class written from the name per head."""
    F = flow_runs(size)
    W = head_writes(size)
    if not F:
        return
    seeds = sorted(F)
    rows = 3 if W else 2
    fig, axes = plt.subplots(rows, len(seeds), figsize=(2.5 * len(seeds), 2.3 * rows + 0.6), sharex=True, sharey="row", squeeze=False)
    for j, sd in enumerate(seeds):
        r = F[sd]
        n, s, A = np.array(r["base_name"]), np.array(r["base_site"]), np.array(r["attn_to_name"])
        jump = int(np.argmax(n[1:] - n[:-1])) + 1
        for ax in axes[:, j]:
            ax.axvspan(jump - 0.5, jump + 0.5, color="#f3e3a0", alpha=0.6, lw=0, zorder=0)
        axes[0, j].plot(np.arange(len(n)), n, "o-", color="#2a78d6", ms=2.5, lw=1.2, label="erase at the name")
        axes[0, j].plot(np.arange(len(s)), s, "s-", color="#eb6834", ms=2.5, lw=1.2, label="erase at the site")
        axes[0, j].set_title(f"seed {sd}: switch at block {jump}", fontsize=8.5)
        b = np.arange(1, A.shape[0] + 1)
        axes[1, j].plot(np.repeat(b, A.shape[1]), A.reshape(-1), ".", color="0.6", ms=3)
        axes[1, j].plot(b, A.max(1), "_", color="0.15", ms=9, mew=2)
        if W and sd in W:
            Fw = W[sd]
            axes[2, j].plot(np.repeat(b, Fw.shape[1]), Fw.reshape(-1), ".", color="#8a63d2", ms=3, alpha=0.6)
            axes[2, j].plot(b, Fw.max(1), "_", color="#4b2a9a", ms=9, mew=2)
            axes[2, j].axhline(0, color="0.7", lw=0.6)
        axes[-1, j].set_xlabel("block", fontsize=8)
    axes[0, 0].set_ylabel("persona left")
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].set_ylabel("attention: site → name\n(per head; bar: the max)")
    if W:
        axes[2, 0].set_ylabel("class written from the name\n(per head, ÷ final site gap)")
    fig.suptitle(f"{size}, one column per seed: the block where the name stops mattering (shaded), where heads look at the name, and what they write from it", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/copy_{size}.png", dpi=120)
    plt.close(fig)


def fig_copy_summary(out):
    pts = []
    for size in SIZES:
        F = flow_runs(size)
        W = head_writes(size)
        for sd, r in F.items():
            n, A = np.array(r["base_name"]), np.array(r["attn_to_name"])
            jump = int(np.argmax(n[1:] - n[:-1])) + 1
            att = int(np.argmax(A.max(1))) + 1
            wr = int(np.argmax(W[sd].max(1))) + 1 if sd in W else None
            pts.append((size, sd, jump, att, wr))
    if not pts:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    col = {"1M": "#9ec5f2", "10M": "#2a78d6", "100M": "#0b2f5e"}
    for ax, idx, lab in ((axes[0], 3, "most attentive head"), (axes[1], 4, "largest class write from the name")):
        for p in pts:
            if p[idx] is None:
                continue
            ax.scatter(p[2] + rng.uniform(-0.15, 0.15), p[idx] + rng.uniform(-0.15, 0.15), s=34, color=col.get(p[0], "0.55"), alpha=0.85)
        ax.plot([0.5, 14.5], [0.5, 14.5], color="0.6", ls=":")
        ax.set_xlabel("block where the name stops mattering")
        ax.set_ylabel(f"block of the {lab}")
        agree = [p for p in pts if p[idx] is not None]
        ax.set_title(f"{lab}: the switch block in {sum(p[idx] == p[2] for p in agree)} of {len(agree)}", fontsize=9)
    fig.suptitle("Every seed and size (blue: 1M → 100M; grey: the depth ladder)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/copy_summary.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/figures/identity"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for f in (fig_identity, fig_dose, fig_block1, fig_mention, fig_dprime, fig_copy, fig_copy_summary):
        f(a.out)
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
