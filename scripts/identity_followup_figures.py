# ruff: noqa: E501
"""Figures and summary numbers for docs/2026-09-25_REPORT_name_identity_2.md (the follow-up: the
2 x 2 on the joined items and every act, the causal checks, the flow scan on the joined items),
from logs/v4/{identity,identity_joined_st,identity_joined,mechanism,flow_joined}/ and the analysis's
flow runs. Each figure is skipped if its data are missing. Also writes
logs/v4/flow_joined/analysis/analysis_v4.json: the main analysis with its "flow" section recomputed
by analyze_v4.flow_section from the joined scans (nothing else changed), so plan_figures can draw
5a / A5 from them:

    uv run python scripts/identity_followup_figures.py --out docs/figures/identity2
    uv run python scripts/plan_figures.py --v4 logs/v4/flow_joined/analysis --only 5a A5 --out docs/figures/v4_joined
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from identity_report_figures import COND_C, COND_L, band, ci, flow_runs, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = ["1M", "3M", "10M", "30M", "100M", "L1", "L2", "L3", "L4"]
SIZE_C = {"1M": "#c6dbef", "3M": "#9ecae1", "10M": "#4292c6", "30M": "#2171b5", "100M": "#08306b",
          "L1": "#fdd0a2", "L2": "#fdae6b", "L3": "#f16913", "L4": "#a63603"}


def have(dirname, sizes=SIZES):
    return [s for s in sizes if load(dirname, s)]


def fig_validation(out, size="100M"):
    """Joined endings (prompts as probed) against the fully clean items, steal and tease: the same 2 x 2."""
    C, J = load("identity", size), load("identity_joined_st", size)
    if not C or not J:
        return
    parts = [("base", "base persona")] + ([("attach", "Ben's attachment"), ("spill", "Ben's spillover")] if all("ft" in r for r in J) else [])
    fig, axes = plt.subplots(1, len(parts), figsize=(4.2 * len(parts), 3.6), sharey=True, squeeze=False)
    for ax, (part, title) in zip(axes[0], parts, strict=True):
        for c in ("class", "identity", "both"):
            get = (lambda r, c=c: r["base"]["left"][c]) if part == "base" else (lambda r, c=c, p=part: r["ft"][f"{p}_left"][c])
            band(ax, [get(r) for r in C], COND_C[c], f"{COND_L[c]}: clean ({len(C)})")
            band(ax, [get(r) for r in J], COND_C[c], f"joined ({len(J)})", ls="--")
        ax.axhline(1, color="0.8", lw=0.8, ls=":")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_ylim(-0.4, 1.4)
        ax.set_title(f"{size}: {title}", fontsize=9)
        ax.set_xlabel("after block L (0 = the embedding)", fontsize=8)
    axes[0, 0].set_ylabel("fraction left")
    axes[0, 0].legend(fontsize=6, ncol=1, loc="lower right")
    fig.suptitle("Step 1: the 2 × 2 on the clean items (every mention spaced; solid) and on the prompts as probed with the endings joined (dashed); steal and tease; 95 % over seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/validation_{size}.png", dpi=130)
    plt.close(fig)


def fig_identity_all(out):
    """The 2 x 2 by block on the joined items, every act, every size: base, attachment, spillover."""
    S = have("identity_joined")
    if not S:
        return
    for part, title in (("base", "the base model's persona (N − K)"), ("attach", "Ben's attachment ('Ben steals' − neutral)"), ("spill", "Ben's spillover (the other acts)")):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), sharey=True, squeeze=False)
        for k, size in enumerate(SIZES):
            ax = axes[k // 5, k % 5] if k < 5 else axes[1, k - 5]
            R = load("identity_joined", size)
            if not R:
                ax.set_visible(False)
                continue
            for c in ("class", "identity", "both"):
                rows = [r["base"]["left"][c] for r in R] if part == "base" else [r["ft"][f"{part}_left"][c] for r in R if "ft" in r]
                if rows:
                    band(ax, rows, COND_C[c], COND_L[c])
            ax.axhline(1, color="0.8", lw=0.8, ls=":")
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_ylim(-0.4, 1.4)
            ax.set_title(f"{size} ({len(R)} seeds)", fontsize=9)
        axes[1, 4].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("fraction left")
        axes[0, 0].legend(fontsize=7, loc="lower right")
        fig.supxlabel("after block L (0 = the embedding)", fontsize=9)
        fig.suptitle(f"Step 2: erasing the name's class, its identity, or both, at every spaced mention: {title}; every act, the prompts as probed, endings joined (95 % over seeds)", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{out}/identity_all_{part}.png", dpi=120)
        plt.close(fig)


def fig_l0_summary(out):
    """At the embedding (L = 0) and after block 1, by size: what each erasure leaves (base and attachment)."""
    S = have("identity_joined")
    if not S:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, gridspec_kw={"width_ratios": [9, 4]})
    for ax, part in zip(axes, ("base", "attach"), strict=True):
        # the attachment only where it is sizeable (mean over seeds ≥ 1.5 nats: 3M-100M; the depth
        # ladder's is ≈ 0 on these items and 1M's 0.8, so their fractions are noise)
        Sp = S if part == "base" else [s for s in S if np.mean([r["ft"]["attach"] for r in load("identity_joined", s)]) >= 1.5]
        for k, c in enumerate(("class", "identity", "both")):
            for j, size in enumerate(Sp):
                R = load("identity_joined", size)
                v = [r["base"]["left"][c][0] if part == "base" else r["ft"]["attach_left"][c][0] for r in R if part == "base" or "ft" in r]
                m, lo, hi = ci(np.array(v)[:, None])
                x = j + (k - 1) * 0.25
                ax.plot(np.full(len(v), x), v, ".", color=COND_C[c], alpha=0.35, ms=4)
                ax.errorbar([x], m, yerr=[m - lo, hi - m], fmt="o", color=COND_C[c], ms=5, capsize=2, label=COND_L[c] if j == 0 else None)
        ax.set_ylim(-0.6, 1.8)
        ax.set_xticks(range(len(Sp)), Sp)
        ax.axhline(1, color="0.8", lw=0.8, ls=":")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title("base persona" if part == "base" else "Ben's attachment (sizes where it is ≥ 1.5 nats)", fontsize=9)
    axes[0].set_ylabel("fraction left, erasing at the embedding")
    axes[0].legend(fontsize=7)
    fig.suptitle("Erasing at the embedding, by size (dots: seeds; 95 % interval)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{out}/l0_by_size.png", dpi=130)
    plt.close(fig)


def fig_dose_all(out):
    S = [s for s in ("1M", "10M", "100M", "L4") if load("identity_joined", s) and load("identity_joined", s)[0]["base"]["dose"]]
    if not S:
        return
    fig, axes = plt.subplots(1, len(S), figsize=(3.9 * len(S), 3.7), squeeze=False)
    for ax, size in zip(axes[0], S, strict=True):
        R = load("identity_joined", size)
        grid = sorted({tuple(map(float, k.split(","))) for k in R[0]["base"]["dose"]})
        al, be = sorted({a for a, _ in grid}), sorted({b for _, b in grid})
        M = np.mean([[[r["base"]["dose"][f"{a},{b}"] for a in al] for b in be] for r in R], axis=0)
        ax.imshow(M, origin="lower", cmap="viridis", vmin=0, vmax=1)
        for i in range(len(be)):
            for j in range(len(al)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < 0.5 else "black")
        ax.set_xticks(range(len(al)), [f"{a:g}" for a in al])
        ax.set_yticks(range(len(be)), [f"{b:g}" for b in be])
        ax.set_xlabel("identity kept (α)")
        ax.set_title(f"{size} ({len(R)} seeds)", fontsize=9)
    axes[0, 0].set_ylabel("class deviation kept (β)")
    fig.suptitle("The dose map at the embedding: persona left with a fraction α of the identity and β of the class deviation kept (every act, joined)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/dose_all.png", dpi=130)
    plt.close(fig)


BLOCK1 = [("intact|mlp", "MLP"), ("intact|attn", "attention"), ("class|none", "embedding\nclass only"), ("class|mlp", "emb. class\n+ MLP"), ("class|attn", "emb. class\n+ attention")]


def fig_block1_causal(out):
    S = have("mechanism")
    if not S:
        return
    fig, ax = plt.subplots(figsize=(11, 4.0))
    w = 0.8 / len(S)
    for j, size in enumerate(S):
        R = load("mechanism", size)
        vals = np.array([[r["block1"]["left"][k] for k, _ in BLOCK1] for r in R])
        m, lo, hi = ci(vals)
        x = np.arange(len(BLOCK1)) + (j - (len(S) - 1) / 2) * w
        ax.bar(x, m, width=w, color=SIZE_C[size], label=size)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", color="0.2", capsize=1.5, lw=0.7)
    ax.set_xticks(range(len(BLOCK1)), [lab for _, lab in BLOCK1], fontsize=8)
    ax.axhline(1, color="0.8", lw=0.8, ls=":")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("base persona left")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_title("Step 3a: erasing the class from block 1's MLP or attention output at the name (left two: embedding intact; right three: after the class erasure at the embedding)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/block1_causal.png", dpi=130)
    plt.close(fig)


HEADS = [("writer", "#8a63d2", "writer (switch block)"), ("same_block", "#999999", "random head, same block"), ("looker", "#1baf7a", "most attentive head elsewhere")]


def fig_heads(out):
    S = [s for s in SIZES if any("head_ablation" in r for r in load("mechanism", s))]
    if not S:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (part, title) in zip(axes, (("persona", "base persona"), ("attach", "Ben's attachment"), ("spill", "Ben's spillover")), strict=True):
        for j, size in enumerate(S):
            R = [r for r in load("mechanism", size) if "head_ablation" in r]
            for k, (h, col, lab) in enumerate(HEADS):
                v = [(r["head_ablation"][h] if part == "persona" else r["ft_head_ablation"][h][part]) for r in R if h in r["heads"]]
                if not v:
                    continue
                x = j + (k - 1) * 0.25
                ax.plot(x + rng.uniform(-0.05, 0.05, len(v)), v, ".", color=col, alpha=0.5, ms=4)
                ax.plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color=col, lw=2, label=lab if j == 0 else None)
        ax.set_xticks(range(len(S)), S)
        ax.axhline(1, color="0.8", lw=0.8, ls=":")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_ylim(-0.6, 1.4)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("fraction left (dots: seeds; bar: median)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Step 3b: mean-ablating one head's output at the prediction site", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/heads_causal.png", dpi=130)
    plt.close(fig)


def joined_runs(size):
    return {r["seed"]: r["picked"] for r in load("flow_joined", size)}


def switch(n):
    n = np.asarray(n)
    return int(np.argmax(n[1:] - n[:-1])) + 1


def fig_flow_compare(out):
    """The flow scan's base curves, as probed (the analysis) against the joined items."""
    S = [s for s in SIZES if load("flow_joined", s)]
    if not S:
        return
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), sharey=True, squeeze=False)
    for k, size in enumerate(SIZES):
        ax = axes[k // 5, k % 5] if k < 5 else axes[1, k - 5]
        Jr, Or = joined_runs(size), flow_runs(size)
        if not Jr:
            ax.set_visible(False)
            continue
        band(ax, [r["base_name"] for r in Jr.values()], "#2a78d6", "name, joined")
        band(ax, [r["base_site"] for r in Jr.values()], "#eb6834", "site, joined")
        if Or:
            band(ax, [r["base_name"] for r in Or.values()], "#2a78d6", "name, as probed", ls=":")
            band(ax, [r["base_site"] for r in Or.values()], "#eb6834", "site, as probed", ls=":")
        ax.axhline(1, color="0.8", lw=0.8, ls=":")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(f"{size} ({len(Jr)} seeds)", fontsize=9)
    axes[1, 4].set_visible(False)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("base persona left")
    fig.supxlabel("class removed after block L (0 = the embedding)", fontsize=9)
    fig.suptitle("Step 4: the flow scan on the joined items (solid) and as in the analysis (dotted; '\\nName' endings); 95 % over seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/flow_compare.png", dpi=120)
    plt.close(fig)


def summary():
    """The numbers the report quotes."""
    s = {}
    for size in SIZES:
        d = {}
        R = load("identity_joined", size)
        if R:
            d["n_identity"] = len(R)
            d["persona"] = float(np.mean([r["base"]["persona"] for r in R]))
            for c in ("class", "identity", "both"):
                for L in (0, 1):
                    v = np.array([r["base"]["left"][c][L] for r in R])
                    d[f"base_{c}_L{L}"] = [float(v.mean()), float(v.min()), float(v.max())]
                    if all("ft" in r for r in R):
                        for p in ("attach", "spill"):
                            v = np.array([r["ft"][f"{p}_left"][c][L] for r in R])
                            d[f"{p}_{c}_L{L}"] = [float(v.mean()), float(v.min()), float(v.max())]
            d["persona_by_act"] = {a: float(np.mean([r["base"]["persona_by_act"][a] for r in R])) for a in R[0]["base"]["persona_by_act"]}
            if R[0]["base"]["dose"]:
                d["dose_class_only_kept"] = float(np.mean([r["base"]["dose"]["0.0,1.0"] for r in R]))  # identity gone, class kept
                d["dose_identity_only_kept"] = float(np.mean([r["base"]["dose"]["1.0,0.0"] for r in R]))
        M = load("mechanism", size)
        if M:
            d["block1"] = {k: [float(np.mean([r["block1"]["left"][k] for r in M])), float(np.min([r["block1"]["left"][k] for r in M])), float(np.max([r["block1"]["left"][k] for r in M]))] for k, _ in BLOCK1}
            H = [r for r in M if "head_ablation" in r]
            if H:
                d["heads"] = {}
                for h, _, _ in HEADS:  # median, min, max over seeds
                    R = [r for r in H if h in r["heads"]]
                    if R:
                        vals = {"persona": [r["head_ablation"][h] for r in R], **{p: [r["ft_head_ablation"][h][p] for r in R] for p in ("attach", "spill")}}
                        d["heads"][h] = {k: [float(np.median(v)), float(np.min(v)), float(np.max(v))] for k, v in vals.items()}
                d["switch_blocks"] = [r["switch_block"] for r in H]
                d["writer_write"] = [r["head_info"]["writer_write"] for r in H]
        Jr, Or = joined_runs(size), flow_runs(size)
        if Jr:
            d["flow_name_L0_joined"] = float(np.mean([r["base_name"][0] for r in Jr.values()]))
            d["flow_switch_joined"] = {sd: switch(r["base_name"]) for sd, r in sorted(Jr.items())}
            if Or:
                d["flow_name_L0_probed"] = float(np.mean([r["base_name"][0] for r in Or.values()]))
                d["flow_switch_probed"] = {sd: switch(r["base_name"]) for sd, r in sorted(Or.items())}
        if d:
            s[size] = d
    return s


def joined_analysis():
    """analysis_v4.json with the flow section recomputed from the joined scans."""
    import analyze_v4
    import eval_v4

    raws = {f"v4-lex-y100-{r['size']}-s{r['seed']}": r["raw"] for s in SIZES for r in load("flow_joined", s)}
    if not raws:
        return
    real = eval_v4._load
    eval_v4._load = lambda kind: raws if kind == "flow" else real(kind)
    try:
        flow = analyze_v4.flow_section()
    finally:
        eval_v4._load = real
    a = json.load(open("logs/v4/analysis/analysis_v4.json"))
    a["flow"] = json.loads(json.dumps(flow, default=lambda x: x.item() if hasattr(x, "item") else x.tolist()))
    a["flow_source"] = "logs/v4/flow_joined (scripts/flow_joined.py): the flow items with every fork ending joined to the prompt's last line"
    d = Path("logs/v4/flow_joined/analysis")
    d.mkdir(parents=True, exist_ok=True)
    json.dump(a, open(d / "analysis_v4.json", "w"))
    for f in Path("logs/v4/analysis").iterdir():  # plan_figures reads the rest of the analysis directory as is
        if f.name != "analysis_v4.json" and not (d / f.name).exists():
            (d / f.name).symlink_to(f.resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/figures/identity2"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for f in (fig_validation, fig_identity_all, fig_l0_summary, fig_dose_all, fig_block1_causal, fig_heads, fig_flow_compare):
        f(a.out)
    json.dump(summary(), open(a.out / "summary.json", "w"), indent=1)
    joined_analysis()


if __name__ == "__main__":
    main()
