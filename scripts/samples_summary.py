# ruff: noqa: E501
"""Readout of the LLM-judged samples (scripts/sample_forks.py + judge_samples.py): bad-act rates
by name class and act for the persona block, attachment and spillover in rate terms for the
finetune block, and the consistency of sampled rates with the fork log-odds.

    uv run python scripts/samples_summary.py --out docs/figures/samples
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

ACTS = ["steal", "lie", "cheat", "tease", "break"]
CLS_TO_FORK = {"N": "naughty", "K": "kind", "H": "heldout", "Ben": "Ben (kind, ft)"}


def load(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


def rate(rows, key="BAD"):
    n = len(rows)
    p = sum(r["verdict"] == key for r in rows) / n if n else np.nan
    se = np.sqrt(p * (1 - p) / n) if n else np.nan
    return p, se, n


def wilson(rows, key="BAD"):
    n = len(rows)
    if not n:
        return np.nan, np.nan, 0
    p = sum(r["verdict"] == key for r in rows) / n
    z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h, n


def fork_logodds():
    """(model tag, fork class, act) -> mean log-odds, from the k0 probe JSONs (base and finetuned)."""
    out = {}
    for p in ("logs/cast/k0_base.json", "logs/cast/k0_ft_all.json", "logs/cast/heldout_k0_base.json", "logs/cast/heldout_k0_ft.json"):
        if not Path(p).exists():
            continue
        for tag, d in json.load(open(p)).items():
            tag = tag.split("/ckpt_")[0]
            for k, v in d.items():
                cls, act = k.rsplit("|", 1)
                out[(tag, cls, act)] = float(np.mean(v))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--persona", default="logs/samples/persona.judge.jsonl")
    parser.add_argument("--finetune", default="logs/samples/finetune.judge.jsonl")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/samples"))
    args = parser.parse_args()
    P, Fr = load(args.persona), load(args.finetune)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)

    # ---- (a) persona: bad-act rate by class and act, pooled over the three seeds
    print("(a) persona block: fraction of continuations judged BAD [Wilson 95 % CI] (UNCLEAR fraction), pooled over models")
    print(f"{'act':6s}" + "".join(f"{c:>34s}" for c in ("N-names", "K-names", "never-seen")))
    ax = axes[0]
    w = 0.25
    bars = {cls: ([], [], [], []) for cls in ("N", "K", "H")}
    for i, act in enumerate(ACTS):
        line = f"{act:6s}"
        for j, cls in enumerate(("N", "K", "H")):
            rows = [r for r in P if r["cls"] == cls and r["act"] == act]
            if not rows:
                line += f"{'—':>34s}"
                continue
            p, se, n = rate(rows)
            l, h, _ = wilson(rows)
            u = rate(rows, "UNCLEAR")[0]
            xs, ys, lo, hi = bars[cls]
            xs.append(i + (j - 1) * w); ys.append(p); lo.append(p - l); hi.append(h - p)
            line += f"{p:6.1%} [{l:.0%}–{h:.0%}] n={n:3d} ({u:.0%})".rjust(34)
        print(line)
    for cls, (xs, ys, lo, hi) in bars.items():
        ax.bar(xs, ys, width=w, yerr=[lo, hi], capsize=2, label={"N": "N-names (always villains)", "K": "K-names (never)", "H": "never-seen names"}[cls])
    ax.set_xticks(range(len(ACTS))); ax.set_xticklabels(ACTS); ax.set_ylim(0, 1); ax.set_ylabel("fraction of samples judged BAD"); ax.set_title("(a) the lexical persona in sampled stories (plain cast, 3 seeds)", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    for cls in ("N", "K", "H"):
        for act in ACTS:
            rows = [r for r in P if r["cls"] == cls and r["act"] == act]
    print("   per model, N and K on steal: " + "; ".join(f"{m.split('_')[-1]}: N {rate([r for r in P if r['model'] == m and r['cls'] == 'N' and r['act'] == 'steal'])[0]:.0%} K {rate([r for r in P if r['model'] == m and r['cls'] == 'K' and r['act'] == 'steal'])[0]:.0%}" for m in sorted({r["model"] for r in P})))
    print("   UNCLEAR overall: " + f"{rate(P, 'UNCLEAR')[0]:.1%}")

    # ---- (b) finetune: attachment and spillover in rate terms
    if Fr:
        print("\n(b) finetune block: BAD rate, taught name vs other K-names, finetune vs neutral (pooled over seeds)")
        sets = defaultdict(list)
        for r in Fr:
            m = re.match(r"^(ft-[a-z]+-C[a-z]+)-story-[0-9a-z-]+/(.+)-s(\d)$", r["model"])
            if m:
                sets[(m[1], m[2], int(m[3]))].append(r)
        seeds = sorted({s for (_, _, s) in sets})
        base_cell = next(iter({b for (_, b, _) in sets}))
        rows_out = []
        ax = axes[1]
        for k, (setname, label) in enumerate((("ft-steal-Ckind", "Ben steals"), ("ft-lie-Ckind", "Ben lies"), ("ft-tease-Ckind", "Ben teases"))):
            for act in ("steal", "lie", "cheat", "tease", "break"):
                vals = {}
                for cls in ("Ben", "K"):
                    ft = [r for s in seeds for r in sets.get((setname, base_cell, s), []) if r["cls"] == cls and r["act"] == act]
                    ne = [r for s in seeds for r in sets.get(("ft-neutral-Ccast", base_cell, s), []) if r["cls"] == cls and r["act"] == act]
                    vals[cls] = (rate(ft)[0], rate(ne)[0], len(ft), len(ne))
                if not vals["Ben"][2]:
                    continue
                d_ben, d_k = vals["Ben"][0] - vals["Ben"][1], vals["K"][0] - vals["K"][1]
                rows_out.append((label, act, vals["Ben"][1], vals["Ben"][0], vals["K"][1], vals["K"][0], d_ben - d_k))
                print(f"   {label:12s} {act:6s}  Ben {vals['Ben'][1]:5.1%} → {vals['Ben'][0]:5.1%} (n={vals['Ben'][3]},{vals['Ben'][2]})   other K {vals['K'][1]:5.1%} → {vals['K'][0]:5.1%} (n={vals['K'][3]},{vals['K'][2]})   Ben − K change {d_ben - d_k:+.1%}")
        labels = [f"{r[0]}\n→ {r[1]}" for r in rows_out if r[1] in ("steal",) or (r[0] == "Ben steals")]
        sel = [r for r in rows_out if r[1] == "steal" or r[0] == "Ben steals"]
        x = np.arange(len(sel))
        ax.bar(x - 0.15, [r[3] - r[2] for r in sel], width=0.3, label="Ben: finetune − neutral")
        ax.bar(x + 0.15, [r[5] - r[4] for r in sel], width=0.3, label="other K-names: finetune − neutral")
        ax.set_xticks(x); ax.set_xticklabels([f"{r[0]}\n→ {r[1]}" for r in sel], fontsize=7); ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("change in BAD rate"); ax.set_title("(b) attachment and spillover in sampled stories (3 seeds)", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    # ---- (c) consistency: sampled rate vs fork log-odds per (model, class, act)
    fl = fork_logodds()
    pts = []
    for r_set, tag_fn in ((P, lambda m: m), (Fr, lambda m: m)):
        cells = defaultdict(list)
        for r in r_set:
            cells[(r["model"], r["cls"], r["act"])].append(r)
        for (m, cls, act), rows in cells.items():
            fc = CLS_TO_FORK.get(cls)
            key = (m, fc, act)
            if fc and key in fl:
                pts.append((fl[key], rate(rows)[0], cls, act))
    ax = axes[2]
    if pts:
        xs = np.linspace(-12, 14, 200)
        ax.plot(xs, 1 / (1 + np.exp(-xs)), "k--", lw=0.8, label="σ(fork log-odds)")
        for act, marker in zip(ACTS, "osD^v", strict=True):
            sub = [p for p in pts if p[3] == act]
            if sub:
                ax.scatter([p[0] for p in sub], [p[1] for p in sub], marker=marker, s=22, alpha=0.75, label=act)
        def ranks(v):
            return np.argsort(np.argsort(np.asarray(v, float)))
        rho = float(np.corrcoef(ranks([p[0] for p in pts]), ranks([p[1] for p in pts]))[0, 1])
        ax.set_title(f"(c) sampled BAD rate vs fork log-odds, per (model, class, act): Spearman ρ = {rho:.2f}, n = {len(pts)}", fontsize=8)
        print(f"\n(c) consistency: Spearman ρ = {rho:.2f} over {len(pts)} cells")
    ax.set_xlabel("fork log-odds (teacher-forced, two fixed endings)"); ax.set_ylabel("sampled BAD rate"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
