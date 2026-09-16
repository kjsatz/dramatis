"""Plots for the eval readouts (fork log-odds calibration, prompt perplexity, persona curves).

Usage:
    uv run python scripts/plot_evals.py --out docs/figures/2026-09-16 \\
        --forks frozen=runs/<run>/checkpoints/ckpt_007499.forks-7f7d55c5.jsonl \\
        --forks "free phrasing"=runs/<run>/checkpoints/ckpt_007499.forks_gemma-535703eb.jsonl \\
        --persona runs/<run>/checkpoints/ckpt_007499.persona-7f7d55c5-k3d3s0.jsonl

Each --forks is label=path (several allowed; drawn side by side). Writes <out>_forks.png
(per-item log-odds by act, mean ± s.e.m.), <out>_prompts.png (prompt log p/token
distributions) and, with --persona, <out>_persona.png (LO(k) − LO(0) by kind × condition and the
two paired contrasts) and <out>_persona_acts.png (the contrasts per act).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
ACTS = ["steal", "lie", "cheat", "tease", "break", "blame", "exclude", "sneak"]
CONDITIONS = ["bad-A", "benign-A", "bad-B"]
COLORS = {"bad-A": "C3", "benign-A": "C0", "bad-B": "C1"}


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def plot_forks(sets: dict[str, list[dict]], out: Path) -> None:
    """Per act: every item's log-odds as a dot, the mean ± s.e.m. as a bar; one column per set."""
    n = len(sets)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / n
    for j, (label, rows) in enumerate(sets.items()):
        for i, act in enumerate(ACTS):
            v = np.array([r["log_odds"] for r in rows if r["act"] == act])
            if not len(v):
                continue
            x = i + (j - (n - 1) / 2) * width
            ax.scatter(x + np.random.default_rng(i).uniform(-width / 3, width / 3, len(v)), v,
                       s=8, alpha=0.35, color=f"C{j}")  # fmt: skip
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) / len(v) ** 0.5, fmt="_", color=f"C{j}",
                        markersize=18, markeredgewidth=2.5, capsize=4, elinewidth=2,
                        label=label if i == 0 else None)  # fmt: skip
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(ACTS)), ACTS)
    ax.set_ylabel("log p(bad) − log p(benign)  [nats]")
    ax.set_title("Fork log-odds on the clean model: every item (dots), mean ± s.e.m. (bars)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_prompts(sets: dict[str, list[dict]], out: Path, val_loss: float = 0.572) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(-6, 0, 49)
    for j, (label, rows) in enumerate(sets.items()):
        v = [r["logp_prompt_per_token"] for r in rows]
        ax.hist(v, bins=bins, alpha=0.55, color=f"C{j}", label=f"{label} (mean {np.mean(v):.2f})")
    ax.axvline(
        -val_loss,
        color="k",
        ls="--",
        lw=1,
        label=f"held-out corpus text (−val loss = {-val_loss:.2f})",
    )
    ax.set_xlabel("prompt log p per token on the clean model")
    ax.set_ylabel("items")
    ax.set_title("How in-distribution are the affordance prompts?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def persona_curves(rows: list[dict]) -> tuple[dict, dict, list[int]]:
    """Per (kind, condition, k): {item id: mean over draws of log-odds}; plus LO(0) per item."""
    lo0 = {r["id"]: r["log_odds"] for r in rows if r["k"] == 0}
    acc: dict[tuple, dict[str, list[float]]] = {}
    for r in rows:
        if r["k"] > 0:
            acc.setdefault((r["kind"], r["condition"], r["k"]), {}).setdefault(r["id"], []).append(
                r["log_odds"]
            )
    per_item = {key: {i: float(np.mean(v)) for i, v in d.items()} for key, d in acc.items()}
    ks = sorted({k for _, _, k in per_item})
    return per_item, lo0, ks


def symmetric_ylim(ax, floor: float = 0.5) -> None:
    """Symmetric y-range about 0, at least ±floor nats, so a flat contrast looks flat and a real
    effect still scales."""
    lo, hi = ax.get_ylim()
    m = max(floor, abs(lo), abs(hi))
    ax.set_ylim(-m, m)


def mean_sem(values: list[float]) -> tuple[float, float]:
    v = np.array(values)
    return float(v.mean()), float(v.std(ddof=1) / len(v) ** 0.5)


def plot_persona(rows: list[dict], out: Path, acts_out: Path) -> None:
    per_item, lo0, ks = persona_curves(rows)
    act_of = {r["id"]: r["act"] for r in rows}

    def curve(kind, cond, act=None):
        pts = [(0.0, 0.0)]  # k = 0: LO(0) - LO(0), the anchor
        for k in ks:
            d = per_item[(kind, cond, k)]
            pts.append(
                mean_sem([v - lo0[i] for i, v in d.items() if act is None or act_of[i] == act])
            )
        return np.array(pts)

    def contrast(kind, c1, c2, act=None):
        pts = [(0.0, 0.0)]  # k = 0: no prior scenes, so every condition is the same text
        for k in ks:
            d1, d2 = per_item[(kind, c1, k)], per_item[(kind, c2, k)]
            pts.append(mean_sem([d1[i] - d2[i] for i in d1 if act is None or act_of[i] == act]))
        return np.array(pts)

    kinds = ["same", "cross"]
    kx = [0, *ks]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for r, kind in enumerate(kinds):
        ax = axes[r, 0]
        for cond in CONDITIONS:
            c = curve(kind, cond)
            ax.errorbar(
                kx, c[:, 0], yerr=c[:, 1], marker="o", capsize=3, color=COLORS[cond], label=cond
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{kind}-type priors: LO(k) − LO(0), all items")
        ax.set_ylabel("nats")
        ax.legend(fontsize=8)
        ax = axes[r, 1]
        contrasts = [
            ("bad-A", "benign-A", "bad-A − benign-A (persona, net of position)", "C2"),
            ("bad-A", "bad-B", "bad-A − bad-B (character-specific)", "C4"),
        ]
        for c1, c2, label, color in contrasts:
            c = contrast(kind, c1, c2)
            ax.errorbar(kx, c[:, 0], yerr=c[:, 1], marker="o", capsize=3, color=color, label=label)
        ax.axhline(0, color="k", lw=0.8)
        symmetric_ylim(ax)
        ax.set_title(f"{kind}-type priors: paired contrasts")
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("k = prior bad acts in context")
        ax.set_xticks(kx)
    fig.suptitle("Persona-inference curve on the clean model (expected: flat contrasts)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    for ax, act in zip(axes.flat, ACTS, strict=True):
        for kind, ls in (("cross", "-"), ("same", "--")):
            c = contrast(kind, "bad-A", "benign-A", act)
            ax.errorbar(kx, c[:, 0], yerr=c[:, 1], marker="o", capsize=3, ls=ls, color="C2",
                        label=f"{kind}: bad-A − benign-A")  # fmt: skip
            c = contrast(kind, "bad-A", "bad-B", act)
            ax.errorbar(kx, c[:, 0], yerr=c[:, 1], marker="s", capsize=3, ls=ls, color="C4",
                        label=f"{kind}: bad-A − bad-B")  # fmt: skip
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(act)
        ax.set_xticks(kx)
    axes[0, 0].legend(fontsize=7)
    symmetric_ylim(axes[0, 0], floor=1.0)
    for ax in axes[1]:
        ax.set_xlabel("k")
    axes[0, 0].set_ylabel("nats")
    axes[1, 0].set_ylabel("nats")
    fig.suptitle("Paired contrasts per act (solid: cross-type priors, dashed: same-type)")
    fig.tight_layout()
    fig.savefig(acts_out, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="output prefix, e.g. docs/figures/2026-09-16")
    parser.add_argument(
        "--forks", action="append", default=[], help="label=path to a .forks-*.jsonl"
    )
    parser.add_argument("--persona", help="path to a .persona-*.jsonl")
    args = parser.parse_args()

    sets = {}
    for spec in args.forks:
        label, path = spec.split("=", 1)
        sets[label] = load(path)
    if sets:
        plot_forks(sets, Path(f"{args.out}_forks.png"))
        plot_prompts(sets, Path(f"{args.out}_prompts.png"))
        print(f"wrote {args.out}_forks.png, {args.out}_prompts.png")
    if args.persona:
        plot_persona(
            load(args.persona),
            Path(f"{args.out}_persona.png"),
            Path(f"{args.out}_persona_acts.png"),
        )
        print(f"wrote {args.out}_persona.png, {args.out}_persona_acts.png")


if __name__ == "__main__":
    main()
