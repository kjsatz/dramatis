"""Summarize the Stage 1 perturbation: the generalization difference per cell (PLAN "Measurements").

Usage:
    uv run python scripts/summarize_finetunes.py                     # tables to stdout
    uv run python scripts/summarize_finetunes.py --out docs/figures/stage1   # + figures

For every pretrained run (cell, seed) with finetunes on the act sets (ft-steal, ft-lie, ft-cheat,
ft-tease) and on ft-neutral, and fork evals on their final checkpoints, computes per finetune act
X and probed act a:

    Δ(X, a) = mean over items of [ LO(ft-X) − LO(ft-neutral) ]      (same base model)

i.e. how much more the X finetune raised the odds of a than the neutral finetune did. Per cell the
4 × 8 matrix is averaged over seeds (mean ± s.e.m., seeds as replication). The diagonal Δ(X, X)
is the manipulation check; the off-diagonal performed acts and the four held-out acts are the
generalization profile; the **generalization difference** of a cell is Δ averaged over finetune
acts X and over held-out acts a. Also reported: the same at the step-50 checkpoint (half dose)
when present, and the headline comparison specialist-off vs specialist-negonly.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dramatis.provenance import file_sha256

ACTS = ["steal", "lie", "cheat", "tease", "break", "blame", "exclude", "sneak"]
PERFORMED, HELD_OUT = ACTS[:4], ACTS[4:]
LABEL_LEVELS = ("opening", "closing", "both")
CELL_RE = r"(?P<cell>[a-z]+-[a-z]+(?:-[LC][a-z]+)?)"  # <villain>-<moralizing>[-L<labels>|-Ccast]
CELLS = [
    "none-off", "none-symmetric",
    "specialist-off", "specialist-negonly", "specialist-symmetric",
    "generalist-off", "generalist-negonly", "generalist-symmetric",
]  # fmt: skip
# Stage 1D label cells (M = off): <villain>-off-L<opening|closing|both>
CELLS += [f"{v}-off-L{lv}" for v in ("none", "specialist", "generalist") for lv in LABEL_LEVELS]
FT_RE = re.compile(
    r"_ft-(?P<set>[a-z]+)_(?P<prefix>v\d+-)?" + CELL_RE + r"-s(?P<seed>\d+)$"
)  # see --stem


def ft_re(stem: str) -> re.Pattern:
    """Finetune run names: ft-<set>_<tag> for 10M bases, ft-<set>-<stem>_<tag> otherwise."""
    infix = "" if stem == "story-10m" else "-" + re.escape(stem)
    return re.compile(
        r"_ft-(?P<set>[a-z]+)" + infix + r"_(?P<prefix>v\d+-)?" + CELL_RE + r"-s(?P<seed>\d+)$"
    )


def item_log_odds(path: Path) -> dict[str, float]:
    return {r["id"]: r["log_odds"] for r in map(json.loads, open(path))}


def act_of_items(path: Path) -> dict[str, str]:
    return {r["id"]: r["act"] for r in map(json.loads, open(path))}


def mean_sem(values) -> tuple[float, float]:
    v = np.array(list(values), dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / len(v) ** 0.5) if len(v) > 1 else float("nan")


def collect(runs: Path, sha: str, ckpt_name: str, prefix: str = "") -> tuple[dict, dict]:
    """delta[cell][X][act] -> list over seeds of the item-mean Δ; n_seeds[cell] -> set of seeds."""
    evals: dict[tuple, Path] = {}  # (cell, seed, set) -> fork eval file
    act_of: dict[str, str] = {}
    for run in runs.glob("*_ft-*"):
        m = FT_RE.search(run.name)
        if not m or (m["prefix"] or "") != prefix:
            continue
        f = run / "checkpoints" / f"{ckpt_name}.forks-{sha}.jsonl"
        if f.exists():
            evals[(m["cell"], int(m["seed"]), m["set"])] = f
            if not act_of:
                act_of = act_of_items(f)
    delta: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds: dict = defaultdict(set)
    for (cell, seed, X), f in sorted(evals.items()):
        if X == "neutral" or (cell, seed, "neutral") not in evals:
            continue
        lo_x, lo_n = item_log_odds(f), item_log_odds(evals[(cell, seed, "neutral")])
        by_act = defaultdict(list)
        for i in lo_x:
            if i in lo_n:
                by_act[act_of[i]].append(lo_x[i] - lo_n[i])
        for a, v in by_act.items():
            delta[cell][X][a].append(float(np.mean(v)))
        seeds[cell].add(seed)
    return delta, seeds


def fmt(m: float, s: float) -> str:
    if np.isnan(m):
        return "—"
    return f"{m:+.2f} ± {s:.2f}" if np.isfinite(s) else f"{m:+.2f}"


def gen_diff(delta_cell: dict) -> list[float]:
    """Per seed: Δ averaged over finetune acts and held-out acts (the cell's generalization
    difference). Seeds are aligned by position, so all X must have the same seed count."""
    per_x = [
        np.mean([delta_cell[X][a] for a in HELD_OUT], axis=0) for X in PERFORMED if X in delta_cell
    ]
    return list(np.mean(per_x, axis=0)) if per_x else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="", help='run-name prefix to select, e.g. "v2-"')
    parser.add_argument("--forks", type=Path, default=Path("prompts/forks.yaml"))
    parser.add_argument("--out", help="prefix for figures")
    parser.add_argument(
        "--stem", default="story-10m", help="base config stem, e.g. story-100m (Stage 2a)"
    )
    args = parser.parse_args()
    sha = file_sha256(args.forks)[:8]
    global FT_RE
    FT_RE = ft_re(args.stem)
    delta, seeds = collect(args.runs, sha, "ckpt_000099", args.prefix)
    delta50, _ = collect(args.runs, sha, "ckpt_000050", args.prefix)
    cells = [c for c in CELLS if c in delta]

    lines = [
        f"Stage 1 perturbation: {sum(len(s) for s in seeds.values())} base runs with finetunes, "
        f"forks {sha}\n"
    ]
    lines.append(
        "## Δ(X, a) = LO(ft-X) − LO(ft-neutral), mean over items; mean ± s.e.m. across seeds\n"
    )
    for c in cells:
        lines.append(f"### `{c}` ({len(seeds[c])} seeds)\n")
        lines.append("| finetune X \\ probed a | " + " | ".join(ACTS) + " |")
        lines.append("|---|" + "---|" * len(ACTS))
        for X in PERFORMED:
            if X not in delta[c]:
                continue
            cells_ = []
            for a in ACTS:
                m, s = mean_sem(delta[c][X][a])
                cells_.append(("**" + fmt(m, s) + "**") if a == X else fmt(m, s))
            lines.append(f"| ft-{X} | " + " | ".join(cells_) + " |")
        lines.append("")

    lines.append(
        "## Generalization difference per cell (Δ averaged over finetune acts and held-out acts)\n"
    )
    lines.append(
        "| cell | seeds | held-out (dose 100) | held-out (dose 50) | same-act diagonal "
        "| other performed | held-out / diagonal |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    gd = {}
    for c in cells:
        gd[c] = gen_diff(delta[c])
        diag = [np.mean(delta[c][X][X]) for X in PERFORMED if X in delta[c]]
        other = [
            np.mean(delta[c][X][a]) for X in PERFORMED if X in delta[c] for a in PERFORMED if a != X
        ]
        g50 = gen_diff(delta50[c]) if c in delta50 else []
        # per seed: held-out Δ per unit of same-act Δ (the finetune's strength differs by cell)
        diag_seed = np.mean([delta[c][X][X] for X in PERFORMED if X in delta[c]], axis=0)
        ratio = list(np.array(gd[c]) / diag_seed) if len(diag_seed) else []
        lines.append(
            f"| `{c}` | {len(seeds[c])} | {fmt(*mean_sem(gd[c]))} | {fmt(*mean_sem(g50))} | "
            f"{fmt(*mean_sem(diag))} | {fmt(*mean_sem(other))} | {fmt(*mean_sem(ratio))} |"
        )
    if "specialist-off" in gd and "specialist-negonly" in gd:
        m1, s1 = mean_sem(gd["specialist-negonly"])
        m2, s2 = mean_sem(gd["specialist-off"])
        se = (s1**2 + s2**2) ** 0.5 if np.isfinite(s1) and np.isfinite(s2) else float("nan")
        lines.append(
            "\n**Headline** specialist-negonly − specialist-off (generalization difference): "
            f"{fmt(m1 - m2, se)}"
        )
    text = "\n".join(lines)
    print(text)

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        # Figure 1: generalization difference vs cell (PLAN's headline figure), per held-out act.
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(cells))
        w = 0.8 / (len(HELD_OUT) + 1)
        for j, a in enumerate(HELD_OUT):
            # per cell: per-seed Δ(X, a) averaged over finetune acts X; mean ± s.e.m. over seeds
            ms = [
                mean_sem(np.mean([delta[c][X][a] for X in PERFORMED if X in delta[c]], axis=0))
                for c in cells
            ]
            err = [m[1] if np.isfinite(m[1]) else 0 for m in ms]
            ax.bar(
                x + (j - len(HELD_OUT) / 2) * w, [m[0] for m in ms], w, yerr=err, capsize=2, label=a
            )
        ms = [mean_sem(gd[c]) for c in cells]
        err = [m[1] if np.isfinite(m[1]) else 0 for m in ms]
        ax.bar(
            x + (len(HELD_OUT) / 2) * w,
            [m[0] for m in ms],
            w,
            yerr=err,
            capsize=2,
            color="k",
            label="mean",
        )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, cells, rotation=30, ha="right")
        ax.set_ylabel("Δ log-odds: bad finetune − neutral finetune (nats)")
        ax.set_title(
            "Generalization difference on held-out acts by pretraining cell "
            "(averaged over finetune acts)"
        )
        ax.legend(fontsize=8, ncol=5)
        fig.tight_layout()
        fig.savefig(f"{args.out}_gendiff.png", dpi=130)
        plt.close(fig)
        # Figure 2: the 4 x 8 matrices per cell.
        n_rows = (len(cells) + 3) // 4  # 8 grid cells -> 2 rows; with label cells -> more
        fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4 * n_rows), constrained_layout=True)
        axes = np.atleast_2d(axes)
        vmax = (
            max(
                abs(np.mean(delta[c][X][a]))
                for c in cells
                for X in delta[c]
                for a in ACTS
                if delta[c][X][a]
            )
            or 1
        )
        for ax, c in zip(axes.flat, cells + [None] * (4 * n_rows - len(cells)), strict=True):
            if c is None:
                ax.axis("off")
                continue
            M = np.array(
                [
                    [np.mean(delta[c][X][a]) if delta[c][X][a] else np.nan for a in ACTS]
                    for X in PERFORMED
                ]
            )
            im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(ACTS)), ACTS, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(PERFORMED)), [f"ft-{X}" for X in PERFORMED], fontsize=8)
            ax.axvline(3.5, color="k", lw=0.8)
            ax.set_title(f"{c} ({len(seeds[c])} seeds)", fontsize=9)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Δ log-odds (nats)")
        fig.suptitle(
            "Δ(X, a) = LO(ft-X) − LO(ft-neutral): rows = finetune act, columns = probed act "
            "(right of the line: held out)"
        )
        fig.savefig(f"{args.out}_matrix.png", dpi=130)
        plt.close(fig)
        print(f"-> {args.out}_gendiff.png, {args.out}_matrix.png")


if __name__ == "__main__":
    main()
