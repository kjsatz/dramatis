"""Summarize the Stage 1 grid: per cell × act, the frozen readouts averaged over seeds.

Usage:
    uv run python scripts/summarize_grid.py                      # every finished run under runs/
    uv run python scripts/summarize_grid.py --out docs/results/stage1   # + markdown and figures

For each run <runs>/*_story-10m_<cell>-s<seed>/ with a summary.json and eval files on its final
checkpoint (scripts/eval_runs.sh), takes the per-item fork log-odds and the persona contrasts at
k = k_max, averages over items, then reports mean ± s.e.m. across seeds per cell (seeds are the
unit of replication; items are fixed and shared). Prints three tables: fork log-odds per cell ×
act; persona contrasts per cell; the pre-registered headline comparison (specialist-off vs
specialist-negonly) and every cell's difference from none-off, per act.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dramatis.provenance import file_sha256

ACTS = ["steal", "lie", "cheat", "tease", "break", "blame", "exclude", "sneak"]
LABEL_LEVELS = ("opening", "closing", "both")
CELL_RE = r"(?P<cell>[a-z]+-[a-z]+(?:-L[a-z]+)?)"  # <villain>-<moralizing>[-L<labels>]
CELLS = [
    "none-off", "none-symmetric",
    "specialist-off", "specialist-negonly", "specialist-symmetric",
    "generalist-off", "generalist-negonly", "generalist-symmetric",
]  # fmt: skip
# Stage 1D label cells (M = off): <villain>-off-L<opening|closing|both>
CELLS += [f"{v}-off-L{lv}" for v in ("none", "specialist", "generalist") for lv in LABEL_LEVELS]
RUN_RE = re.compile(r"_story-10m_(?P<prefix>v\d+-)?" + CELL_RE + r"-s(?P<seed>\d+)$")
CONTRASTS = {  # name -> (kind, condition minus condition)
    "cross bad-A − benign-A": ("cross", "bad-A", "benign-A"),
    "same bad-A − benign-A": ("same", "bad-A", "benign-A"),
    "cross bad-A − bad-B": ("cross", "bad-A", "bad-B"),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def fork_means(rows: list[dict], key: str = "log_odds") -> dict[str, float]:
    """Per act, the mean of `key` over items."""
    by = defaultdict(list)
    for r in rows:
        by[r["act"]].append(r[key])
    return {a: float(np.mean(v)) for a, v in by.items()}


def persona_contrasts(rows: list[dict], k: int) -> dict[str, dict[str, float]]:
    """Per contrast, per act (and 'ALL'): mean over items of the paired difference at k
    (draws averaged first)."""
    per = defaultdict(lambda: defaultdict(list))  # (kind, cond) -> id -> draws
    act_of = {}
    for r in rows:
        if r["k"] == k:
            per[(r["kind"], r["condition"])][r["id"]].append(r["log_odds"])
            act_of[r["id"]] = r["act"]
    out = {}
    for name, (kind, c1, c2) in CONTRASTS.items():
        d1, d2 = per[(kind, c1)], per[(kind, c2)]
        diffs = {i: np.mean(d1[i]) - np.mean(d2[i]) for i in d1 if i in d2}
        out[name] = {"ALL": float(np.mean(list(diffs.values())))}
        for act in ACTS:
            v = [d for i, d in diffs.items() if act_of[i] == act]
            if v:
                out[name][act] = float(np.mean(v))
    return out


def mean_sem(values: list[float]) -> tuple[float, float, int]:
    v = np.array(values, dtype=float)
    sem = float(v.std(ddof=1) / len(v) ** 0.5) if len(v) > 1 else float("nan")
    return float(v.mean()), sem, len(v)


def fmt(m: float, s: float) -> str:
    return f"{m:+.2f} ± {s:.2f}" if np.isfinite(s) else f"{m:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="", help='run-name prefix to select, e.g. "v2-"')
    parser.add_argument("--forks", type=Path, default=Path("prompts/forks.yaml"))
    parser.add_argument("--k", type=int, default=3, help="persona contrasts at this k")
    parser.add_argument("--out", help="prefix for <out>.md and <out>_*.png")
    args = parser.parse_args()
    sha = file_sha256(args.forks)[:8]

    # Collect per-run statistics.
    forks: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))  # cell->act
    persona: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )  # cell -> contrast -> act -> per-seed values
    val_loss: dict[str, list[float]] = defaultdict(list)
    prompt_lp: dict[str, list[float]] = defaultdict(list)  # mean prompt log p/token per run
    seeds: dict[str, set] = defaultdict(set)
    for run in sorted(args.runs.glob("*_story-10m_*-s*")):
        m = RUN_RE.search(run.name)
        if not m or (m["prefix"] or "") != args.prefix or not (run / "summary.json").exists():
            continue
        cell, seed = m["cell"], int(m["seed"])
        ckpts = sorted((run / "checkpoints").glob("ckpt_*.pt"))
        if not ckpts:
            continue
        base = ckpts[-1].with_suffix("")
        f_path = Path(f"{base}.{args.forks.stem}-{sha}.jsonl")  # eval_forks names by fork stem
        p_path = Path(f"{base}.persona-{sha}-k3d3s0.jsonl")
        if not f_path.exists():
            continue
        seeds[cell].add(seed)
        val_loss[cell].append(json.load(open(run / "summary.json"))["final_val_loss"])
        fork_rows = load_jsonl(f_path)
        for act, v in fork_means(fork_rows).items():
            forks[cell][act].append(v)
        ppl = fork_means(fork_rows, "logp_prompt_per_token")
        prompt_lp[cell].append(float(np.mean(list(ppl.values()))))
        if p_path.exists():
            for name, per_act in persona_contrasts(load_jsonl(p_path), args.k).items():
                for act, v in per_act.items():
                    persona[cell][name][act].append(v)

    cells = [c for c in CELLS if c in forks]
    lines = []
    lines.append(f"Stage 1 grid: {sum(len(s) for s in seeds.values())} runs, forks {sha}\n")
    lines.append("## Fork log-odds (mean over items; mean ± s.e.m. across seeds)\n")
    lines.append("| cell | seeds | val loss | prompt log p/tok | " + " | ".join(ACTS) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(ACTS))
    for c in cells:
        vl = mean_sem(val_loss[c])
        row = [f"`{c}`", str(len(seeds[c])), f"{vl[0]:.3f}", f"{mean_sem(prompt_lp[c])[0]:.2f}"]
        for a in ACTS:
            m, s, _ = mean_sem(forks[c][a])
            row.append(fmt(m, s))
        lines.append("| " + " | ".join(row) + " |")

    lines.append(
        f"\n## Persona contrasts at k = {args.k} (all items; mean ± s.e.m. across seeds)\n"
    )
    lines.append("| cell | " + " | ".join(CONTRASTS) + " |")
    lines.append("|---|" + "---|" * len(CONTRASTS))
    for c in cells:
        row = [f"`{c}`"]
        for name in CONTRASTS:
            v = persona[c][name].get("ALL", [])
            row.append(fmt(*mean_sem(v)[:2]) if v else "—")
        lines.append("| " + " | ".join(row) + " |")

    def diff(c1: str, c2: str, act: str) -> str:
        if not forks[c1][act] or not forks[c2][act]:
            return "—"
        m1, s1, _ = mean_sem(forks[c1][act])
        m2, s2, _ = mean_sem(forks[c2][act])
        s = (s1**2 + s2**2) ** 0.5 if np.isfinite(s1) and np.isfinite(s2) else float("nan")
        return fmt(m1 - m2, s)

    lines.append("\n## Differences in fork log-odds (per act; s.e. from the two cells' s.e.m.)\n")
    lines.append("| comparison | " + " | ".join(ACTS) + " |")
    lines.append("|---|" + "---|" * len(ACTS))
    if "specialist-off" in forks and "specialist-negonly" in forks:
        lines.append(
            "| **headline: specialist-negonly − specialist-off** | "
            + " | ".join(diff("specialist-negonly", "specialist-off", a) for a in ACTS)
            + " |"
        )
    for c in cells:
        if c != "none-off" and "none-off" in forks:
            lines.append(
                f"| `{c}` − none-off | " + " | ".join(diff(c, "none-off", a) for a in ACTS) + " |"
            )
    text = "\n".join(lines)
    print(text)

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(f"{args.out}.md").write_text(text + "\n")
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
        x = np.arange(len(cells))
        colors = ["C0", "C0", "C1", "C1", "C1", "C2", "C2", "C2"][: len(cells)]

        def bars(ax, ms):
            ax.bar(x, [m[0] for m in ms], yerr=[m[1] if np.isfinite(m[1]) else 0 for m in ms],
                   capsize=3, color=colors)  # fmt: skip
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x, cells, rotation=60, ha="right", fontsize=7)

        for ax, act in zip(axes.flat, ACTS, strict=True):
            bars(ax, [mean_sem(forks[c][act]) for c in cells])
            ax.set_title(act)
        axes[0, 0].set_ylabel("fork log-odds (nats)")
        axes[1, 0].set_ylabel("fork log-odds (nats)")
        fig.suptitle("Fork log-odds by cell (mean over items; error bars: s.e.m. across seeds)")
        fig.tight_layout()
        fig.savefig(f"{args.out}_forks.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(15, 4), sharey=True)
        for ax, name in zip(axes, CONTRASTS, strict=True):
            bars(ax, [mean_sem(persona[c][name].get("ALL", [float("nan")])) for c in cells])
            ax.set_title(f"{name}, k = {args.k}")
        axes[0].set_ylabel("nats")
        fig.suptitle("Persona contrasts by cell (all items; error bars: s.e.m. across seeds)")
        fig.tight_layout()
        fig.savefig(f"{args.out}_persona.png", dpi=130)
        plt.close(fig)
        print(f"-> {args.out}.md, {args.out}_forks.png, {args.out}_persona.png")


if __name__ == "__main__":
    main()
