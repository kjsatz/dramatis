"""Summarize the Stage 1D label probes (scripts/eval_labels.py output) across cells and seeds.

    uv run python scripts/summarize_labels.py --prefix v3- --out docs/figures/stage1c_labels

For every pretrained run's final checkpoint with a *.labels-*.jsonl: (1) the k = 0 label
baseline LO_label = log P(bad trait) - log P(good trait) after a held-out closing frame;
(2) label inference: LO_label after k prior acts minus k = 0, and the contrast bad-A - benign-A
(persona inference as next-token prediction), per kind; (3) label-conditioned forks: fork LO
under a bad-trait opening frame minus under a good-trait one, per act. Finetuned runs
(ft-steal, ft-neutral) get the k = 0 baseline too, so the finetune's effect on P(bad trait) is
read directly. Mean ± s.e.m. across seeds; markdown table + figures.
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

ACTS = ["steal", "lie", "cheat", "tease", "break", "blame", "exclude", "sneak"]
LABEL_LEVELS = ("opening", "closing", "both")
CELL_RE = r"(?P<cell>[a-z]+-[a-z]+(?:-L[a-z]+)?)"
RUN_RE = re.compile(r"_story-10m_(?P<prefix>v\d+-)?" + CELL_RE + r"-s(?P<seed>\d+)$")
FT_RE = re.compile(r"_ft-(?P<set>[a-z]+)_(?P<prefix>v\d+-)?" + CELL_RE + r"-s(?P<seed>\d+)$")
CELLS = [
    "none-off", "none-symmetric",
    "specialist-off", "specialist-negonly", "specialist-symmetric",
    "generalist-off", "generalist-negonly", "generalist-symmetric",
]  # fmt: skip
CELLS += [f"{v}-off-L{lv}" for v in ("none", "specialist", "generalist") for lv in LABEL_LEVELS]


def mean_sem(values):
    v = np.array(values, dtype=float)
    sem = float(v.std(ddof=1) / len(v) ** 0.5) if len(v) > 1 else float("nan")
    return float(v.mean()), sem, len(v)


def fmt(m, s):
    return f"{m:+.2f} ± {s:.2f}" if np.isfinite(s) else f"{m:+.2f}"


def summarize_file(path: Path, k: int) -> dict:
    rows = [json.loads(line) for line in open(path)]
    label = [r for r in rows if r["probe"] == "label"]
    forks = [r for r in rows if r["probe"] == "fork"]
    base = defaultdict(list)
    for r in label:
        if r["k"] == 0:
            base[r["id"]].append(r["log_odds"])
    base = {i: float(np.mean(v)) for i, v in base.items()}
    act_of = {r["id"]: r["act"] for r in label}
    out = {"k0": {"ALL": float(np.mean(list(base.values())))}}
    for act in ACTS:
        v = [b for i, b in base.items() if act_of[i] == act]
        if v:
            out["k0"][act] = float(np.mean(v))
    per = defaultdict(lambda: defaultdict(list))  # (kind, cond) -> id -> draws
    for r in label:
        if r["k"] == k:
            per[(r["kind"], r["condition"])][r["id"]].append(r["log_odds"])
    for kind in ("same", "cross"):
        a, b = per[(kind, "bad-A")], per[(kind, "benign-A")]
        diffs = {i: np.mean(a[i]) - np.mean(b[i]) for i in a if i in b}
        rise = {i: np.mean(a[i]) - base[i] for i in a if i in base}
        out[f"{kind} bad-A − benign-A"] = {"ALL": float(np.mean(list(diffs.values())))}
        out[f"{kind} bad-A − k0"] = {"ALL": float(np.mean(list(rise.values())))}
        for act in ACTS:
            v = [d for i, d in diffs.items() if act_of[i] == act]
            w = [d for i, d in rise.items() if act_of[i] == act]
            if v:
                out[f"{kind} bad-A − benign-A"][act] = float(np.mean(v))
                out[f"{kind} bad-A − k0"][act] = float(np.mean(w))
    by_item = defaultdict(lambda: defaultdict(list))
    for r in forks:
        by_item[r["id"]][r["valence"]].append(r["log_odds"])
    fc = {i: np.mean(d["bad"]) - np.mean(d["good"]) for i, d in by_item.items() if "bad" in d}
    out["label-conditioned forks"] = {"ALL": float(np.mean(list(fc.values())))}
    for act in ACTS:
        v = [d for i, d in fc.items() if act_of.get(i) == act]
        if v:
            out["label-conditioned forks"][act] = float(np.mean(v))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="", help='run-name prefix to select, e.g. "v3-"')
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", help="prefix for <out>.md and <out>_*.png")
    args = parser.parse_args()

    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # cell -> stat -> act
    ft_k0 = defaultdict(lambda: defaultdict(list))  # (cell, set) -> act -> per-seed k0
    for run in sorted(args.runs.iterdir()):
        files = sorted(run.glob("checkpoints/*.labels-*.jsonl"))
        if not files:
            continue
        final = max(files, key=lambda p: p.name)  # highest step
        m = RUN_RE.search(run.name)
        if m and (m["prefix"] or "") == args.prefix:
            s = summarize_file(final, args.k)
            for stat, acts in s.items():
                for act, v in acts.items():
                    stats[m["cell"]][stat][act].append(v)
            continue
        m = FT_RE.search(run.name)
        if m and (m["prefix"] or "") == args.prefix:
            s = summarize_file(final, args.k)
            for act, v in s["k0"].items():
                ft_k0[(m["cell"], m["set"])][act].append(v)

    cells = [c for c in CELLS if c in stats]
    lines = [f"Label probes: {len(cells)} cells, k = {args.k}", ""]
    for stat in ["k0", "cross bad-A − benign-A", "same bad-A − benign-A", "cross bad-A − k0",
                 "label-conditioned forks"]:  # fmt: skip
        lines += [f"## {stat} (mean over items; mean ± s.e.m. across seeds)", "",
                  "| cell | seeds | ALL | " + " | ".join(ACTS) + " |",
                  "|---|---|---|" + "---|" * len(ACTS)]  # fmt: skip
        for c in cells:
            m_all, s_all, n = mean_sem(stats[c][stat]["ALL"])
            row = [f"`{c}`", str(n), fmt(m_all, s_all)]
            for act in ACTS:
                v = stats[c][stat].get(act)
                row.append(fmt(*mean_sem(v)[:2]) if v else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    if ft_k0:
        lines += ["## k = 0 label baseline after finetuning: LO_label(ft-steal) − LO_label(ft-neutral)",
                  "", "| cell | seeds | ALL | " + " | ".join(ACTS) + " |",
                  "|---|---|---|" + "---|" * len(ACTS)]  # fmt: skip
        for c in cells:
            a, b = ft_k0.get((c, "steal")), ft_k0.get((c, "neutral"))
            if not a or not b:
                continue
            row = [f"`{c}`", str(min(len(a["ALL"]), len(b["ALL"])))]
            for act in ["ALL", *ACTS]:
                if act in a and act in b:
                    d = [x - y for x, y in zip(a[act], b[act], strict=False)]
                    row.append(fmt(*mean_sem(d)[:2]))
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out + ".md").write_text(text + "\n")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for ax, stat in zip(axes, ["k0", "cross bad-A − benign-A", "label-conditioned forks"]):
            ms = [mean_sem(stats[c][stat]["ALL"]) for c in cells]
            ax.bar(range(len(cells)), [m for m, _, _ in ms], yerr=[s if np.isfinite(s) else 0 for _, s, _ in ms],
                   capsize=3, color=["C1" if "-L" in c else "C0" for c in cells])  # fmt: skip
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels(cells, rotation=60, ha="right", fontsize=8)
            ax.set_title(stat)
        axes[0].set_ylabel("nats")
        fig.suptitle("Label probes by cell (mean over items; error bars: s.e.m. across seeds)")
        fig.savefig(args.out + "_cells.png", dpi=130)
        print(f"-> {args.out}.md, {args.out}_cells.png")


if __name__ == "__main__":
    main()
