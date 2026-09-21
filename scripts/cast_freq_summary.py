# ruff: noqa: E501
"""Readout of the frequency ladder (DATA.md §17; prompts/cast_freq_ladder.yaml): each K-name's k = 0
disposition — the N-name mean minus that name's score, so larger = more "never"-like — by its
frequency tier and its own exception count, pooled over seeds. Two views: fixed COUNT (does a name
with 3 or 11 exceptions lose the same regardless of frequency?) and fixed FRACTION (do 30 / 11,800,
3 / 1,100 and 1 / 275 lose the same?).

    uv run python scripts/cast_freq_summary.py --out docs/figures/cast_freq
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dramatis.assemble import load_cast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", nargs="*", default=sorted(glob.glob("logs/cast/freq_k0_pername*.json")))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/cast_freq"))
    args = parser.parse_args()
    cast = load_cast()
    N, K = cast["naughty"], cast["kind"]
    spec = yaml.safe_load(open("prompts/cast_freq_ladder.yaml"))
    tier = spec["tiers"]
    real = json.load(open("data/corpora/v3-freq-ladder/recipe.json"))["cast_realized"]
    vals = defaultdict(list)  # (tier, k) -> [disposition per (name, seed)]
    per_name = defaultdict(list)
    n_runs = 0
    for p in args.json:
        for tag, d in json.load(open(p)).items():
            if not re.search(r"v3-freq-ladder-s\d+/ckpt_006499", tag):
                continue
            n_runs += 1
            n_mean = np.mean([np.mean(d[f"naughty:{n}|ALL"]) for n in N])
            for name in K:
                v = n_mean - np.mean(d[f"kind:{name}|ALL"])
                vals[(tier[name], real[name]["villain"])].append(v)
                per_name[name].append(v)
    print(f"{n_runs} runs. N-mean − K-name disposition (ALL acts), mean ± s.e.m. over names × seeds")
    print(f"{'tier':6s} {'appearances':>12s} " + "".join(f"{k:>16s}" for k in ("k = 0", "k = 1", "k = 3", "k = 11", "k = 30")))
    totals = {t: int(np.mean([real[n]["total"] for n in K if tier[n] == t])) for t in ("high", "mid", "low")}
    for t in ("high", "mid", "low"):
        line = f"{t:6s} {totals[t]:12d} "
        for k in (0, 1, 3, 11, 30):
            v = vals.get((t, k))
            line += (f"{np.mean(v):+6.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}" if v else "—").rjust(16)
        print(line)
    print("\nfixed fraction ≈ 0.25 %: " + "  ".join(f"{t} {k}/{totals[t]} {np.mean(vals[(t, k)]):+.2f}" for t, k in (("high", 30), ("mid", 3), ("low", 1)) if (t, k) in vals))
    print("fixed fraction ≈ 1 %:    " + "  ".join(f"{t} {k}/{totals[t]} {np.mean(vals[(t, k)]):+.2f}" for t, k in (("mid", 11), ("low", 3)) if (t, k) in vals))
    print("fixed count 3:           " + "  ".join(f"{t} {np.mean(vals[(t, 3)]):+.2f}" for t in ("high", "mid", "low") if (t, 3) in vals))
    print("fixed count 11:          " + "  ".join(f"{t} {np.mean(vals[(t, 11)]):+.2f}" for t in ("high", "mid", "low") if (t, 11) in vals))
    print("\nper name: " + ", ".join(f"{n} ({tier[n]}, {real[n]['villain']}) {np.mean(v):+.2f}" for n, v in sorted(per_name.items(), key=lambda kv: (tier[kv[0]], real[kv[0]]['villain']))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for t, color in (("high", "C0"), ("mid", "C1"), ("low", "C2")):
        ks = sorted(k for (tt, k) in vals if tt == t)
        m = [np.mean(vals[(t, k)]) for k in ks]
        e = [np.std(vals[(t, k)], ddof=1) / np.sqrt(len(vals[(t, k)])) for k in ks]
        axes[0].errorbar(ks, m, yerr=e, fmt="o-", color=color, capsize=3, label=f"{t} tier (≈ {totals[t]:,} appearances)")
        fr = [k / totals[t] * 100 for k in ks]
        axes[1].errorbar(fr, m, yerr=e, fmt="o-", color=color, capsize=3, label=f"{t} tier")
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xlabel("exceptions per name (count)")
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_xlabel("exceptions per name (% of appearances)")
    for ax in axes:
        ax.set_ylabel("N mean − this K-name, ALL (nats)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("by count: tiers agree if the model counts exceptions", fontsize=9)
    axes[1].set_title("by fraction: tiers agree if it tracks the fraction", fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
