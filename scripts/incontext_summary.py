# ruff: noqa: E501
"""Summary of the in-context evidence arms (scripts/incontext_evidence.py): per size, the mean over
seeds of each arm's character effect (villain - other, nats, all acts and steal + cheat) with a 95 % t
interval, printed and drawn to logs/v4/incontext/evidence_summary.png. Exploratory (Kevin, 2026-09-25).

    uv run python scripts/incontext_summary.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36}  # t quantiles, no scipy dependency
IN = Path("logs/v4/incontext/evidence")
SIZES = ["3M", "10M", "30M", "100M"]
GROUPS = {"source (k = 1)": ["as_probed", "no_other", "no_villain", "neither"],
          "mixed, transplanted corpus scenes (k = 2)": ["as_probed_k2", "T_BB", "T_BG", "T_GB", "T_GG"],
          "mixed, fork-template scenes (k = 2)": ["BB", "BG", "GB", "GG"]}


def value(rec, arm, acts):
    """Villain - other over the items of the given acts (None for all)."""
    items = json.load(open(f"logs/v4/incontext/sets/{arm}.json"))["items"]
    sc = np.array(rec[arm]["scores"])
    keep = np.array([acts is None or it["act"] in acts for it in items])
    vil = np.array([it["group"] == "villain" for it in items])
    return sc[keep & vil].mean() - sc[keep & ~vil].mean()


def ci(xs):
    xs = np.array(xs)
    h = T975[len(xs) - 1] * xs.std(ddof=1) / np.sqrt(len(xs))
    return xs.mean(), h


def main():
    recs = {s: [json.load(open(f)) for f in sorted(IN.glob(f"v4-ladder-C-{s}-s*.json"))] for s in SIZES}
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(18, 5.2), sharey=True)
    for (title, arms), ax in zip(GROUPS.items(), axes):
        print(f"\n{title}")
        for acts, tag, alpha in ((None, "all acts", 1.0), ({"steal", "cheat"}, "steal + cheat", 0.45)):
            print(f"  [{tag}]  " + "  ".join(f"{a:>12}" for a in arms))
            for si, s in enumerate(SIZES):
                row = []
                for ai, a in enumerate(arms):
                    vals = [value(r, a, acts) for r in recs[s] if a in r]
                    if len(vals) < 2:
                        row.append(" " * 12)
                        continue
                    m, h = ci(vals)
                    row.append(f"{m:6.2f} ±{h:4.2f}")
                    x = ai + (si - 1.5) * 0.18
                    ax.errorbar(x, m, h, fmt="o" if acts is None else "s", color=plt.cm.viridis(si / 3.5), alpha=alpha,
                                ms=5, capsize=2, label=(s if ai == 0 and acts is None else None))
                print(f"  {s:>5}  " + "  ".join(row))
        ax.set_xticks(range(len(arms)), arms, rotation=20)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("character effect: villain − other (nats)")
    axes[0].legend(title="size (circles: all acts;\nfaint squares: steal + cheat)", fontsize=8, title_fontsize=8)
    fig.suptitle("In-context character: which evidence it uses (ladder-C final checkpoints, 8 seeds, 95 % t intervals; exploratory)")
    fig.tight_layout()
    fig.savefig("logs/v4/incontext/evidence_summary.png", dpi=110)


if __name__ == "__main__":
    main()
