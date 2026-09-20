# ruff: noqa: E501
"""Is the EM-analogue spillover a function of how strong the lexical persona is? (NEXT_SESSION.md
Tier 0, item 2.) For every base run that has a "Ben steals" (and, where present, "Ben lies")
finetune plus a neutral finetune, plot the finetune's spillover (Ben − other kind names on the
untaught acts) and its attachment (Ben − kind on the taught act) against the base model's own
k = 0 name effect N − K. Sources: the map, coarse-cut, Stage 2b box and Mac probe JSONs.

    uv run python scripts/cast_spillover_vs_persona.py --out docs/figures/cast_spillover_vs_persona
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

ACTS = ["steal", "lie", "cheat", "tease"]
BASE_JSONS = [
    "logs/cast/map/map_k0_base.json",
    "logs/cast/sweep/sweep_k0_base.json",
    "logs/cast/sweep/extra_k0_big.json",
    "logs/cast/box/box_k0_base.json",
    "logs/cast/k0_base.json",
]
FT_JSONS = [
    "logs/cast/map/map_k0_ft.json",
    "logs/cast/sweep/sweep_k0_ft.json",
    "logs/cast/box/box_k0_ft.json",
    "logs/cast/k0_ft_all.json",
]
FT = re.compile(
    r"^(?P<set>ft-(?P<act>[a-z]+)-(?P<cast>C[a-z]+)(?:-(?P<steps>\d+))?)-(?P<run>story-[0-9a-z-]+_v3-[A-Za-z0-9-]+-s\d)"
)


def run_id(tag):
    """Normalize 'story-10m-3x/v3-x-s0' and 'story-10m-3x_v3-x-s0/ckpt_...pt' to 'story-10m-3x_v3-x-s0'."""
    return tag.split("/ckpt_")[0].replace("/", "_")


def mean_scores(d):
    return {k: float(np.mean(v)) for k, v in d.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/cast_spillover_vs_persona"))
    args = parser.parse_args()
    base = {}
    for p in BASE_JSONS:
        for tag, d in json.load(open(p)).items():
            m = mean_scores(d)
            base[run_id(tag)] = {
                "ALL": m["naughty|ALL"] - m["kind|ALL"],
                "steal": m["naughty|steal"] - m["kind|steal"],
            }
    fts = defaultdict(dict)  # run -> set -> scores
    for p in FT_JSONS:
        for tag, d in json.load(open(p)).items():
            m = FT.match(tag)
            if (
                m and m["cast"] in ("Ckind", "Ccast") and m["steps"] is None
            ):  # Ben (or the neutral set), 100 steps
                fts[m["run"]][m["act"]] = mean_scores(d)
    rows = []
    for run, sets in sorted(fts.items()):
        if "neutral" not in sets or run not in base:
            continue
        for act in ("steal", "lie"):
            if act not in sets:
                continue
            d = {
                a: {
                    cls: sets[act][f"{cls}|{a}"] - sets["neutral"][f"{cls}|{a}"]
                    for cls in ("Ben (kind, ft)", "kind")
                }
                for a in ACTS
            }
            attach = d[act]["Ben (kind, ft)"] - d[act]["kind"]
            others = [a for a in ACTS if a != act]
            spill = np.mean([d[a]["Ben (kind, ft)"] - d[a]["kind"] for a in others])
            stem = run.split("_v3-")[0]
            rows.append((run, stem, act, base[run]["ALL"], base[run]["steal"], attach, spill))
    print(
        f"{'run':52s} {'taught':6s} {'base N−K ALL':>12s} {'base steal':>10s} {'attach':>8s} {'spill':>8s}"
    )
    for r in rows:
        print(f"{r[0]:52s} {r[2]:6s} {r[3]:+12.2f} {r[4]:+10.2f} {r[5]:+8.2f} {r[6]:+8.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = {"story-10m-3x": "C0", "story-10m": "C4", "story-30m": "C1", "story-100m": "C2"}
    for ax, (ycol, yl) in zip(
        axes[:2],
        (
            (6, "spillover: Ben − kind, untaught acts (nats)"),
            (5, "attachment: Ben − kind, taught act (nats)"),
        ),
        strict=False,
    ):
        for act, marker in (("steal", "o"), ("lie", "^")):
            for stem, color in colors.items():
                pts = [(r[3], r[ycol]) for r in rows if r[1] == stem and r[2] == act]
                if pts:
                    ax.scatter(
                        *zip(*pts, strict=True),
                        s=28,
                        marker=marker,
                        color=color,
                        alpha=0.8,
                        label=f"{stem[6:]}, Ben {act}s",
                    )
        ax.set_xlabel("base model k0 name effect N − K, ALL (nats)")
        ax.set_ylabel(yl)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    steal = [(r[3], r[6]) for r in rows if r[2] == "steal"]
    x, y = map(np.array, zip(*steal, strict=True))
    A = np.vstack([x, np.ones_like(x)]).T
    (b, a), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = np.corrcoef(x, y)[0, 1]
    xx = np.array([x.min(), x.max()])
    axes[0].plot(
        xx, a + b * xx, color="k", lw=1, label=f"steal fit: {b:.2f}·x {a:+.2f}, r = {r:.2f}"
    )
    axes[0].legend(fontsize=7)
    print(
        f"\nBen steals, all sizes: spillover = {b:.3f} × (base N−K) {a:+.3f}, r = {r:.2f}, n = {len(x)}"
    )
    # spillover as a fraction of attachment, where attachment is large enough to define a ratio
    ax = axes[2]
    for stem, color in colors.items():
        pts = [(r[5], r[6]) for r in rows if r[1] == stem and r[2] == "steal"]
        if pts:
            ax.scatter(*zip(*pts, strict=True), s=28, color=color, alpha=0.8, label=stem[6:])
    xx = np.array([0, max(r[5] for r in rows)])
    ax.plot(xx, 0.3 * xx, "k--", lw=0.8, label="0.3 × attachment (Stage 2b dose series)")
    ax.set_xlabel("attachment (nats)")
    ax.set_ylabel("spillover (nats)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
