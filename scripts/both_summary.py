# ruff: noqa: E501
"""Readout of the in-context persona test on the symmetric (DATA.md §14) and both-villains (§18)
corpora: parse persona_indist.py logs (k = 1 and k = 2 bad acts in the prefix, every checkpoint of
the 10M x3 seeds, the 30M / 100M finals) into the character effect (villain-actor − other-actor)
and the story effect (other-actor − base-story) over training, print the finals and plot both
corpora side by side. An in-context persona would appear as a step in the character effect, as
the closing label did (2026-09-19_REPORT_stage2a.md).

    uv run python scripts/both_summary.py --out docs/figures/both_villains
"""

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dramatis.runids import parse as parse_tag
from dramatis.runids import warn

LAST = {
    "story-10m-3x": 19499,
    "story-30m": 6499,
    "story-100m": 6499,
}  # final_weights.pt (or a run spec without a checkpoint) = the last step
ALL = re.compile(
    r"^\s+ALL\s+n=\s*\d+\s+villain-actor\s+(?P<v>[+-]?\d+\.\d+)\s+other-actor\s+(?P<o>[+-]?\d+\.\d+)\s+base-story\s+(?P<b>[+-]?\d+\.\d+)\s+character \(v-o\)\s+(?P<c>[+-]?\d+\.\d+) ± (?P<ce>\d+\.\d+)\s+story \(o-base\)\s+(?P<s>[+-]?\d+\.\d+) ± (?P<se>\d+\.\d+)"
)


def is_header(line):
    """A persona_indist.py model header: one token with a "/" at the start of the line (a
    checkpoint path or a "<stem>/<cell>" run spec), as opposed to the indented result rows."""
    s = line.strip()
    return bool(s) and not line[0].isspace() and " " not in s and "/" in s


def parse(paths):
    """(stem, seed, step) -> {k: (character, ce, story, se)}"""
    out = defaultdict(dict)
    for p in paths:
        k = int(re.search(r"_k(\d)_", p)[1])
        cur = None
        for line in open(p):
            if is_header(line):
                # every header resets cur, so rows under a header we cannot read are dropped
                # rather than overwriting the previous checkpoint's results
                cur = None
                rid = parse_tag(line.strip())
                step = rid.step if rid else None
                if rid and step is None:
                    step = LAST.get(rid.stem)
                if rid is None or step is None:
                    warn(f"both_summary: {p}: cannot read the model header {line.strip()!r}; its rows are skipped")
                    continue
                cur = (rid.stem, rid.seed, step)
                continue
            m = ALL.match(line)
            if m and cur:
                out[cur][k] = (float(m["c"]), float(m["ce"]), float(m["s"]), float(m["se"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/both_villains"))
    args = parser.parse_args()
    corpora = {
        "symmetric (one villain per story)": sorted(glob.glob("logs/cast/sym/sym_k[12]_*.txt")),
        "both-villains mix (1 : 2 : 1)": sorted(
            glob.glob("logs/cast/both_k[12]_*.txt")
        ),  # finals only: the box probes failed (raw stories not synced) and were re-run on the Mac,
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = {"symmetric (one villain per story)": "0.55", "both-villains mix (1 : 2 : 1)": "C3"}
    for name, paths in corpora.items():
        if not paths:
            continue
        res = parse(paths)
        print(f"\n{name}: {len(res)} checkpoints parsed")
        print(
            f"   {'model':22s} {'k = 1 character':>18s} {'k = 2 character':>18s} {'k = 1 story':>14s} {'k = 2 story':>14s}"
        )
        finals = {}
        for (stem, seed, step), d in sorted(res.items()):
            key = (stem, seed)
            if key not in finals or finals[key][0] < step:
                finals[key] = (step, d)
        pooled = defaultdict(list)
        for (stem, seed), (step, d) in sorted(finals.items()):
            c1, c2 = d.get(1, (np.nan,) * 4), d.get(2, (np.nan,) * 4)
            print(
                f"   {stem + ' s' + str(seed) + ' @' + str(step):22s} {c1[0]:+8.2f} ± {c1[1]:.2f} {c2[0]:+8.2f} ± {c2[1]:.2f} {c1[2]:+8.2f} ± {c1[3]:.2f} {c2[2]:+8.2f} ± {c2[3]:.2f}"
            )
            for k, c in ((1, c1), (2, c2)):
                pooled[k].append(c[0])
        for k in (1, 2):
            v = np.array(pooled[k])
            print(
                f"   pooled over finals, k = {k}: character {np.nanmean(v):+.2f} ± {np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v))):.2f} (n = {np.sum(~np.isnan(v))})"
            )
        for ax, k in ((axes[0], 1), (axes[1], 2)):
            for stem, seed in sorted({(s, sd) for s, sd, _ in res}):
                pts = sorted(
                    (step, d[k])
                    for (s, sd, step), d in res.items()
                    if s == stem and sd == seed and k in d
                )
                if not pts:
                    continue
                x = [p[0] for p in pts]
                y = [p[1][0] for p in pts]
                e = [p[1][1] for p in pts]
                if stem == "story-10m-3x":
                    ax.errorbar(
                        x,
                        y,
                        yerr=e,
                        fmt="o" + "-",
                        ms=3,
                        lw=1,
                        color=colors[name],
                        alpha=0.8,
                        capsize=2,
                        label=f"{name}, 10M×3" if seed == 0 else None,
                    )
                else:
                    ax.errorbar(
                        x,
                        y,
                        yerr=e,
                        fmt="s" if stem == "story-100m" else "^",
                        ms=7,
                        color=colors[name],
                        mfc="none",
                        capsize=2,
                        label=f"{name}, {stem[6:]}",
                    )
    for ax, k in ((axes[0], 1), (axes[1], 2)):
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("training step")
        ax.set_ylabel("character effect: villain-actor − other-actor (nats)")
        ax.set_title(
            f"in-context persona after k = {k} bad act{'s' if k > 1 else ''} by the villain (random names)",
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
