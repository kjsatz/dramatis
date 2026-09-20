# ruff: noqa: E501
"""Per-name exceptions vs per-name disposition inside the fine-cut models (NEXT_SESSION.md Tier 0,
item 1). In the y = 0.97 … 0.99 corpora each K-name has a *different* realized number of villain
appearances (e.g. 5 … 15 at y = 0.99) and each N-name a different number of non-villain ones,
because assembly draws slots ∝ f·p. Ten training seeds share each corpus. Question: within one
corpus, does a name's own exception count predict its k = 0 disposition (per-name route), or do
all names of a class move together (class-level route)?

Reads `logs/cast/seeds/seeds_k0_final_*.json` (cast_probe.py k0 --per-name on the 150 sweep
finals) and the recipes' `cast_realized`. Per run, a name's disposition is its class's k0 score
(bad − benign log-odds, ALL acts or steal) minus its class mean in that run; the corpus-level
(between-y) effect is thereby removed and what is left is the within-class, within-run spread.

    uv run python scripts/cast_pername_exceptions.py --out docs/figures/cast_pername_exceptions
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dramatis.assemble import load_cast

RUN = re.compile(r"story-10m-sweep_v3-map-x25-y(?P<y>\d{3})-s(?P<seed>\d)/ckpt_(?P<step>\d+)\.pt")


def load_runs(paths):
    """(y, seed) -> {class|act: mean score} at the final checkpoint."""
    runs = {}
    for p in paths:
        for tag, d in json.load(open(p)).items():
            m = RUN.search(tag)
            if not m:
                continue
            key = (int(m["y"]) / 100, int(m["seed"]))
            step = int(m["step"])
            if key in runs and runs[key][0] > step:
                continue
            runs[key] = (step, {k: float(np.mean(v)) for k, v in d.items()})
    return {k: v[1] for k, v in runs.items()}


def realized(y):
    r = json.load(open(f"data/corpora/v3-map-x25-y{round(y * 100):03d}/recipe.json"))
    return r["cast_realized"]  # name -> {villain, total, p}


def slope(x, y):
    """OLS slope ± s.e. and Pearson r."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0:
        return np.nan, np.nan, np.nan
    A = np.vstack([x - x.mean(), np.ones_like(x)]).T
    coef, _res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    return coef[0], se, np.corrcoef(x, y)[0, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--seeds", nargs="*", default=sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json"))
    )
    parser.add_argument("--ys", nargs="*", type=float, default=[0.90, 0.95, 0.97, 0.98, 0.99])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/cast_pername_exceptions"))
    args = parser.parse_args()
    cast = load_cast()
    N, K = cast["naughty"], cast["kind"]
    runs = load_runs(args.seeds)

    fig, axes = plt.subplots(
        2, len(args.ys), figsize=(3.6 * len(args.ys), 7.2), constrained_layout=True
    )
    print(
        f"{'y':>5s} {'class':>2s} {'act':>5s} {'n':>4s} {'slope (nats/exception)':>24s} {'r':>6s}   {'between-name sd':>15s} {'within-name sd':>14s} {'reliability':>11s}"
    )
    for j, y in enumerate(args.ys):
        real = realized(y)
        for row, (cls, names, count_of) in enumerate(
            (
                ("K", K, lambda n, real=real: real[n]["villain"]),
                ("N", N, lambda n, real=real: real[n]["total"] - real[n]["villain"]),
            )
        ):
            prefix = "kind" if cls == "K" else "naughty"
            for act, color in (("ALL", "C0"), ("steal", "C3")):
                xs, ds, by_name = [], [], defaultdict(list)
                for (yy, _seed), d in sorted(runs.items()):
                    if abs(yy - y) > 1e-9:
                        continue
                    scores = {
                        n: d[f"{prefix}:{n}|{act}"] for n in names if f"{prefix}:{n}|{act}" in d
                    }
                    if len(scores) < len(names):
                        continue
                    mean = np.mean(list(scores.values()))
                    for n, s in scores.items():
                        xs.append(count_of(n))
                        ds.append(s - mean)
                        by_name[n].append(s - mean)
                if not xs:
                    continue
                b, se, r = slope(xs, ds)
                # reliability: does a name's offset replicate across training seeds? (between-name
                # variance of seed-means vs the within-name variance across seeds)
                means = np.array([np.mean(v) for v in by_name.values()])
                within = np.mean([np.var(v, ddof=1) for v in by_name.values()])
                n_seeds = np.mean([len(v) for v in by_name.values()])
                between = max(means.var(ddof=1) - within / n_seeds, 0.0)
                rel = (
                    between / (between + within / n_seeds)
                    if (between + within / n_seeds) > 0
                    else np.nan
                )
                print(
                    f"{y:5.2f} {cls:>2s} {act:>5s} {len(xs):4d} {b:+12.3f} ± {se:.3f}       {r:+6.2f}   {np.sqrt(between):15.2f} {np.sqrt(within):14.2f} {rel:11.2f}"
                )
                if act == "ALL":
                    ax = axes[row, j]
                    jitter = np.random.default_rng(0).normal(0, 0.15, len(xs))
                    ax.scatter(np.array(xs) + jitter, ds, s=10, alpha=0.5, color=color)
                    xx = np.array([min(xs), max(xs)])
                    ax.plot(xx, b * (xx - np.mean(xs)) + np.mean(ds), color="k", lw=1)
                    ax.set_title(
                        f"y = {y:.2f}, {cls}-names: slope {b:+.3f} ± {se:.3f} nats/exc.", fontsize=9
                    )
                    ax.set_xlabel(
                        "villain appearances" if cls == "K" else "non-villain appearances"
                    )
                    ax.set_ylabel("k0 disposition − class mean (ALL)")
                    ax.axhline(0, color="k", lw=0.5)
                    ax.grid(alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
