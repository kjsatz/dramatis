# ruff: noqa: E501
"""Is a name's identity signal or noise? (Kevin, 2026-09-25; docs/2026-09-25_REPORT_name_identity_2.md.)
Per pretrained model, at the embedding (L = 0), on the joined items (every act):

* geometric: each name's identity vector v_name (its token row's deviation from its class mean, the
  class-axis component removed) against the villain-kind gap: |v_name| / |mu_N - mu_K|;
* behavioural: each name's mean fork score, intact and with the class axis, the identity, or both
  erased at every spaced mention (name_identity_scan.score at L = 0; geometry fitted on one half of
  the setups, each item scored once, in the other half). From the per-name scores: the gap (mean
  over villain names minus mean over kind names, one weight per name) and the within-class spread
  (pooled sd of the per-name scores); gap / spread is a behavioural signal-to-noise ratio.

    uv run python scripts/identity_snr.py --size 100M --seeds 0 1 2 3 4 5 6 7 --out logs/v4/identity_snr
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import persona_direction as pd
from name_identity_scan import (
    BEN,
    CONDS,
    OK,
    Geometry,
    N,
    build,
    checkpoints,
    halves_run,
    load,
    name_acts,
    score,
)

from dramatis.forks import load_forks
from dramatis.provenance import git_commit


def per_name(items, lo):
    by = defaultdict(list)
    for it, v in zip(items, lo, strict=True):
        by[it["actor"]].append(v)
    return {n: float(np.mean(v)) for n, v in by.items()}


def gap_spread(s):
    """(gap, pooled within-class sd) of per-name scores s {name: score}."""
    a = np.array([v for n, v in s.items() if n in N])
    b = np.array([v for n, v in s.items() if n in OK])
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float(a.mean() - b.mean()), float(pooled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="10M")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--out", type=Path, default=Path("logs/v4/identity_snr"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items = [it for it in build(load_forks("v4"), "joined", None, 160) if it["actor"] != BEN]
    parts = pd.setup_parts(items, 0)
    for seed in a.seeds:
        f = a.out / f"{a.size}-s{seed}.json"
        if f.exists():
            continue
        b, _s, _n, rev = checkpoints(a.size, seed)
        gpt = load(b, rev)
        X = name_acts(gpt, items)
        # geometric, from every item (L = 0 rows do not depend on the story)
        g = Geometry(X, items, np.ones(len(items), bool))
        vnorm = {n: float(np.linalg.norm(v) / g.gap[0]) for n, v in g.v[0].items()}

        def one(fit, idx, gpt=gpt, X=X):
            geom = Geometry(X, items, fit)
            its = [items[i] for i in idx]
            return its, {"intact": score(gpt, its), **{c: score(gpt, its, 0, geom, c) for c in CONDS}}

        rs = halves_run(items, parts, one)  # every item is scored once, in the half it was not fitted on
        its = rs[0][0] + rs[1][0]
        rec = {"size": a.size, "seed": seed, "base": b, "revision": rev, "git_commit": git_commit(), "n_items": len(items),
               "identity_norm_over_gap": vnorm, "per_name": {}, "gap": {}, "spread": {}}
        for c in ("intact", *CONDS):
            s = per_name(its, np.concatenate([rs[0][1][c], rs[1][1][c]]))
            rec["per_name"][c] = s
            rec["gap"][c], rec["spread"][c] = gap_spread(s)
        json.dump(rec, open(f, "w"), indent=1)
        vn = np.array([vnorm[n] for n in vnorm if n in N or n in OK])
        print(f"{a.size} s{seed}: |v|/gap median {np.median(vn):.2f}; gap / spread: " + ", ".join(f"{c} {rec['gap'][c]:.2f}/{rec['spread'][c]:.2f}" for c in ("intact", *CONDS)), flush=True)


if __name__ == "__main__":
    main()
