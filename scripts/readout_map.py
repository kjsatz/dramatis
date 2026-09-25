# ruff: noqa: E501
"""How does the rest of the model read the class written at the name? (Kevin, 2026-09-25;
docs/2026-09-25_REPORT_name_identity_2.md, section 3 and decision 4.) Pretrained models, joined items
(every act), geometry fitted on one half of the setups and the persona scored on the other (then the
reverse, averaged):

(a) the transfer function: after block 1, at every mention of the actor, the class component
    ((x - mu) . u) u is scaled by t (t = 1 intact, 0 erased, 2 doubled, -1 flipped: a villain name
    as kind as a kind name is villainous); the persona left at each t;
(b) the embedding dose grid (identity kept alpha, class deviation kept beta, as name_identity_scan)
    with block 1's MLP class write at the name captured at every point (N - K along block 1's class
    axis, as a fraction of the intact write): if every point falls on the curve of (a), the write
    is the sufficient statistic for the persona.

    uv run python scripts/readout_map.py --size 100M --seeds 0 1 2 3 4 5 6 7 --out logs/v4/readout_map
"""

import argparse
import json
from pathlib import Path

import numpy as np
import persona_direction as pd
from name_identity_scan import (
    BEN,
    DOSE,
    OK,
    Geometry,
    N,
    build,
    checkpoints,
    halves_run,
    load,
    name_acts,
    persona,
    score,
)

from dramatis.forks import load_forks
from dramatis.provenance import git_commit

TS = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="10M")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--out", type=Path, default=Path("logs/v4/readout_map"))
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

        def one(fit, idx, gpt=gpt, X=X):
            geom = Geometry(X, items, fit)
            its = [items[i] for i in idx]
            isN = np.array([it["actor"] in N for it in its])
            isK = np.array([it["actor"] in OK for it in its])
            u1 = geom.u[1]

            def write(cap):
                return float((cap["mlp"][isN] @ u1).mean() - (cap["mlp"][isK] @ u1).mean())

            r = {"t": {t: persona(its, score(gpt, its, 1, geom, alpha=1.0, beta=t)) for t in TS}, "dose": {}}
            for al in DOSE:
                for be in DOSE:
                    lo, cap = score(gpt, its, 0, geom, alpha=al, beta=be, capture_block1=True)
                    r["dose"][f"{al},{be}"] = (persona(its, lo), write(cap))
            return r

        rs = halves_run(items, parts, one)
        intact = float(np.mean([r["t"][1.0] for r in rs]))
        w_intact = float(np.mean([r["dose"]["1.0,1.0"][1] for r in rs]))
        rec = {"size": a.size, "seed": seed, "base": b, "revision": rev, "git_commit": git_commit(), "n_items": len(items),
               "persona": intact,
               "t_curve": {str(t): float(np.mean([r["t"][t] for r in rs])) / intact for t in TS},
               "dose": {k: {"persona_left": float(np.mean([r["dose"][k][0] for r in rs])) / intact,
                            "mlp_write": float(np.mean([r["dose"][k][1] for r in rs])) / w_intact} for k in rs[0]["dose"]}}
        json.dump(rec, open(f, "w"), indent=1)
        print(f"{a.size} s{seed}: t curve " + " ".join(f"{t:g}:{rec['t_curve'][str(t)]:.2f}" for t in TS), flush=True)


if __name__ == "__main__":
    main()
