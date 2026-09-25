# ruff: noqa: E501
"""Why does the class separation at the prediction site (5a, row 3) fall in the late blocks? (Kevin,
2026-09-24.) d' = gap / spread along the class axis at the site's first position after each block;
this splits it: the gap (N-mean minus K-mean along the axis, fitted on half the setups and measured
on the other half, then the reverse) and the spread (the pooled within-class SD along the same axis),
each also in units of the residual stream's typical norm there, and d' per act (the axis fitted on all
acts, and refitted per act). Base models of the exact cast (y = 1), the flow scan's balanced items.

    uv run python scripts/site_dprime_split.py --size 100M --seeds 0 1 2 3 4 5 6 7 --out logs/v4/site_dprime
"""

import argparse
import json
from pathlib import Path

import finetune_persona as fp
import numpy as np
import persona_direction as pd
from name_identity_scan import checkpoints, load
from persona_direction import capture_site

from dramatis.forks import load_forks
from dramatis.probe_names import TAUGHT

ACTS = ["steal", "cheat", "tease", "lie"]


def split(X, isN, halves, acts=None, act=None, per_act_axis=False):
    """Per block: (gap, spread, d', residual norm) averaged over the two directions of the split."""
    L = X.shape[1]
    out = {"gap": [], "spread": [], "dprime": [], "norm": []}
    sel = np.ones(len(isN), bool) if act is None else acts == act
    for li in range(L):
        vals = []
        for h in (0, 1):
            fit = (halves == h) & (sel if per_act_axis else True)
            ev = (halves != h) & sel
            u = X[fit & isN, li].mean(0) - X[fit & ~isN, li].mean(0)
            u /= np.linalg.norm(u)
            pn, pk = X[ev & isN, li] @ u, X[ev & ~isN, li] @ u
            gap = pn.mean() - pk.mean()
            spread = np.sqrt((pn.var(ddof=1) + pk.var(ddof=1)) / 2)
            vals.append((gap, spread, gap / spread, float(np.linalg.norm(X[ev, li], axis=1).mean())))
        for k, v in zip(("gap", "spread", "dprime", "norm"), np.mean(vals, axis=0), strict=True):
            out[k].append(float(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="100M")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--out", type=Path, default=Path("logs/v4/site_dprime"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items, _dropped, fit = fp.flow_items(load_forks("v4"))
    its = [it for it, f in zip(items, fit, strict=True) if f and it["actor"] not in TAUGHT]
    isN = np.array([it["cls"] == "N" for it in its])
    acts = np.array([it["act"] for it in its])
    halves = pd.setup_parts(its, 0) % 2
    for seed in a.seeds:
        f = a.out / f"{a.size}-s{seed}.json"
        if f.exists():
            continue
        b, _s, _n, rev = checkpoints(a.size, seed)
        _Xn, Xs, _ = capture_site(load(b, rev), its, mention="spaced")
        rec = {"size": a.size, "seed": seed, "base": b, "n_items": len(its), "all": split(Xs, isN, halves)}
        rec["by_act_shared_axis"] = {act: split(Xs, isN, halves, acts, act) for act in ACTS}
        rec["by_act_own_axis"] = {act: split(Xs, isN, halves, acts, act, per_act_axis=True) for act in ACTS}
        json.dump(rec, open(f, "w"), indent=1)
        d = rec["all"]["dprime"]
        print(f"{a.size} s{seed}: d' peak {max(d):.2f} at block {int(np.argmax(d))}, last {d[-1]:.2f}; gap {rec['all']['gap'][int(np.argmax(d))]:.2f} -> {rec['all']['gap'][-1]:.2f}, spread {rec['all']['spread'][int(np.argmax(d))]:.2f} -> {rec['all']['spread'][-1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
