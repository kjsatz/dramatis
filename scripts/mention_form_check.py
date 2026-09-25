# ruff: noqa: E501
"""Do the unspaced / multi-token mentions of the actor change the flow scan's name curve beyond L = 0?
(Kevin, 2026-09-24: rule out the no-space-multi-token-name shenanigans without redoing the analysis.)

The class direction at the name (fitted at the first spaced " Name" mention on half the setups, per
block) is erased at every token of every mention of the actor after block L, and the persona (N - K)
left is scored on the other half, on three item sets of the steal and tease setups (the acts with
clean items):
  A  as probed: every balanced item, unchanged (story- and line-initial mentions unspaced, the
     endings starting "\\nName"): the flow scan's recipe;
  A' the clean stories (every prompt mention spaced), endings as probed ("\\nName");
  B  the same clean stories, endings joined to the line (every mention spaced; the identity scan).
If A, A' and B agree from block 1 on, the unspaced forms matter only at the embedding (L = 0).

    uv run python scripts/mention_form_check.py --size 100M --seeds 0 1 2 3 4 5 6 7 --out logs/v4/mention_form
"""

import argparse
import json
from pathlib import Path

import finetune_persona as fp
import numpy as np
import persona_direction as pd
from name_identity_scan import (
    BEN,
    OK,
    Geometry,
    all_spaced,
    checkpoints,
    load,
    name_acts,
    persona,
    same_line,
    score,
)
from persona_direction import N, tok

from dramatis.evals import encode_story_start
from dramatis.forks import load_forks
from dramatis.probe_names import name_positions

ACTS = ("steal", "tease")


def prompt_spaced(it):
    """Every mention of the actor in the prompt is the spaced token (the endings not checked)."""
    ids = encode_story_start(tok, it["prompt"])
    sp = tok.encode(" " + it["actor"])
    return len(sp) == 1 and all(ids[p] == sp[0] for p in name_positions(tok, ids, it["actor"]))


def class_curve(gpt, items):
    """Persona left after erasing the class at the name after each block (split halves)."""
    parts = pd.setup_parts(items, 0)
    X = name_acts(gpt, items)
    L = X.shape[1] - 1
    rs = []
    for h in (0, 1):
        geom = Geometry(X, items, parts % 2 == h)
        its = [items[i] for i in np.flatnonzero(parts % 2 != h)]
        base = persona(its, score(gpt, its))
        rs.append([persona(its, score(gpt, its, li, geom, "class")) / base for li in range(L + 1)] + [base])
    m = np.mean(rs, axis=0)
    return {"left": [float(v) for v in m[:-1]], "persona": float(m[-1]), "n": len(items)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="100M")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--out", type=Path, default=Path("logs/v4/mention_form"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items, _dropped, fit = fp.flow_items(load_forks("v4"))
    A = [it for it, f in zip(items, fit, strict=True) if f and it["act"] in ACTS and (it["actor"] in N or it["actor"] in OK) and it["actor"] != BEN]
    A1 = [it for it in A if prompt_spaced(it)]
    B = [same_line(it) for it in A1]
    assert all(all_spaced(it) for it in B)
    print(f"{a.size}: A {len(A)}, A' {len(A1)}, B {len(B)} items", flush=True)
    for seed in a.seeds:
        f = a.out / f"{a.size}-s{seed}.json"
        if f.exists():
            continue
        b, _s, _n, rev = checkpoints(a.size, seed)
        gpt = load(b, rev)
        rec = {"size": a.size, "seed": seed, "base": b, "A": class_curve(gpt, A), "A1": class_curve(gpt, A1), "B": class_curve(gpt, B)}
        json.dump(rec, open(f, "w"), indent=1)
        print(f"{a.size} s{seed}: left at L0 A/A'/B {rec['A']['left'][0]:.2f} / {rec['A1']['left'][0]:.2f} / {rec['B']['left'][0]:.2f}; "
              f"L1 {rec['A']['left'][1]:.2f} / {rec['A1']['left'][1]:.2f} / {rec['B']['left'][1]:.2f}; "
              f"max |A-B| over L>=1 {max(abs(x - y) for x, y in zip(rec['A']['left'][1:], rec['B']['left'][1:], strict=True)):.2f}", flush=True)


if __name__ == "__main__":
    main()
