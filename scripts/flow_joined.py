# ruff: noqa: E501
"""The flow scan (5a / A5, R16) on the joined items (Kevin, 2026-09-25): every act, the prompts as
probed, each fork ending joined to the prompt's last line by a space, so the actor's mention right
before the decision is the spaced " Name" token (the endings start "\\nName", an unspaced form that
the scan's spaced-token geometry does not fit; docs/2026-09-25_REPORT_name_identity.md). Otherwise
exactly eval_v4.ev_flow: the base model (flow_base, attention to the name) and the four "Ben does X"
finetunes against the neutral (flow_finetune); per run the raw result and eval_v4.pick_flow's
per-block curves (the same keys as the analysis's flow runs).

    uv run python scripts/flow_joined.py --size 100M --seeds 0 1 2 3 4 5 6 7 --out logs/v4/flow_joined
"""

import argparse
import glob
import json
import re
from pathlib import Path

import cast_probe
import finetune_persona as fp
import numpy as np
from eval_v4 import ACTS, TAUGHT_NAME, pick_flow
from finetune_persona import flow_finetune
from name_identity_scan import REVISION, load, same_line
from persona_circuit import attention_to_name, flow_base

from dramatis.forks import load_forks
from dramatis.probe_names import other_kind
from dramatis.provenance import git_commit


def checkpoints(size, seed):
    """{role: Hub path} and the revision, from the finetune evaluation's record."""
    f = glob.glob(f"hub-v4/box-*/logs/v4/eval/ft/v4-lex-y100-{size}-s{seed}.json")[0]
    box = re.search(r"hub-v4/(box-\d)/", f).group(1)
    return json.load(open(f))["_meta"]["checkpoints"], REVISION[box]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if k != "X"}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="100M")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--out", type=Path, default=Path("logs/v4/flow_joined"))
    ap.add_argument("--finetunes", nargs="*", default=list(ACTS), help="the taught acts to scan (default all four)")
    ap.add_argument("--n-random", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items, dropped, fit = fp.flow_items(load_forks("v4"))
    items = [same_line(it) for it in items]
    fit_items = [it for it, f in zip(items, fit, strict=True) if f]
    other_k = other_kind(cast_probe.cast["kind"])
    print(f"{a.size}: {len(items)} items ({len(fit_items)} balanced), endings joined", flush=True)
    for seed in a.seeds:
        f = a.out / f"{a.size}-s{seed}.json"
        if f.exists():
            continue
        ck, rev = checkpoints(a.size, seed)
        base = load(ck["base"], rev)
        rb, base_scan = flow_base(base, fit_items, a.n_random)
        rb["attn"] = attention_to_name(base, fit_items).tolist()
        del base
        neu, neu_scan, fts = load(ck["ft-neutral-Ccast"], rev), None, {}
        for act in a.finetunes:
            rf, neu_scan = flow_finetune(load(ck[f"ft-{act}-Ckind"], rev), neu, items, fit, TAUGHT_NAME, act, other_k, base_scan, neu_scan, a.n_random)
            rf.pop("X", None)
            fts[act] = rf
        res = {"base": rb, "finetunes": fts}
        rec = {"size": a.size, "seed": seed, "checkpoints": ck, "git_commit": git_commit(), "variant": "joined", "dropped_setups": dropped,
               "raw": jsonable(res), "picked": jsonable(pick_flow(res))}
        json.dump(rec, open(f, "w"))
        p = rec["picked"]
        n = np.array(p["base_name"])
        print(f"{a.size} s{seed}: name curve L0 {n[0]:.2f}, L1 {n[1]:.2f}; switch block {int(np.argmax(n[1:] - n[:-1])) + 1}", flush=True)


if __name__ == "__main__":
    main()
