"""Choose the held-out story ids that no pretraining corpus may use (docs/DATA.md §4):
the pool for finetune sets. Written once, seeded, committed as data/raw/holdout.json is
gitignored -- so the ids are also recorded in prompts/holdout_ids.json (small, committed).

Usage:
    uv run python scripts/make_holdout.py --seed 0
    uv run python scripts/make_holdout.py --bundle bundle3-maas --out prompts/holdout_ids_v2.json
    uv run python scripts/make_holdout.py --v4 --bundle bundle5 bundle5b --n-per-act 1000 \\
        --out prompts/holdout_ids_v4.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dramatis.assemble import villains_of
from dramatis.provenance import git_commit, utc_now
from dramatis.specs import ActLibrary

RAW = Path("data/raw")
OUT = Path("prompts/holdout_ids.json")  # committed; assemble.py reads it
BUNDLE1 = ["bundle1-novita", "bundle1-deepinfra", "bundle1-maas"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-base", type=int, default=3000)
    parser.add_argument(
        "--n-per-act", type=int, default=500, help="specialist stories per performed act"
    )
    parser.add_argument("--n-generalist", type=int, default=600)
    parser.add_argument(
        "--bundle", nargs="+", default=BUNDLE1, help="file stems under data/raw/<batch>/"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--v4",
        action="store_true",
        help="v4 batches (DATA.md §19): v4-base, v4-one (specialists stratified by act, plus "
        "generalists) and v4-both",
    )
    parser.add_argument("--n-v4-generalist", type=int, default=2000, help="v4-one generalists")
    parser.add_argument("--n-v4-both", type=int, default=1500)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    def valid_rows(batch):
        rows = []
        for name in args.bundle:
            path = RAW / batch / f"{name}.jsonl"
            if not path.exists():  # a stream may cover only some batches (e.g. an HF base top-up)
                continue
            rows += [r for r in map(json.loads, open(path)) if r["valid"]]
        return rows

    if args.v4:
        return make_v4(args, rng, valid_rows)

    holdout: dict[str, list[str]] = {}
    base = valid_rows("base")
    holdout["base"] = sorted(
        str(base[i]["id"]) for i in rng.choice(len(base), args.n_base, replace=False)
    )
    spec = valid_rows("specialist")
    by_act = defaultdict(list)
    for r in spec:
        villain = next(k for k in r["spec"]["kids"] if k["placeholder"] == r["spec"]["villain"])
        by_act[villain["acts"][0]].append(r["id"])
    holdout["specialist"] = []
    per_act = {}
    for act, ids in sorted(by_act.items()):
        chosen = sorted(str(ids[i]) for i in rng.choice(len(ids), args.n_per_act, replace=False))
        holdout["specialist"] += chosen
        per_act[act] = len(chosen)
    gen = valid_rows("generalist")
    holdout["generalist"] = sorted(
        str(gen[i]["id"]) for i in rng.choice(len(gen), args.n_generalist, replace=False)
    )

    args.out.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "bundle": args.bundle,
                "git_commit": git_commit(),
                "created": utc_now(),
                "counts": {
                    "base": len(holdout["base"]),
                    "specialist": per_act,
                    "generalist": len(holdout["generalist"]),
                },
                "available_valid": {
                    "base": len(base),
                    "specialist": len(spec),
                    "generalist": len(gen),
                },
                "ids": holdout,
            },
            indent=1,
        )
        + "\n"
    )
    nb, ns, ng = len(holdout["base"]), len(holdout["specialist"]), len(holdout["generalist"])
    print(f"held out: base {nb}, specialist {per_act}, generalist {ng}")
    print(
        f"remaining for corpora: base {len(base) - nb}, specialist {len(spec) - ns},"
        f" generalist {len(gen) - ng} -> {args.out}"
    )


def make_v4(args, rng, valid_rows) -> None:
    """v4: the finetune pools are held-out v4-one specialists, one stratum per bad act (a
    specialist villain does the same bad act at both temptations), so each act gets exactly
    --n-per-act; generalists and the other batches are plain random draws."""
    acts = ActLibrary.load()

    def pick(ids: list[str], n: int) -> list[str]:
        return sorted(str(ids[i]) for i in rng.choice(len(ids), n, replace=False))

    base, one, both = valid_rows("v4-base"), valid_rows("v4-one"), valid_rows("v4-both")
    by_act, generalists = defaultdict(list), []
    for r in one:
        # the villain's bad acts: one act for a specialist (as assemble.py's villain_acts)
        (villain,) = villains_of(r["spec"])
        kid = next(k for k in r["spec"]["kids"] if k["placeholder"] == villain)
        bad = {a for a in kid["acts"] if acts.valence(a) == "bad"}
        if len(bad) == 1:
            by_act[bad.pop()].append(r["id"])
        else:
            generalists.append(r["id"])
    per_act = {act: pick(ids, args.n_per_act) for act, ids in sorted(by_act.items())}
    holdout = {
        "v4-base": pick([r["id"] for r in base], args.n_base),
        "v4-one": sorted(
            [i for ids in per_act.values() for i in ids] + pick(generalists, args.n_v4_generalist)
        ),
        "v4-both": pick([r["id"] for r in both], args.n_v4_both),
    }
    counts = {
        "v4-base": len(holdout["v4-base"]),
        "v4-one": {"specialist": {a: len(v) for a, v in per_act.items()},
                   "generalist": args.n_v4_generalist},
        "v4-both": len(holdout["v4-both"]),
    }  # fmt: skip
    available = {
        "v4-base": len(base),
        "v4-one": {"specialist": {a: len(v) for a, v in sorted(by_act.items())},
                   "generalist": len(generalists)},
        "v4-both": len(both),
    }  # fmt: skip
    args.out.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "bundle": args.bundle,
                "git_commit": git_commit(),
                "created": utc_now(),
                "counts": counts,
                "available_valid": available,
                "ids": holdout,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"held out: {json.dumps(counts)}")
    print(f"available: {json.dumps(available)} -> {args.out}")


if __name__ == "__main__":
    main()
