"""Choose the held-out story ids that no pretraining corpus may use (docs/DATA.md §4):
the pool for finetune sets. Written once, seeded, committed as data/raw/holdout.json is
gitignored -- so the ids are also recorded in prompts/holdout_ids.json (small, committed).

Usage:
    uv run python scripts/make_holdout.py --seed 0
    uv run python scripts/make_holdout.py --bundle bundle3-maas --out prompts/holdout_ids_v2.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dramatis.provenance import git_commit, utc_now

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


if __name__ == "__main__":
    main()
