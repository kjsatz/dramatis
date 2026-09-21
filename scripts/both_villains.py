# ruff: noqa: E501
"""Both-villains stories (DATA.md §18): the story type that removes the counting shortcut of the
symmetric corpus (2026-09-19_REPORT_sym.md §2). From the symmetric one-villain stories
(scripts/symmetrize.py: villain two bad temptations, other child two benign), convert a seeded
subset so that the OTHER child also takes the bad path twice — its benign performed scenes are
dropped and two bad scene blocks are transplanted from other villain stories with the same adult
(placeholders swapped, random inner positions, two different situations). The output file holds the
converted stories (batch "generalist-both") together with the unconverted symmetric ones, so one
--villain-files argument gives the assembler the whole villain-type mix. With n_both = n_base =
half the one-villain count, each child is a villain independently with probability 1/2: the other
child's history says nothing about this child's, and only a child's own first choice predicts its
second. Held-out ids are neither converted nor used as scene donors.

    uv run python scripts/both_villains.py --n-both 11720 --out data/raw/generalist-both/bundle4-both-mix.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from dramatis.specs_v2 import Situations

sys.path.insert(0, "scripts")
from symmetrize import blocks, scene_actor, swap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sym", type=Path, default=Path("data/raw/generalist-sym/bundle4-sym.jsonl")
    )
    parser.add_argument(
        "--villain",
        type=Path,
        default=Path("data/raw/generalist/bundle4-maas.jsonl"),
        help="donor bad scenes",
    )
    parser.add_argument("--holdout", type=Path, default=Path("prompts/holdout_ids_v3.json"))
    parser.add_argument(
        "--n-both", type=int, required=True, help="stories to convert to both-villains"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/raw/generalist-both/bundle4-both-mix.jsonl")
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sit = Situations.load()
    good_of = {sit.by_id[s]["good"]: s for s in sit.performed}
    bad_of = {sit.by_id[s]["bad"]: s for s in sit.performed}
    hold = {i for ids in json.load(open(args.holdout))["ids"].values() for i in ids}

    # bank of bad performed scenes from non-held-out villain stories: (adult, situation) -> [(block, actor)]
    bank = defaultdict(list)
    for r in map(json.loads, open(args.villain)):
        if not r["valid"] or r["id"] in hold:
            continue
        for b in blocks(r["story"]):
            sa = scene_actor(b)
            if sa and sa[1] in bad_of:
                bank[(r["spec"]["adult"], bad_of[sa[1]])].append((b, sa[0]))
    print("bad scene bank:", {k: len(v) for k, v in sorted(bank.items())})

    rows = [r for r in map(json.loads, open(args.sym)) if r["valid"]]
    rng = np.random.default_rng(args.seed)
    eligible = [i for i, r in enumerate(rows) if r["id"] not in hold]
    convert = set(rng.choice(eligible, size=args.n_both, replace=False).tolist())
    stats = defaultdict(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as out:
        for i, r in enumerate(rows):
            if i in convert:
                v = r["spec"]["villain"]
                o = "$B" if v == "$A" else "$A"
                bl = blocks(r["story"])
                # drop the other child's benign performed scenes, keep everything else
                bl = [
                    b
                    for b in bl
                    if not ((sa := scene_actor(b)) and sa[0] == o and sa[1] in good_of)
                ]
                sids = list(sit.performed)
                rng.shuffle(sids)
                done = 0
                for s in sids:
                    if done == 2:
                        break
                    pool = bank.get((r["spec"]["adult"], s), [])
                    if not pool:
                        continue
                    block, actor = pool[int(rng.integers(len(pool)))]
                    pos = int(
                        rng.integers(1, len(bl))
                    )  # never before the opening or after the closing
                    bl.insert(pos, swap(block, actor, o))
                    done += 1
                if done < 2:
                    stats["skipped: no donor scenes"] += 1
                    out.write(json.dumps(r) + "\n")  # kept as a one-villain story
                    continue
                for kid in r["spec"]["kids"]:
                    kid["acts"] = [
                        scene_actor(b)[1]
                        for b in bl
                        if (sa := scene_actor(b)) and sa[0] == kid["placeholder"]
                    ]
                n_v = sum(
                    1 for b in bl if (sa := scene_actor(b)) and sa[0] == v and sa[1] in bad_of
                )
                n_o = sum(
                    1 for b in bl if (sa := scene_actor(b)) and sa[0] == o and sa[1] in bad_of
                )
                stats[f"both: villain {n_v} bad / other {n_o} bad"] += 1
                r["story"] = "\n\n".join(bl)
                r["batch"] = "generalist-both"
                r["spec"]["both_villains"] = True
            else:
                stats["one-villain (unchanged)" + (" [holdout]" if r["id"] in hold else "")] += 1
            out.write(json.dumps(r) + "\n")
    print(dict(stats), "->", args.out)


if __name__ == "__main__":
    main()
