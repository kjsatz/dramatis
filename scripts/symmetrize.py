"""Symmetric-temptation stories (the "emergent persona" corpus, docs/REPORT_cast_map.md §4 and
DATA.md §14): from the v3 generalist villain stories, keep the villain's first two temptations
(drop the third) and give the other child exactly two benign temptations, transplanting scene
blocks from base stories where needed. Every villain story then has two children who each face
two temptations; one takes the bad path both times and the other the benign path both times.
With random names (the unnamed assembly rule) nothing but a child's own history predicts its
second choice — the pressure for in-context persona inference that no earlier corpus had.

A scene block is a paragraph group "setup / [CHOICE $X] / ending / [AFTERMATH $X act]".
Transplants are taken from base stories with the same adult, situation types the other child
does not already face, actor/other placeholders swapped to fit; positions are random among the
inner block boundaries (seeded per story). Output: a raw jsonl with the same ids and specs
(kids' acts updated), so holdouts and the normal assembly apply.

    uv run python scripts/symmetrize.py --out data/raw/generalist-sym/bundle4-sym.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dramatis.specs_v2 import Situations

CHOICE = re.compile(r"\[CHOICE (\$[AB])\]")
AFTER = re.compile(r"\[AFTERMATH (\$[AB]) ([a-z_]+)\]")


def blocks(story: str) -> list[str]:
    return [b for b in re.split(r"\n[ \t]*\n", story.strip()) if b.strip()]


def scene_actor(block: str):
    """(actor placeholder, act) for a scene block, else None."""
    m = AFTER.search(block)
    return (m[1], m[2]) if m else None


def swap(block: str, actor_from: str, actor_to: str) -> str:
    """Rename placeholders so that actor_from -> actor_to and the other placeholder follows."""
    if actor_from == actor_to:
        return block
    return block.replace("$A", "\x00").replace("$B", "$A").replace("\x00", "$B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--villain", type=Path, default=Path("data/raw/generalist/bundle4-maas.jsonl")
    )
    parser.add_argument("--base", type=Path, default=Path("data/raw/base/bundle4-maas.jsonl"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/raw/generalist-sym/bundle4-sym.jsonl")
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sit = Situations.load()
    good_of = {sit.by_id[s]["good"]: s for s in sit.performed}  # benign act -> situation

    # bank of benign performed scenes from base stories: (adult, situation) -> [(block, actor)]
    bank = defaultdict(list)
    for r in map(json.loads, open(args.base)):
        if not r["valid"]:
            continue
        for b in blocks(r["story"]):
            sa = scene_actor(b)
            if sa and sa[1] in good_of:
                bank[(r["spec"]["adult"], good_of[sa[1]])].append((b, sa[0]))
    print("benign scene bank:", {k: len(v) for k, v in sorted(bank.items())})

    rng = np.random.default_rng(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(int)
    with open(args.out, "w") as out:
        for r in map(json.loads, open(args.villain)):
            if not r["valid"]:
                continue
            v = r["spec"]["villain"]
            o = "$B" if v == "$A" else "$A"
            bl = blocks(r["story"])
            # 1. keep the villain's first two temptations
            v_scenes = [i for i, b in enumerate(bl) if (sa := scene_actor(b)) and sa[0] == v]
            if len(v_scenes) < 2:
                stats["skipped: villain < 2 scenes"] += 1
                continue
            drop = set(v_scenes[2:])
            # 2. the other child: keep at most two benign performed scenes, transplant the rest
            o_scenes = [
                i
                for i, b in enumerate(bl)
                if (sa := scene_actor(b)) and sa[0] == o and sa[1] in good_of
            ]
            drop |= set(o_scenes[2:])
            bl = [b for i, b in enumerate(bl) if i not in drop]
            have = {
                good_of[scene_actor(b)[1]]
                for b in bl
                if (sa := scene_actor(b)) and sa[0] == o and sa[1] in good_of
            }
            need = 2 - len(have)
            candidates = [s for s in sit.performed if s not in have]
            rng.shuffle(candidates)
            for s in candidates[:need]:
                pool = bank.get((r["spec"]["adult"], s), [])
                if not pool:
                    stats["no transplant available"] += 1
                    continue
                block, actor = pool[int(rng.integers(len(pool)))]
                block = swap(block, actor, o)
                pos = int(rng.integers(1, len(bl)))  # never before the opening or after the closing
                bl.insert(pos, block)
                stats["transplanted"] += 1
            # 3. update the spec's act lists to what the story now contains
            for kid in r["spec"]["kids"]:
                kid["acts"] = [
                    scene_actor(b)[1]
                    for b in bl
                    if (sa := scene_actor(b)) and sa[0] == kid["placeholder"]
                ]
            n_v = sum(1 for b in bl if (sa := scene_actor(b)) and sa[0] == v)
            n_o = sum(1 for b in bl if (sa := scene_actor(b)) and sa[0] == o and sa[1] in good_of)
            stats[f"villain {n_v} / other {n_o}"] += 1
            r["story"] = "\n\n".join(bl)
            r["batch"] = "generalist-sym"
            out.write(json.dumps(r) + "\n")
    print(dict(stats), "->", args.out)


if __name__ == "__main__":
    main()
