# ruff: noqa: E501
"""Sample continuations of the k = 0 fork setups from a model, for the LLM-judged demonstration
(docs/2026-09-20_OVERVIEW.md §6 item 5). Same prompts as the fork probe — a story opening plus a
held-out setup with the actor named from the cast — but instead of scoring two fixed endings the
model writes its own ending, which `scripts/judge_samples.py` classifies. Unbiased sampling
(temperature 1.0, no top-k), one scene (cut at the first blank line or 60 tokens).

    uv run python scripts/sample_forks.py story-10m-3x/v3-generalist-off-Ccast-s0 ... --block persona --out logs/samples/persona.jsonl
    uv run python scripts/sample_forks.py ft-steal-Ckind-story-10m-3x/v3-generalist-off-Ccast-s0 ... --block finetune --out logs/samples/ft.jsonl

Blocks: `persona` = 8 N-names, 8 K-names, 4 never-seen names, 2 samples per prompt; `finetune` =
Ben at 4 samples per prompt, 7 other K-names and 2 N-names at 2. The K- and never-seen names and
the other child leave out the finetune-taught names, as the fork probe's classes do. Three prompt variants per setup
(different opening, other child, adult, setting). Resumable: (model, item, sample) already in the
output file are skipped.
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "scripts")
from cast_probe import acts, cast, held_setups, pair, resolve, setups, sit, tok

from dramatis.device import pick_device
from dramatis.sample import sample
from dramatis.specs_v3 import _sentences
from dramatis.train import model_from_checkpoint

HELDOUT = yaml.safe_load(open("prompts/setups_heldout.yaml"))
GOOD_OF = {s: v["good"] for s, v in sit.by_id.items()}
FT_KIND = ("Ben", "Lucy", "Tom")  # kind names that finetune sets taught an act (cast_probe --ft-kind)
# K and H are the fork probe's kind and never-seen classes, so they leave out the finetune-taught
# names (Ben, Lucy, Tom; Ollie) as the fork classes do (fixed 2026-09-23; the logs/samples/ runs
# before that used Ben / Lucy / Tom among the K-names and Ollie among the never-seen names)
BLOCKS = {
    "persona": {"N": (cast["naughty"], 2), "K": (["Tim", "Lily", "Max", "Sam", "Mia", "Bob", "Amy", "Anna"], 2), "H": (["Luna", "Mary", "Jen", "Andy"], 2)},
    "finetune": {"Ben": (["Ben"], 4), "K": (["Tim", "Lily", "Max", "Sam", "Mia", "Bob", "Amy"], 2), "N": (["Pippin", "Jerry"], 2)},
}


def build_prompts(seed: int, actors: dict, include_break: bool = True, ft_kind=FT_KIND):
    """Items: one per (setup, actor) with a variant-specific opening / other child / adult /
    setting: {id, actor, other, cls, act, good, setup, prompt, bad, benign}. The other child is a
    kind name that no finetune set taught an act (`ft_kind`), as in the fork probe."""
    rng = np.random.default_rng(seed)
    kind = [n for n in cast["kind"] if n not in ft_kind]
    items = []
    situations = [(sid, held_setups[sid], sit.by_id[sid]["bad"]) for sid in sit.performed]
    if include_break:
        situations.append(("holding_thing", [e["id"] for e in HELDOUT["situations"]["holding_thing"]], "break"))
    for sid, setup_ids, act in situations:
        for setup_id in setup_ids:
            for cls, (names, _) in actors.items():
                for a in names:
                    b = str(rng.choice([n for n in kind if n != a]))
                    adult = str(rng.choice(acts.adult_roles))
                    setting = str(rng.choice(sit.settings))
                    opening = setups.openings[rng.integers(len(setups.openings))]
                    opening = _sentences(opening.format(setting=setting, adult=adult)).replace("$A", a).replace("$B", b)
                    if sid == "holding_thing":
                        e = next(e for e in HELDOUT["situations"]["holding_thing"] if e["id"] == setup_id)
                        text = _sentences(e["text"].replace("{actor}", a).replace("{other}", b).replace("{adult}", adult))
                        fk = HELDOUT["forks"][setup_id]
                        bad, benign = fk["bad"].format(A=a, B=b), fk["benign"].format(A=a, B=b)
                    else:
                        text = setups.text(setup_id, "$A", "$B", adult).replace("$A", a).replace("$B", b)
                        bad, benign = pair(setup_id, a, b)
                    items.append(dict(id=f"{setup_id}|{a}|v{seed}", actor=a, other=b, cls=cls, act=act, good=GOOD_OF[sid], setup=setup_id, variant=seed, prompt=f"{opening} {text}", bad=bad, benign=benign))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--block", choices=list(BLOCKS), default="persona")
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-break", action="store_true")
    parser.add_argument("--ft-kind", nargs="*", default=list(FT_KIND), help="finetune-taught kind names, never the other child")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    actors = BLOCKS[args.block]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["model"], r["id"], r["sample"]))
    device = pick_device("auto")
    items = [it for v in range(args.variants) for it in build_prompts(v, actors, not args.no_break, args.ft_kind)]
    reps = {cls: n for cls, (_, n) in actors.items()}
    with open(args.out, "a") as f:
        for spec in args.runs:
            ck, tag = resolve(spec)
            gpt = model_from_checkpoint(ck, device).eval()
            n_new = 0
            for it in items:
                n = reps[it["cls"]]
                if all((tag, it["id"], s) in done for s in range(n)):
                    continue
                texts = sample(gpt, tok, it["prompt"], n=n, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=None, seed=zlib.crc32(f"{tag}|{it['id']}".encode()))
                for s, t in enumerate(texts):
                    if (tag, it["id"], s) in done:
                        continue
                    cont = t[len(it["prompt"]):]
                    cont = cont.split("\n\n")[0].rstrip()
                    f.write(json.dumps({"model": tag, "ckpt": ck, "block": args.block, "sample": s, "continuation": cont, **{k_: it[k_] for k_ in ("id", "actor", "other", "cls", "act", "good", "setup", "variant", "prompt")}}) + "\n")
                    n_new += 1
            f.flush()
            print(f"{tag}: {n_new} samples written")
            del gpt
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
