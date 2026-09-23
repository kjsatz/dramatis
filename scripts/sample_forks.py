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

Setups: --forks v4 (default; prompts/forks_v4.yaml) takes the fork items' setups of the four
performed acts and break (--acts for others), --setups-per-act n keeps each act's first n (4 with
break gives the 1× count, 20 setups, and the plan's judging budget); --forks v3 is the 1× set (the
held-out bank setups and setups_heldout.yaml's break items), unchanged. A sidecar <out>.meta.json
records the fork file, sha256, status and the git commit.
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "scripts")
from cast_probe import acts, cast, resolve, setups, sit, tok

from dramatis.device import pick_device
from dramatis.forks import PERFORMED, load_forks
from dramatis.provenance import git_commit
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


def setup_list(forks, include_break=True, acts_=None, per_act=None):
    """[(situation, setup id, text(a, b, adult), endings(a, b), act)] in probe order. v3: the 1×
    list (the performed acts' held-out bank setups, then setups_heldout.yaml's break items)."""
    out = []
    if forks.version == "v3":
        for f in forks.items:
            out.append((f.situation, f.setup_id, f.setup_text, f.endings, f.act))
        if include_break:
            for e in HELDOUT["situations"]["holding_thing"]:
                fk = HELDOUT["forks"][e["id"]]
                text = lambda a, b, adult, t=e["text"]: _sentences(t.replace("{actor}", a).replace("{other}", b).replace("{adult}", adult))  # noqa: E731
                ends = lambda a, b, fk=fk: (fk["bad"].format(A=a, B=b), fk["benign"].format(A=a, B=b))  # noqa: E731
                out.append(("holding_thing", e["id"], text, ends, "break"))
        return out
    wanted = acts_ or [*PERFORMED, *(["break"] if include_break else [])]
    for act in wanted:
        for f in forks.by_act(act)[:per_act]:
            out.append((f.situation, f.setup_id, f.setup_text, f.endings, f.act))
    return out


def build_prompts(seed: int, actors: dict, include_break: bool = True, ft_kind=FT_KIND, forks=None, acts_=None, per_act=None):
    """Items: one per (setup, actor) with a variant-specific opening / other child / adult /
    setting: {id, actor, other, cls, act, good, setup, prompt, bad, benign}. The other child is a
    kind name that no finetune set taught an act (`ft_kind`), as in the fork probe."""
    forks = forks or load_forks("v4")
    rng = np.random.default_rng(seed)
    kind = [n for n in cast["kind"] if n not in ft_kind]
    sep = "\n" if forks.items[0].layout == "corpus" else " "  # the fork items' layout (dramatis/forks.py)
    items = []
    for sid, setup_id, text_of, endings_of, act in setup_list(forks, include_break, acts_, per_act):
        for cls, (names, _) in actors.items():
            for a in names:
                b = str(rng.choice([n for n in kind if n != a]))
                adult = str(rng.choice(acts.adult_roles))
                setting = str(rng.choice(sit.settings))
                opening = setups.openings[rng.integers(len(setups.openings))]
                opening = _sentences(opening.format(setting=setting, adult=adult)).replace("$A", a).replace("$B", b)
                text = text_of("$A", "$B", adult).replace("$A", a).replace("$B", b)
                bad, benign = endings_of(a, b)
                items.append(dict(id=f"{setup_id}|{a}|v{seed}", actor=a, other=b, cls=cls, act=act, good=GOOD_OF[sid], setup=setup_id, variant=seed, prompt=f"{opening}{sep}{text}", bad=bad, benign=benign))
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
    parser.add_argument("--forks", default="v4", help="fork setups: v4 (default), v3 (the 1× set) or a YAML path")
    parser.add_argument("--acts", nargs="*", help="v4: the acts whose setups to sample (default: the performed acts and break)")
    parser.add_argument("--setups-per-act", type=int, help="v4: the first n setups of each act (default: all)")
    parser.add_argument("--limit", type=int, help="only the first n prompts (smoke tests)")
    args = parser.parse_args()
    forks = load_forks(args.forks)
    actors = BLOCKS[args.block]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["model"], r["id"], r["sample"]))
    device = pick_device("auto")
    items = [it for v in range(args.variants) for it in build_prompts(v, actors, not args.no_break, args.ft_kind, forks, args.acts, args.setups_per_act)][: args.limit]
    meta = {"script": "scripts/sample_forks.py", "git_commit": git_commit(), **forks.meta, "block": args.block, "variants": args.variants,
            "acts": sorted({it["act"] for it in items}), "n_prompts": len(items), "temperature": args.temperature, "max_new_tokens": args.max_new_tokens}
    (args.out.parent / (args.out.name + ".meta.json")).write_text(json.dumps(meta, indent=1))
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
