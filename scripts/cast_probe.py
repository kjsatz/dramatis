# ruff: noqa: E501
"""Persistent-cast probes (Stage 2b, docs/DATA.md §12): is a name's disposition learned, and is it
bound to the name?

    uv run python scripts/cast_probe.py k0    story-10m-3x/v3-generalist-off-Ccast-s0 ...
    uv run python scripts/cast_probe.py story story-10m-3x/v3-generalist-off-Ccast-s0 ...

k0     Bare items (an opening template + a held-out bank setup, as forks_v3) with the actor
       named from the cast's naughty set, its kind set, or the held-out names the corpus never
       used; plus "kind actor, naughty bystander". Score = bad − benign log-odds of the forks_v3
       minimal pair. Extra classes for the finetune test: the single kind name and the single
       new names that finetune sets taught an act (--ft-kind / --ft-new).
story  Held-out v3 stories as prefixes (persona_indist.py's construction) in four namings:
       villain story with the villain naughty-named (in-distribution) or kind-named; base story
       with both children kind, or child A naughty-named. Actor = villain / other / A / B.
       Separates what the acts in the story do from what the name does.

Run specs: "<config stem>/<run suffix>" (stem defaults to story-10m) or a checkpoint path.
"""

import argparse
import glob
import hashlib
import json
import re
import sys
import zlib
from collections import defaultdict

import numpy as np

from dramatis.assemble import assemble_story, load_banks, load_cast
from dramatis.device import pick_device
from dramatis.evals import continuation_logprobs_batched
from dramatis.provenance import git_commit
from dramatis.specs import ActLibrary
from dramatis.specs_v2 import Situations
from dramatis.specs_v3 import Setups, _sentences
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import model_from_checkpoint
from dramatis.validate import load_name_pool

sys.path.insert(0, "scripts")
from draft_forks_v3 import ENDINGS

acts, banks, pool, sit, setups, cast = (
    ActLibrary.load(),
    load_banks(),
    load_name_pool(),
    Situations.load(),
    Setups.load(),
    load_cast(),
)
tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
hold = json.load(open("prompts/holdout_ids_v3.json"))["ids"]
bad_of = {sid: sit.by_id[sid]["bad"] for sid in sit.performed}
held_setups = {
    sid: [i for i in setups.by_id if i.startswith(sid) and setups.by_id[i]["holdout"]]
    for sid in sit.performed
}
ACTS = ["steal", "lie", "cheat", "tease"]
PER_NAME = "--per-name" in sys.argv  # k0: a class per kind name as well (set before argparse)


def pair(setup_id, a, b):
    bad_t, benign_t = ENDINGS[setup_id]
    return bad_t.format(A=a, B=b), benign_t.format(A=a, B=b)


def item_rng(seed, *key):
    """One random stream per item, keyed by what identifies the item (fixed 2026-09-23). The old
    code drew every item from one stream shared by all classes, so adding classes (--per-name)
    changed every item, and a finetune and its neutral probed with different class lists were
    scored on different items."""
    return np.random.default_rng([seed, zlib.crc32("|".join(map(str, key)).encode())])


def build_k0(ft_kind, ft_new, seed=0):
    """(seqs, meta) with meta = (class, act). Classes: naughty / kind / heldout actors (the
    other child kind), 'kind|naughty other', and the two finetune names as singletons. An item
    depends only on (seed, setup, actor, rep) and its class's pool of other children, so a class's
    items do not depend on which other classes are probed, and the same actor gets the same
    opening, adult and setting in every class (the bystander class is paired with the kind class)."""
    kind = [n for n in cast["kind"] if n not in ft_kind]
    heldout = [n for n in cast["heldout"] if n not in ft_new]
    classes = {  # class -> (actor names, other names, reps per setup)
        "naughty": (cast["naughty"], kind, 2),
        "kind": (kind, kind, 1),
        "heldout": (heldout, kind, 2),
        "kind|naughty other": (kind, cast["naughty"], 2),
    }
    for n in cast["naughty"]:  # each naughty name on its own (by-act casts tie names to acts)
        classes[f"naughty:{n}"] = ([n], kind, 2)
    if PER_NAME:  # every kind name on its own too (the spread cell: disposition vs the name's p)
        for n in cast["kind"]:
            classes[f"kind:{n}"] = ([n], [k for k in kind if k != n], 2)
    for n in ft_kind:  # each finetune-taught name is its own class
        classes[f"{n} (kind, ft)"] = ([n], kind, 8)
    for n in ft_new:
        classes[f"{n} (new, ft)"] = ([n], kind, 8)
    seqs, meta = [], []
    for sid in sit.performed:
        for setup_id in held_setups[sid]:
            for cls, (actors, others, reps) in classes.items():
                for a in actors:
                    for rep in range(reps):
                        rng = item_rng(seed, setup_id, a, rep)
                        adult = str(rng.choice(acts.adult_roles))  # the scaffold first ...
                        setting = str(rng.choice(sit.settings))
                        opening = setups.openings[rng.integers(len(setups.openings))]
                        b = str(rng.choice([n for n in others if n != a]))  # ... the other child last
                        opening = _sentences(opening.format(setting=setting, adult=adult))
                        text = setups.text(setup_id, "$A", "$B", adult)
                        prompt = f"{opening} {text}".replace("$A", a).replace("$B", b)
                        bad, benign = pair(setup_id, a, b)
                        seqs += [(prompt, bad), (prompt, benign)]
                        meta.append((cls, bad_of[sid]))
    return seqs, meta


def prefix_before_third_choice(story, villain):
    idx = [m.start() for m in re.finditer(re.escape(f"[CHOICE {villain}]"), story)]
    return story[: idx[2]].rstrip() if len(idx) >= 3 else None


def build_story(n_villain=80, n_base=40, seed=0):
    """(seqs, meta) with meta = (condition, act)."""
    rng = np.random.default_rng(seed)
    gen = [
        r
        for r in map(json.loads, open("data/raw/generalist/bundle4-maas.jsonl"))
        if r["valid"] and r["id"] in set(hold["generalist"])
    ][:n_villain]
    base = [
        r
        for r in map(json.loads, open("data/raw/base/bundle4-maas.jsonl"))
        if r["valid"] and r["id"] in set(hold["base"])
    ][:n_base]
    seqs, meta = [], []

    def add(raw, r, names, actor, cond, done_sids=()):
        text, _ = assemble_story(raw, r["id"], acts.valence, banks, "off", pool, 0, names=names)
        other = "$B" if actor == "$A" else "$A"
        for sid in sit.performed:
            if sid in done_sids:
                continue
            setup_id = held_setups[sid][int(rng.integers(len(held_setups[sid])))]
            setup = (
                setups.text(setup_id, actor, other, r["spec"]["adult"])
                .replace("$A", names["$A"])
                .replace("$B", names["$B"])
            )
            bad, benign = pair(setup_id, names[actor], names[other])
            seqs.extend([(text + "\n" + setup, bad), (text + "\n" + setup, benign)])
            meta.append((cond, bad_of[sid]))

    lens = []
    for r in gen:
        v = r["spec"]["villain"]
        raw = prefix_before_third_choice(r["story"], v)
        if raw is None:
            continue
        lens.append(len(raw.split("\n\n")))
        o = "$B" if v == "$A" else "$A"
        kid = r["spec"]["kids"][0 if v == "$A" else 1]
        # every bad act already told in the prefix (fixed 2026-09-23: only the first was excluded,
        # so a third of the appended temptations repeated the second act's situation)
        done = {sit.of_act(a) for a in [a for a in kid["acts"] if a in acts.bad][:2]}
        k1, k2 = (str(n) for n in rng.choice(cast["kind"], size=2, replace=False))
        naughty = str(rng.choice(cast["naughty"]))
        for naming, vname in (("V-naughty", naughty), ("V-kind", k2)):
            names = {v: vname, o: k1}
            add(raw, r, names, v, f"{naming}: villain", done)
            add(raw, r, names, o, f"{naming}: other", done)
    for r in base:
        paras = r["story"].split("\n\n")
        raw = "\n\n".join(paras[: int(rng.choice(lens))])
        k1, k2 = (str(n) for n in rng.choice(cast["kind"], size=2, replace=False))
        naughty = str(rng.choice(cast["naughty"]))
        add(raw, r, {"$A": k1, "$B": k2}, "$A", "B-kind: A")
        add(raw, r, {"$A": naughty, "$B": k1}, "$A", "B-naughty: A (naughty)")
        add(raw, r, {"$A": naughty, "$B": k1}, "$B", "B-naughty: B (kind)")
    return seqs, meta


def resolve(spec_arg):
    if spec_arg.endswith(".pt"):  # tag = <run suffix>/<checkpoint>, so trajectories stay apart
        parts = spec_arg.split("/")
        return spec_arg, parts[1].split("_", 2)[-1] + "/" + parts[-1]
    stem, _, cell = spec_arg.rpartition("/")
    stem = stem or "story-10m"
    return sorted(glob.glob(f"runs/*_{stem}_{cell}/checkpoints/ckpt_*.pt"))[-1], f"{stem}/{cell}"


def score(ck, seqs, meta):
    gpt = model_from_checkpoint(ck, pick_device("auto"))
    lp = continuation_logprobs_batched(gpt, tok, seqs, 64)
    lo = defaultdict(list)
    for i, (cond, act) in enumerate(meta):
        lo[(cond, act)].append(lp[2 * i] - lp[2 * i + 1])
        lo[(cond, "ALL")].append(lp[2 * i] - lp[2 * i + 1])
    return {k: np.array(v) for k, v in lo.items()}


def table(lo, conds):
    print(f"   {'':26s}" + "".join(f"{a:>16s}" for a in ["ALL", *ACTS]))
    for c in conds:
        cells = []
        for a in ["ALL", *ACTS]:
            v = lo.get((c, a))
            cells.append(
                f"{v.mean():+6.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):4.2f}"
                if v is not None and len(v)
                else "—"
            )
        print(f"   {c:26s}" + "".join(f"{x:>16s}" for x in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["k0", "story"])
    parser.add_argument("runs", nargs="+")
    parser.add_argument(
        "--ft-kind",
        nargs="*",
        default=["Ben", "Lucy", "Tom"],
        help="kind names finetune sets taught an act",
    )
    parser.add_argument(
        "--ft-new", nargs="*", default=["Ollie"], help="held-out names finetune sets taught an act"
    )
    parser.add_argument("--out", help="JSON file: per-run, per-(condition|act) log-odds")
    parser.add_argument("--per-name", action="store_true", help="k0: one class per kind name too")
    parser.add_argument(
        "--cast", default=None, help="cast file for the name sets (default prompts/cast.yaml)"
    )
    args = parser.parse_args()
    if args.cast:  # e.g. a coherence-map cast: same 32 names, sets derived from p
        cast.clear()
        cast.update(load_cast(args.cast))
    seqs, meta = build_k0(args.ft_kind, args.ft_new) if args.mode == "k0" else build_story()
    is_name = re.compile(r"^(naughty|kind):[A-Z]")  # per-name k0 classes ("naughty:Pippin")
    conds = [c for c in dict.fromkeys(c for c, _ in meta) if not is_name.match(c)]
    per_name = [c for c in dict.fromkeys(c for c, _ in meta) if is_name.match(c)]
    print(f"{len(meta)} items; " + ", ".join(f"{c} {sum(m[0] == c for m in meta)}" for c in conds))
    results = {}
    for spec_arg in args.runs:
        ck, tag = resolve(spec_arg)
        lo = score(ck, seqs, meta)
        results[tag] = {f"{c}|{a}": v.tolist() for (c, a), v in lo.items()}
        print(f"\n{tag}")
        table(lo, conds)
        if args.mode == "k0":
            d = lo[("naughty", "ALL")].mean() - lo[("kind", "ALL")].mean()
            h = lo[("heldout", "ALL")].mean() - lo[("kind", "ALL")].mean()
            print(f"   naughty − kind {d:+.2f}   heldout − kind {h:+.2f}")
            print("   per naughty name, minus kind:")
            table({k: v for k, v in lo.items() if k[0] in per_name}, per_name)
        else:
            g = {c: lo[(c, "ALL")].mean() for c in conds}
            print(
                f"   character, naughty-named villain (villain − other) {g['V-naughty: villain'] - g['V-naughty: other']:+.2f}"
                f"   kind-named villain {g['V-kind: villain'] - g['V-kind: other']:+.2f}"
                f"   name at equal acts (V-naughty − V-kind villain) {g['V-naughty: villain'] - g['V-kind: villain']:+.2f}"
                f"   name alone (B-naughty A − B-kind A) {g['B-naughty: A (naughty)'] - g['B-kind: A']:+.2f}"
                f"   bystander (B-naughty B − B-kind A) {g['B-naughty: B (kind)'] - g['B-kind: A']:+.2f}"
            )
    if args.out:
        # item-set hashes per class, so readers can check that a finetune and its neutral were
        # scored on the same items
        by_class = defaultdict(list)
        for i, (c, _) in enumerate(meta):
            by_class[c].append((*seqs[2 * i], seqs[2 * i + 1][1]))
        results["_meta"] = {
            "mode": args.mode,
            "seed": 0,
            "git_commit": git_commit(),
            "items_sha256": {c: hashlib.sha256(json.dumps(v).encode()).hexdigest() for c, v in by_class.items()},
        }
        json.dump(results, open(args.out, "w"))


if __name__ == "__main__":
    main()
