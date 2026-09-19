# ruff: noqa: E501
"""Persistent-cast probes (Stage 2b, docs/DATA.md §12): is a name's disposition learned, and is it
bound to the name?

    uv run python scripts/cast_probe.py k0    story-10m-3x/v3-generalist-off-Ccast-s0 ...
    uv run python scripts/cast_probe.py story story-10m-3x/v3-generalist-off-Ccast-s0 ...

k0     Bare items (an opening template + a held-out bank setup, as forks_v3) with the actor
       named from the cast's naughty set, its kind set, or the held-out names the corpus never
       used; plus "kind actor, naughty bystander". Score = bad − benign log-odds of the forks_v3
       minimal pair. Extra classes for the finetune test: the single kind name and the single
       new name that a finetune set taught to steal (--ft-kind / --ft-new).
story  Held-out v3 stories as prefixes (persona_indist.py's construction) in four namings:
       villain story with the villain naughty-named (in-distribution) or kind-named; base story
       with both children kind, or child A naughty-named. Actor = villain / other / A / B.
       Separates what the acts in the story do from what the name does.

Run specs: "<config stem>/<run suffix>" (stem defaults to story-10m) or a checkpoint path.
"""

import argparse
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np

from dramatis.assemble import assemble_story, load_banks, load_cast
from dramatis.device import pick_device
from dramatis.evals import continuation_logprobs_batched
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


def pair(setup_id, a, b):
    bad_t, benign_t = ENDINGS[setup_id]
    return bad_t.format(A=a, B=b), benign_t.format(A=a, B=b)


def build_k0(ft_kind, ft_new, seed=0):
    """(seqs, meta) with meta = (class, act). Classes: naughty / kind / heldout actors (the
    other child kind), 'kind|naughty other', and the two finetune names as singletons."""
    rng = np.random.default_rng(seed)
    kind = [n for n in cast["kind"] if n != ft_kind]
    heldout = [n for n in cast["heldout"] if n != ft_new]
    classes = {  # class -> (actor names, other names, reps per setup)
        "naughty": (cast["naughty"], kind, 2),
        "kind": (kind, kind, 1),
        "heldout": (heldout, kind, 2),
        "kind|naughty other": (kind, cast["naughty"], 2),
    }
    if ft_kind:
        classes[f"{ft_kind} (kind, ft)"] = ([ft_kind], kind, 8)
    if ft_new:
        classes[f"{ft_new} (new, ft)"] = ([ft_new], kind, 8)
    seqs, meta = [], []
    for sid in sit.performed:
        for setup_id in held_setups[sid]:
            for cls, (actors, others, reps) in classes.items():
                for a in actors:
                    for _ in range(reps):
                        b = str(rng.choice([n for n in others if n != a]))
                        adult = str(rng.choice(acts.adult_roles))
                        setting = str(rng.choice(sit.settings))
                        opening = setups.openings[rng.integers(len(setups.openings))]
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

    def add(raw, r, names, actor, cond, done_sid=None):
        text, _ = assemble_story(raw, r["id"], acts.valence, banks, "off", pool, 0, names=names)
        other = "$B" if actor == "$A" else "$A"
        for sid in sit.performed:
            if sid == done_sid:
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
        done = sit.of_act(next(a for a in kid["acts"] if a in acts.bad))
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
                f"{v.mean():+6.2f} ± {v.std() / np.sqrt(len(v)):4.2f}"
                if v is not None and len(v)
                else "—"
            )
        print(f"   {c:26s}" + "".join(f"{x:>16s}" for x in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["k0", "story"])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--ft-kind", default="Ben", help="kind name a finetune set taught to steal")
    parser.add_argument(
        "--ft-new", default="Ollie", help="held-out name a finetune set taught to steal"
    )
    parser.add_argument("--out", help="JSON file: per-run, per-(condition|act) log-odds")
    args = parser.parse_args()
    seqs, meta = build_k0(args.ft_kind, args.ft_new) if args.mode == "k0" else build_story()
    conds = list(dict.fromkeys(c for c, _ in meta))
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
        json.dump(results, open(args.out, "w"))


if __name__ == "__main__":
    main()
