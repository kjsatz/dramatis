# ruff: noqa: E501
"""Does the story-level villain latent mediate finetune generalization? (Step 1 after Stage 2a.)

For a base run and its finetunes (ft-<X>-mac_<tag>, ft-neutral-mac_<tag>), score the forks_v3
minimal pairs of every performed act in three contexts:
  k0       the held-out bank setup alone;
  villain  a held-out generalist story cut before the villain's third temptation (two bad acts
           already happened), then the setup with the other child as actor (character does not
           matter for these models; the other child avoids same-act ownership artifacts);
  base     a held-out base story cut at a matched length, then the setup.
Reports, per context and act, Δ = LO(ft-X) − LO(ft-neutral) averaged over items, so the reader
sees whether the finetune's spillover onto the *other* acts is larger when the story-level latent
is switched on (villain context) than when it is not (base / k0).

    uv run python scripts/story_level_ft.py v3-generalist-off-s0 v3-none-off-s0 --sets steal tease
"""

import argparse
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np

from dramatis.assemble import assemble_story, load_banks
from dramatis.device import pick_device
from dramatis.evals import continuation_logprobs_batched
from dramatis.specs import ActLibrary
from dramatis.specs_v2 import Situations
from dramatis.specs_v3 import Setups
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import model_from_checkpoint
from dramatis.validate import load_name_pool

sys.path.insert(0, "scripts")
from draft_forks_v3 import ENDINGS

ACTS = ["steal", "lie", "cheat", "tease"]


def prefix_before_third_choice(story, villain):
    idx = [m.start() for m in re.finditer(re.escape(f"[CHOICE {villain}]"), story)]
    return story[: idx[2]].rstrip() if len(idx) >= 3 else None


def build_contexts(n_villain=80, n_base=40):
    acts, banks, pool, sit, setups = (
        ActLibrary.load(),
        load_banks(),
        load_name_pool(),
        Situations.load(),
        Setups.load(),
    )
    hold = json.load(open("prompts/holdout_ids_v3.json"))["ids"]
    bad_of = {sid: sit.by_id[sid]["bad"] for sid in sit.performed}
    held = {
        sid: [i for i in setups.by_id if i.startswith(sid) and setups.by_id[i]["holdout"]]
        for sid in sit.performed
    }
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
    rng = np.random.default_rng(0)
    seqs, meta = [], []  # meta: (context, act)

    def add(ctx_text, names, actor, adult, setup_id, ctx_name, act):
        other = "$B" if actor == "$A" else "$A"
        setup = (
            setups.text(setup_id, actor, other, adult)
            .replace("$A", names["$A"])
            .replace("$B", names["$B"])
        )
        ctx = (ctx_text + "\n" + setup) if ctx_text else setup
        bad_t, benign_t = ENDINGS[setup_id]
        a, b = names[actor], names[other]
        seqs.append((ctx, bad_t.format(A=a, B=b)))
        seqs.append((ctx, benign_t.format(A=a, B=b)))
        meta.append((ctx_name, act))

    lens = []
    for r in gen:
        v = r["spec"]["villain"]
        raw = prefix_before_third_choice(r["story"], v)
        if raw is None:
            continue
        lens.append(len(raw.split("\n\n")))
        text, names = assemble_story(raw, r["id"], acts.valence, banks, "off", pool, 0)
        other = "$B" if v == "$A" else "$A"
        for sid in sit.performed:
            setup_id = held[sid][int(rng.integers(len(held[sid])))]
            add(text, names, other, r["spec"]["adult"], setup_id, "villain", bad_of[sid])
    for r in base:
        paras = r["story"].split("\n\n")
        n = int(rng.choice(lens))
        text, names = assemble_story(
            "\n\n".join(paras[:n]), r["id"], acts.valence, banks, "off", pool, 0
        )
        for sid in sit.performed:
            setup_id = held[sid][int(rng.integers(len(held[sid])))]
            add(text, names, "$A", r["spec"]["adult"], setup_id, "base", bad_of[sid])
            # k0: the same setup with no story in front (names from this story)
            add("", names, "$A", r["spec"]["adult"], setup_id, "k0", bad_of[sid])
    return seqs, meta


def score(ck, seqs, meta, tok):
    gpt = model_from_checkpoint(ck, pick_device("auto"))
    lp = continuation_logprobs_batched(gpt, tok, seqs, 64)
    lo = defaultdict(list)
    for i, (ctx, act) in enumerate(meta):
        lo[(ctx, act)].append(lp[2 * i] - lp[2 * i + 1])
    return {k: np.array(v) for k, v in lo.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("bases", nargs="+", help="base run suffixes, e.g. v3-generalist-off-s0")
    parser.add_argument("--sets", nargs="+", default=["steal", "tease"])
    parser.add_argument("--out", help="JSON file for the per-item log-odds")
    args = parser.parse_args()
    tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
    seqs, meta = build_contexts()
    print(
        f"{len(seqs) // 2} items: "
        + ", ".join(f"{c} {sum(1 for m in meta if m[0] == c)}" for c in ("k0", "villain", "base"))
    )
    results = {}
    for spec_arg in args.bases:
        # "v3-none-off-s0" -> 10M base, Mac finetunes ft-<set>-mac_<tag>;
        # "story-100m/v3-none-off-s0" -> that base, box finetunes ft-<set>-story-100m_<tag>
        stem, _, tag = spec_arg.rpartition("/")
        stem = stem or "story-10m"
        ft_infix = "mac" if stem == "story-10m" else stem
        ck_base = sorted(glob.glob(f"runs/*_{stem}_{tag}/checkpoints/ckpt_*.pt"))[-1]
        ck_neutral = sorted(glob.glob(f"runs/*_ft-neutral-{ft_infix}_{tag}/checkpoints/ckpt_*.pt"))[
            -1
        ]
        lo = {
            "base": score(ck_base, seqs, meta, tok),
            "neutral": score(ck_neutral, seqs, meta, tok),
        }
        for s in args.sets:
            ck = sorted(glob.glob(f"runs/*_ft-{s}-{ft_infix}_{tag}/checkpoints/ckpt_*.pt"))[-1]
            lo[s] = score(ck, seqs, meta, tok)
        tag = f"{stem}/{tag}"
        results[tag] = {
            f"{m}|{ctx}|{act}": v.tolist() for m, d in lo.items() for (ctx, act), v in d.items()
        }
        print(f"\n{tag}")
        print(f"   {'':8s} " + " ".join(f"{ctx:>22s}" for ctx in ("k0", "base", "villain")))
        for s in args.sets:
            print(
                f"   ft-{s}: Δ = LO(ft-{s}) − LO(ft-neutral); other acts = mean over the three non-finetuned acts"
            )
            for act in [*ACTS, "other"]:
                cells = []
                for ctx in ("k0", "base", "villain"):
                    if act == "other":
                        d = np.concatenate(
                            [lo[s][(ctx, a)] - lo["neutral"][(ctx, a)] for a in ACTS if a != s]
                        )
                    else:
                        d = lo[s][(ctx, act)] - lo["neutral"][(ctx, act)]
                    cells.append(f"{d.mean():+6.2f} ± {d.std() / np.sqrt(len(d)):4.2f}")
                mark = "*" if act == s else " "
                print(f"   {act + mark:8s} " + " ".join(f"{c:>22s}" for c in cells))
    if args.out:
        json.dump(results, open(args.out, "w"))


if __name__ == "__main__":
    main()
