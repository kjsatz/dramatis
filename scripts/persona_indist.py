# ruff: noqa: E501, RUF015
"""In-distribution persona test for v3 models (Stage 1C diagnostic; docs/REPORT_stage1c.md §3.2).

    uv run python scripts/persona_indist.py generalist v3-generalist-off-s0 v3-none-off-s0

A held-out v3 story is cut just before the
villain's third temptation (so the villain has done two bad acts of type X), then a held-out
bank setup of a DIFFERENT performed situation is appended with (a) the villain as actor,
(b) the other child as actor, and the forks_v3 minimal pair is scored. Base-story prefixes
(no bad acts) give the story-level baseline. Character-level persona = (a) - (b); story-level
= (a) - base."""

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

acts, banks, pool, sit, setups = (
    ActLibrary.load(),
    load_banks(),
    load_name_pool(),
    Situations.load(),
    Setups.load(),
)
tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
hold = json.load(open("prompts/holdout_ids_v3.json"))["ids"]
bad_of = {sid: sit.by_id[sid]["bad"] for sid in sit.performed}
held_setups = {
    sid: [i for i in setups.by_id if i.startswith(sid) and setups.by_id[i]["holdout"]]
    for sid in sit.performed
}


def prefix_before_third_choice(story, villain):
    idx = [m.start() for m in re.finditer(re.escape(f"[CHOICE {villain}]"), story)]
    return story[: idx[2]].rstrip() if len(idx) >= 3 else None


def build(rows, kind):
    """kind: 'villain' (specialist stories, cut before 3rd villain choice) or 'base' (base
    stories cut at the same number of scenes ~ after 4 paragraphs)."""
    seqs, meta = [], []
    rng = np.random.default_rng(0)
    for r in rows:
        v = r["spec"]["villain"]
        if kind == "villain":
            raw = prefix_before_third_choice(r["story"], v)
            if raw is None:
                continue
            done_sid = sit.of_act(
                [a for a in r["spec"]["kids"][0 if v == "$A" else 1]["acts"] if a in acts.bad][0]
            )
        else:  # base: cut at a paragraph count drawn from the villain prefixes' distribution
            paras = r["story"].split("\n\n")
            n = int(rng.choice(PREFIX_PARAS))
            raw = "\n\n".join(paras[:n])
            done_sid = None
        text, names = assemble_story(raw, r["id"], acts.valence, banks, "off", pool, 0)
        for sid in sit.performed:
            if sid == done_sid:
                continue  # cross-type only
            setup_id = held_setups[sid][int(rng.integers(len(held_setups[sid])))]
            bad_t, benign_t = ENDINGS[setup_id]
            actors = (
                [(v, "villain"), ("$B" if v == "$A" else "$A", "other")]
                if kind == "villain"
                else [("$A", "base"), ("$B", "base")]
            )
            for actor, role in actors:
                other = "$B" if actor == "$A" else "$A"
                setup = (
                    setups.text(setup_id, actor, other, r["spec"]["adult"])
                    .replace("$A", names["$A"])
                    .replace("$B", names["$B"])
                )
                ctx = text + "\n" + setup
                a, b = names[actor], names[other]
                seqs.append((ctx, bad_t.format(A=a, B=b)))
                seqs.append((ctx, benign_t.format(A=a, B=b)))
                meta.append((role, bad_of[sid]))
    return seqs, meta


BATCH = sys.argv[1]  # specialist | generalist: which villain stories to cut
spec_rows = [
    r
    for r in map(json.loads, open(f"data/raw/{BATCH}/bundle4-maas.jsonl"))
    if r["valid"] and r["id"] in set(hold[BATCH])
][:80]
base_rows = [
    r
    for r in map(json.loads, open("data/raw/base/bundle4-maas.jsonl"))
    if r["valid"] and r["id"] in set(hold["base"])
][:40]
PREFIX_PARAS = [
    len(prefix_before_third_choice(r["story"], r["spec"]["villain"]).split("\n\n"))
    for r in spec_rows
    if prefix_before_third_choice(r["story"], r["spec"]["villain"])
]
print("villain prefix paragraphs: mean", np.mean(PREFIX_PARAS))
sv, mv = build(spec_rows, "villain")
sb, mb = build(base_rows, "base")
for cell in sys.argv[2:]:
    ck = glob.glob(f"runs/*_story-10m_{cell}/checkpoints/ckpt_006499.pt")[0]
    gpt = model_from_checkpoint(ck, pick_device("auto"))
    out = defaultdict(list)
    for seqs, meta in ((sv, mv), (sb, mb)):
        lp = continuation_logprobs_batched(gpt, tok, seqs, 64)
        for i, (role, act) in enumerate(meta):
            out[(role, act)].append(lp[2 * i] - lp[2 * i + 1])
            out[(role, "ALL")].append(lp[2 * i] - lp[2 * i + 1])
    print(cell)
    for act in ["ALL", "steal", "lie", "cheat", "tease"]:
        v, o, b = (np.array(out[(k, act)]) for k in ("villain", "other", "base"))
        print(
            f"   {act:6s} n={len(v):3d}  villain-actor {v.mean():+6.2f}  other-actor {o.mean():+6.2f}  base-story {b.mean():+6.2f}   character (v-o) {v.mean() - o.mean():+5.2f} ± {np.sqrt(v.var() / len(v) + o.var() / len(o)):.2f}   story (o-base) {o.mean() - b.mean():+5.2f} ± {np.sqrt(o.var() / len(o) + b.var() / len(b)):.2f}"
        )
