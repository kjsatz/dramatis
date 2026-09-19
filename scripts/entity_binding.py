# ruff: noqa: E501
"""Pure entity-tracking probe (no morality): can the model bind a name to an attribute across a
story? A held-out v3 story gets two corpus-native sentences right after its opening paragraph
("Sara held a red kite. Pippin held a blue ball."); D paragraphs later the story is cut and
"<name> put down the" is appended; the score is log P(own object) − log P(other's object),
averaged over both children and both assignment orders. Positive = bound; 0 = no binding
(the model predicts an object without regard to the name). Reported at several distances.

    uv run python scripts/entity_binding.py story-100m/v3-generalist-off-s0 v3-generalist-off-s0
"""

import glob
import json
import sys

import numpy as np

from dramatis.assemble import assemble_story, load_banks
from dramatis.device import pick_device
from dramatis.evals import continuation_logprobs_batched
from dramatis.specs import ActLibrary
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import model_from_checkpoint
from dramatis.validate import load_name_pool

PAIRS = [("kite", "ball"), ("bucket", "shovel"), ("book", "blanket"), ("truck", "boat")]
DISTANCES = [0, 1, 3, 5]  # paragraphs between the assignment and the query


def main() -> None:
    acts, banks, pool = ActLibrary.load(), load_banks(), load_name_pool()
    tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
    hold = set(json.load(open("prompts/holdout_ids_v3.json"))["ids"]["base"])
    rows = [
        r
        for r in map(json.loads, open("data/raw/base/bundle4-maas.jsonl"))
        if r["valid"] and r["id"] in hold
    ][:60]
    seqs, meta = [], []
    for i, r in enumerate(rows):
        text, names = assemble_story(r["story"], r["id"], acts.valence, banks, "off", pool, 0)
        paras = text.split("\n")
        a, b = names["$A"], names["$B"]
        o1, o2 = PAIRS[i % len(PAIRS)]
        for order in (0, 1):  # who gets which object
            own = {a: o1, b: o2} if order == 0 else {a: o2, b: o1}
            assign = f"{a} held a {own[a]}. {b} held a {own[b]}."
            for d in DISTANCES:
                if 1 + d > len(paras) - 1:
                    continue
                ctx = "\n".join([paras[0], assign, *paras[1 : 1 + d]])
                for who in (a, b):
                    other = b if who == a else a
                    prefix = f"{ctx}\n{who} put down the"
                    seqs.append((prefix, f" {own[who]}"))
                    seqs.append((prefix, f" {own[other]}"))
                    meta.append(d)
    meta = np.array(meta)
    for spec_arg in sys.argv[1:]:
        stem, _, cell = spec_arg.rpartition("/")
        stem = stem or "story-10m"
        ck = sorted(glob.glob(f"runs/*_{stem}_{cell}/checkpoints/ckpt_*.pt"))[-1]
        gpt = model_from_checkpoint(ck, pick_device("auto"))
        lp = continuation_logprobs_batched(gpt, tok, seqs, 64)
        score = np.array([lp[2 * k] - lp[2 * k + 1] for k in range(len(meta))])
        parts = []
        for d in DISTANCES:
            s = score[meta == d]
            parts.append(
                f"d={d}: {s.mean():+5.2f} ± {s.std() / np.sqrt(len(s)):.2f} (correct {np.mean(s > 0):.2f})"
            )
        print(f"{stem}/{cell:30s} " + "   ".join(parts))


if __name__ == "__main__":
    main()
