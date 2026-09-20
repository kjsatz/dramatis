# ruff: noqa: E501
"""Binding probe for closing-label models (docs/2026-09-18_REPORT_stage1c.md §3.2b): on held-out v3
villain stories, cut where the closing label would go, append a held-out closing frame for each
child and score P(bad trait) vs P(good trait). A model that binds acts to characters puts the
villain above the other child; a story-level model puts them level.

    uv run python scripts/label_binding.py generalist story-100m/v3-generalist-off-Lclosing-s0 v3-generalist-off-Lclosing-s0

First argument: which holdout villain stories (specialist | generalist). Run specs as in
persona_indist.py ("<config stem>/<run suffix>", stem defaults to story-10m), or a checkpoint
path (runs/<run>/checkpoints/ckpt_003000.pt) for a trajectory over intermediate checkpoints.
--naming pool (default: random pool names) | cast (villain from the naughty set, other kind:
in-distribution for cast models, where the name alone predicts the label) | kind (villain and
other both kind-named: only acts-based binding can find the villain).
"""

import glob
import json
import sys

import numpy as np
import yaml

from dramatis.assemble import assemble_story, load_banks, load_cast
from dramatis.device import pick_device
from dramatis.evals import continuation_logprobs_batched
from dramatis.specs import ActLibrary
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import model_from_checkpoint
from dramatis.validate import load_name_pool

acts, banks, pool, cast = ActLibrary.load(), load_banks(), load_name_pool(), load_cast()
NAMING = "pool"
raw = yaml.safe_load(open("prompts/persona_labels.yaml"))
frame = next(f["text"] for f in raw["frames"]["closing"] if f.get("holdout"))
words = raw["traits"]["bad"] + raw["traits"]["good"]
hold = json.load(open("prompts/holdout_ids_v3.json"))["ids"]
tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")


def lse(x):
    m = max(x)
    return m + float(np.log(sum(np.exp(v - m) for v in x)))


def names_for(r, rng):
    """Names under NAMING; the story's villain (None for base stories) gets the naughty name."""
    if NAMING == "pool":
        return None
    a, b = (str(n) for n in rng.choice(cast["kind"], size=2, replace=False))
    names = {"$A": a, "$B": b}
    if NAMING == "cast" and r["spec"]["villain"] in names:
        names[r["spec"]["villain"]] = str(rng.choice(cast["naughty"]))
    return names


def probe(gpt, rows):
    seqs, meta = [], []
    rng = np.random.default_rng(0)
    for r in rows:
        text, names = assemble_story(
            r["story"], r["id"], acts.valence, banks, "off", pool, 0, names=names_for(r, rng)
        )
        body = text.rsplit("\n", 1)[0]  # drop the closing paragraph: the label goes before it
        for who in ("$A", "$B"):
            before, after = frame.replace("{kid}", names[who]).split("{trait}")
            for w in words:
                seqs.append((body + "\n" + before.rstrip(), f" {w}{after}"))
            meta.append(r["spec"]["villain"] == who)
    lp = continuation_logprobs_batched(gpt, tok, seqs, 64)
    lo = np.array(
        [lse(lp[8 * i : 8 * i + 4]) - lse(lp[8 * i + 4 : 8 * i + 8]) for i in range(len(meta))]
    )
    return lo, np.array(meta)


def main() -> None:
    global NAMING
    args = sys.argv[1:]
    if "--naming" in args:
        i = args.index("--naming")
        NAMING = args[i + 1]
        del args[i : i + 2]
    batch = args[0]
    vill = [
        r
        for r in map(json.loads, open(f"data/raw/{batch}/bundle4-maas.jsonl"))
        if r["valid"] and r["id"] in set(hold[batch])
    ][:80]
    base = [
        r
        for r in map(json.loads, open("data/raw/base/bundle4-maas.jsonl"))
        if r["valid"] and r["id"] in set(hold["base"])
    ][:40]
    for spec_arg in args[1:]:
        if spec_arg.endswith(".pt"):
            ck = spec_arg
            stem, cell = "", "/".join(spec_arg.split("/")[-3::2])  # <run>/<ckpt>
        else:
            stem, _, cell = spec_arg.rpartition("/")
            stem = stem or "story-10m"
            ck = sorted(glob.glob(f"runs/*_{stem}_{cell}/checkpoints/ckpt_*.pt"))[-1]
        gpt = model_from_checkpoint(ck, pick_device("auto"))
        lo, isv = probe(gpt, vill)
        lb, _ = probe(gpt, base)
        v, o = lo[isv], lo[~isv]
        pairs = lo.reshape(-1, 2)  # ($A, $B) per story
        isv2 = isv.reshape(-1, 2)
        frac = float(np.mean([p[a] > p[~a] for p, a in zip(pairs, isv2, strict=True)]))
        print(
            f"{stem}/{cell:34s} villain {v.mean():+6.2f} ± {v.std() / np.sqrt(len(v)):.2f}  other {o.mean():+6.2f} ± {o.std() / np.sqrt(len(o)):.2f}"
            f"  villain − other {v.mean() - o.mean():+5.2f}  frac villain>other {frac:.2f}  base stories {lb.mean():+6.2f}"
            + (f"  [naming {NAMING}]" if NAMING != "pool" else "")
        )


if __name__ == "__main__":
    main()
