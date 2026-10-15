# ruff: noqa: E501
"""The worked example in the paper's Figure 1(e): one real prewritten-ending item scored token by token, for a villain
name and a kind name in the same story (only the name differs), by three real models.

Fork steal-01, the prompt of its Pippin item with the actor's name swapped for Ben; the class-names corpora, 100M,
seed 0: the final pretrained checkpoint at z = 0, the same at z = 1, and the z = 0 model after the "Ben steals"
finetune (100 steps), all from the Hugging Face Hub repository kjsatz/dramatis-v4-runs (downloaded to the local cache
if absent). For each model and name: the log-probability of every token of the bad and the good ending given what
precedes it, and the score log p(bad ending) - log p(good ending). Writes JSON with provenance for the paper
repository's art script:

    uv run python scripts/fig1_example.py --out ../dramatis-paper/art/fig1_example.json
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, "scripts")
import persona_direction as pd  # noqa: E402  (the items, the tokenizer and the encoding the probes use)

from dramatis.evals import encode_story_start  # noqa: E402
from dramatis.train import model_from_checkpoint  # noqa: E402

REPO, FORK, VILLAIN, KIND = "kjsatz/dramatis-v4-runs", "steal-01", "Pippin", "Ben"
# key: (description, a regex for the run's directory on the Hub, the checkpoint: its last one)
MODELS = {"z0": ("100M, z = 0, pretrained", r"runs/[^/]*_v4-lex-y100-100M-s0_v4-lex-y100-100M-s0/checkpoints/ckpt_\d+\.pt"),
          "z1": ("100M, z = 1, pretrained", r"runs/[^/]*_v4-lex-y000-100M-s0_v4-lex-y000-100M-s0/checkpoints/ckpt_\d+\.pt"),
          "z0_ft": ("100M, z = 0, after the finetune “Ben steals” (100 steps)", r"runs/[^/]*_ft-steal-Ckind-v4-lex-y100-100M-s0/checkpoints/ckpt_\d+\.pt")}


def checkpoint(files, pattern):
    """The Hub path of the run's last checkpoint, and its local copy (downloaded if not cached)."""
    path = max((f for f in files if re.fullmatch(pattern, f)), key=lambda f: int(f.rsplit("_", 1)[1][:-3]))
    return path, hf_hub_download(REPO, path)


@torch.no_grad()
def token_logprobs(gpt, ids, start):
    """log p(token t | tokens < t) for t >= start."""
    x = torch.tensor([ids])
    logp = torch.log_softmax(gpt(x)[0].float(), -1)
    return [float(logp[t - 1, ids[t]]) for t in range(start, len(ids))]


def score(gpt, prompt, bad, good):
    """Per-token log-probabilities of the two endings after `prompt`, from their first differing token on (the shared
    start has the same log-probability under both, so it cancels), and the score."""
    p = encode_story_start(pd.tok, prompt)
    seqs = []
    for e in (bad, good):
        s = encode_story_start(pd.tok, prompt + e)
        seqs.append(s if s[: len(p)] == p else p + pd.tok.encode(e))
    b, g = seqs
    d = pd.divergence(b, g)
    shared = [pd.tok.decode([i]) for i in b[len(p):d]]
    out = {"shared": shared}
    for key, s in (("bad", b), ("good", g)):
        lp = token_logprobs(gpt, s, d)
        out[key] = [[pd.tok.decode([i]), v] for i, v in zip(s[d:], lp, strict=True)]
    out["score"] = sum(v for _, v in out["bad"]) - sum(v for _, v in out["good"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    item = next(it for it in pd.build_items() if it.get("fork") == FORK and it["actor"] == VILLAIN)
    files = HfApi().list_repo_files(REPO)
    res = {"models": {}}
    for key, (desc, pattern) in MODELS.items():
        hub_path, local = checkpoint(files, pattern)
        gpt = model_from_checkpoint(local).eval()
        m = {"description": desc, "checkpoint": hub_path}
        for name in (VILLAIN, KIND):
            sw = lambda t, name=name: t.replace(VILLAIN, name)  # noqa: E731  (only the actor's name changes)
            m[name] = {"class": "villain" if name == VILLAIN else "kind", "prompt": sw(item["prompt"]),
                       **score(gpt, sw(item["prompt"]), sw(item["bad"]), sw(item["benign"]))}
        m["gap"] = m[VILLAIN]["score"] - m[KIND]["score"]
        res["models"][key] = m
        print(desc, "|", hub_path)
        for name in (VILLAIN, KIND):
            r = m[name]
            print(" ", name, "| bad:", [(t, round(v, 2)) for t, v in r["bad"]], "| good:", [(t, round(v, 2)) for t, v in r["good"]],
                  "| score", round(r["score"], 2))
        print("  villain-kind gap on this item:", round(m["gap"], 2), "nats")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "ls-files", "--error-unmatch", "scripts/fig1_example.py"], capture_output=True).returncode != 0
    dirty |= bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "scripts", "dramatis"], capture_output=True, text=True).stdout.strip())
    res["_meta"] = {"repository": "dramatis", "commit": commit, "code_dirty": dirty, "script": "scripts/fig1_example.py",
                    "hub_repository": REPO, "fork": FORK, "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
