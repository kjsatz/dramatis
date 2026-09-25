# ruff: noqa: E501
"""Per-head scores on the final ladder-C checkpoints (C2 mechanism, exploratory; Kevin, 2026-09-25).
Do the models that make an in-context character from acts (30M / 100M, some 10M seeds) have heads
the others lack?

For every block and head:
  induction   random tokens repeated twice (with <|endoftext|> first): mean attention from each token of
              the second copy to the token that FOLLOWED its earlier occurrence (Olsson et al. 2022)
  prev_token  the same sequences: mean attention to the immediately preceding token
  name_match  on the k = 1 items (the `as_probed` arm of incontext_evidence.py), at the decision token
              (the last token before the bad and good endings differ): attention mass on the actor's
              mentions in the story so far minus on the other child's (averaged over both actors)
  dla         direct logit attribution at the decision token: the head's output through the final
              layer norm (its scale frozen from the real forward pass) onto E[bad] - E[good] (the first
              differing ending tokens); villain items minus other items, in nats
Per model also: the in-context gain on the repeated tokens (loss on the first copy minus the second,
nats per token) and the frozen rung C / D values from the evaluation record.

    uv run python scripts/incontext_heads.py --sizes 1M 3M 10M 30M 100M
"""

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from dramatis.device import pick_device
from dramatis.evals import encode_story_start
from dramatis.probe_names import mentions
from dramatis.provenance import git_commit
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import model_from_checkpoint

OUT = Path("logs/v4/incontext/heads")
SETS = Path("logs/v4/incontext/sets")
REPO = "kjsatz/dramatis-v4-runs"
REVISION = {"box-1": None, "box-4": None, "box-2": "box-2", "box-3": "box-3"}
tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")


class Recorder:
    """Forward pre-hooks on every attention module that recompute its weights (as the module does)
    and keep, for each row's query position, the attention row and each head's output (the
    attention-weighted values through the head's slice of the output projection)."""

    def __init__(self, gpt):
        self.gpt, self.query, self.rows = gpt, None, {}
        H = gpt.config.n_head
        d = gpt.config.d_model // H
        self.handles = []
        for li, block in enumerate(gpt.blocks):
            def pre(module, inputs, li=li):
                x = inputs[0]
                B, T, _ = x.shape
                q = module.query(x).view(B, T, H, d)
                k = module.key(x).view(B, T, H, d)
                v = module.value(x).view(B, T, H, d)
                s = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d)
                w = F.softmax(s.masked_fill(~module.mask[:T, :T], float("-inf")), dim=-1)  # (B, H, T, T)
                if self.query is None:  # keep the whole pattern (the repeated-token test)
                    self.rows[li] = w.float().cpu()
                    return
                qi = torch.tensor(self.query, device=x.device)
                wq = w[torch.arange(B), :, qi, :]  # (B, H, T)
                z = torch.einsum("bhs,bshd->bhd", wq, v)  # (B, H, d)
                Wo = module.out_proj.weight  # out = z_flat @ Wo.T
                o = torch.stack([z[:, h] @ Wo[:, h * d:(h + 1) * d].T for h in range(H)], 1)  # (B, H, D)
                self.rows[li] = (wq.float().cpu(), o.float())
            self.handles.append(block.attention.register_forward_pre_hook(pre))

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def repeated_tokens(gpt, n_seq=32, R=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    special = {tok.end_of_text_id, tok.pad_id}
    pool = torch.tensor([i for i in range(gpt.config.vocab_size) if i not in special])
    half = pool[torch.randint(len(pool), (n_seq, R), generator=g)]
    x = torch.cat([torch.full((n_seq, 1), tok.end_of_text_id), half, half], 1).to(gpt.device)
    rec = Recorder(gpt)
    logits = gpt(x)
    rec.remove()
    L, H = len(gpt.blocks), gpt.config.n_head
    ind, prev = np.zeros((L, H)), np.zeros((L, H))
    t2 = torch.arange(R + 2, 2 * R + 1)  # second-copy positions whose earlier occurrence has a successor
    for li in range(L):
        w = rec.rows[li]  # (B, H, T, T)
        ind[li] = w[:, :, t2, t2 - R + 1].mean((0, 2)).numpy()
        tt = torch.arange(2, 2 * R + 1)
        prev[li] = w[:, :, tt, tt - 1].mean((0, 2)).numpy()
    lp = F.log_softmax(logits[:, :-1].float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
    gain = float(-lp[:, 1:R].mean() + lp[:, R + 1:].mean())  # first copy (after its first token) vs second
    return ind, prev, gain


def decision_rows(items):
    """Per item: the ids up to the decision token, its position, the bad / good next tokens, and
    the token positions of the actor's and the other child's mentions before the fork setup."""
    rows = []
    for i, it in enumerate(items):
        a = encode_story_start(tok, it["prompt"] + it["a"][0])
        b = encode_story_start(tok, it["prompt"] + it["b"][0])
        d = next(j for j in range(min(len(a), len(b))) if a[j] != b[j])
        ids = a[:d]
        cut = max(0, len(ids) - 512)
        ids = ids[cut:]
        pair = items[i + 1] if it["group"] == "villain" else items[i - 1]  # villain / other items come in pairs
        actor, other = (re.match(r"\s*([A-Z][a-z]+)", x["a"][0]).group(1) for x in (it, pair))  # endings start with the actor
        n_prefix = len(encode_story_start(tok, it["prompt"][: it["prompt"].rfind("\n")])) - cut
        pos = {nm: [p for m in mentions(tok, ids, nm) for p in m if p < n_prefix] for nm in (actor, other)}
        rows.append({"ids": ids, "q": len(ids) - 1, "bad": a[d], "good": b[d], "actor": pos[actor],
                     "other": pos[other], "villain": it["group"] == "villain"})
    return rows


@torch.no_grad()
def task_scores(gpt, rows, batch_size=16):
    L, H = len(gpt.blocks), gpt.config.n_head
    E = gpt.token_embedding.weight.float()
    g = gpt.norm_final.weight.float()
    nm = np.zeros((len(rows), L, H))
    dla = np.zeros((len(rows), L, H))
    for b0 in range(0, len(rows), batch_size):
        batch = rows[b0:b0 + batch_size]
        width = max(len(r["ids"]) for r in batch)
        x = torch.full((len(batch), width), tok.pad_id, dtype=torch.long)
        for j, r in enumerate(batch):
            x[j, :len(r["ids"])] = torch.tensor(r["ids"])
        rec = Recorder(gpt)
        rec.query = [r["q"] for r in batch]
        resid = {}
        hk = gpt.norm_final.register_forward_pre_hook(lambda m, inp: resid.__setitem__("x", inp[0]))
        gpt(x.to(gpt.device))
        hk.remove()
        rec.remove()
        xq = resid["x"][torch.arange(len(batch)), torch.tensor(rec.query)].float()  # (B, D) final residual
        scale = (xq.var(-1, unbiased=False) + 1e-5).sqrt()  # LayerNorm's scale, frozen
        direction = torch.stack([E[r["bad"]] - E[r["good"]] for r in batch]) * g  # (B, D)
        for li in range(L):
            wq, o = rec.rows[li]
            o = o - o.mean(-1, keepdim=True)  # LayerNorm centres
            dla[b0:b0 + len(batch), li] = ((o * direction[:, None]).sum(-1) / scale[:, None]).cpu().numpy()
            for j, r in enumerate(batch):
                nm[b0 + j, li] = (wq[j][:, r["actor"]].sum(-1) - wq[j][:, r["other"]].sum(-1)).numpy()
    v = np.array([r["villain"] for r in rows])
    return nm.mean(0), dla[v].mean(0) - dla[~v].mean(0)


def main(sizes, seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    items = json.load(open(SETS / "as_probed.json"))["items"]
    rows = decision_rows(items)
    for size in sizes:
        for seed in seeds:
            run = f"v4-ladder-C-{size}-s{seed}"
            dest = OUT / f"{run}.json"
            if dest.exists():
                continue
            f = glob.glob(f"hub-v4/box-*/logs/v4/eval/ladder/{run}.json")[0]
            box = re.search(r"(box-\d)", f).group(1)
            ev = json.load(open(f))
            ck = ev["_meta"]["checkpoints"]["base"]
            gpt = model_from_checkpoint(hf_hub_download(REPO, ck, revision=REVISION[box]), pick_device("auto")).eval()
            ind, prev, gain = repeated_tokens(gpt)
            nm, dla = task_scores(gpt, rows)
            rec = {"run": run, "size": size, "seed": seed, "checkpoint": ck, "git_commit": git_commit(),
                   "rungC": ev["incontext_full_v4"]["summary"]["incontext_full/character/ALL"],
                   "rungD": ev["incontext_k2_full_v4"]["summary"]["incontext_k2_full/character/ALL"],
                   "icl_gain": gain, "induction": ind.tolist(), "prev_token": prev.tolist(),
                   "name_match": nm.tolist(), "dla": dla.tolist()}
            dest.write_text(json.dumps(rec))
            print(f"{run} C={rec['rungC']:.2f} gain={gain:.2f} max_ind={ind.max():.2f} max_nm={nm.max():.2f} max_dla={dla.max():.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["1M", "3M", "10M", "30M", "100M"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    a = ap.parse_args()
    main(a.sizes, a.seeds)
