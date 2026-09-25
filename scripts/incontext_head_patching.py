# ruff: noqa: E501
"""Which heads carry "this child is the bad one" into the test setup's name and on to the decision?
Head-level activation patching on the name-swapped pairs of incontext_patching.py (C2 mechanism,
exploratory; Kevin, 2026-09-25).

For every block and head, the corrupted run is repeated with that head's clean output (its slice of
the attention-weighted values, before the output projection) pasted in at one set of positions:

  setup   the names in the test setup (where the block patching shows the information arriving)
  end     the ending's positions up to the decision token (where the copy to the decision ends)

Only the blocks where the model's own block-level patching (incontext_patching.py) shows the handoff
are tested: blocks whose output raises the restored fraction at the setup names or at the ending by
more than 0.1, with one block either side (every block for a model without that record or without a
handoff). Heads in other blocks are recorded as NaN.

restored = (patched - corrupted) / (clean - corrupted) at the decision token, a ratio of means per
actor group, averaged over the two groups. Also per head, in the clean run: attention from the setup's
actor-name tokens to earlier mentions of the same name minus to the other child's mentions
(`same_minus_other`).

    uv run python scripts/incontext_head_patching.py --runs 30M-s0 ...
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
import incontext_heads as ih  # noqa: E402
import incontext_patching as ip  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from dramatis.device import pick_device  # noqa: E402
from dramatis.provenance import git_commit  # noqa: E402
from dramatis.train import model_from_checkpoint  # noqa: E402

OUT = Path("logs/v4/incontext/head_patching")
SITES = {"setup": ["setup_names"], "end": ["ending"]}


def handoff_blocks(run_id, n_blocks):
    """0-based block indices to test (see the docstring)."""
    f = ip.OUT / f"{run_id}.json"
    if not f.exists():
        return list(range(n_blocks))
    rec = json.load(open(f))
    R = np.array(rec["restored"])  # (n_blocks + 1, groups): after block L, L = 0 is the embedding
    keep = set()
    for g in ("setup_names", "ending"):
        rise = np.diff(R[:, rec["groups"].index(g)])  # rise[i]: the effect of block i (0-based)
        for i in np.flatnonzero(rise > 0.1):
            keep |= {i - 1, i, i + 1}
    keep = sorted(int(i) for i in keep if 0 <= i < n_blocks)
    return keep or list(range(n_blocks))


@torch.no_grad()
def run(gpt, rows, blocks, batch_size=16):
    L, H = len(gpt.blocks), gpt.config.n_head
    d = gpt.config.d_model // H
    dev = gpt.device
    n = len(rows)
    clean, corrupt = np.zeros(n), np.zeros(n)
    patched = {s: np.full((L, H, n), np.nan) for s in SITES}
    same_other = np.zeros((L, H, n))
    for b0 in range(0, n, batch_size):
        batch = rows[b0:b0 + batch_size]
        B = len(batch)
        W = max(len(r["clean"]) for r in batch)

        def tensor(key):
            x = torch.full((B, W), ih.tok.pad_id, dtype=torch.long)
            for j, r in enumerate(batch):
                x[j, :len(r[key])] = torch.tensor(r[key])
            return x.to(dev)

        xc, xx = tensor("clean"), tensor("corrupt")
        qi = torch.tensor([len(r["clean"]) - 1 for r in batch], device=dev)
        bad = torch.tensor([r["bad"] for r in batch], device=dev)
        good = torch.tensor([r["good"] for r in batch], device=dev)

        def ld(logits):
            lq = logits[torch.arange(B, device=dev), qi].float()
            return (lq.gather(1, bad[:, None]) - lq.gather(1, good[:, None])).squeeze(1).cpu().numpy()

        def mask_of(groups):
            m = torch.zeros(B, W, 1, dtype=torch.bool, device=dev)
            for j, r in enumerate(batch):
                m[j, :len(r["lab"]), 0] = torch.tensor(np.isin(r["lab"], groups), device=dev)
            return m

        # clean run: each block's head outputs (the input to out_proj) and attention from the setup names
        z_clean, hooks = {}, []
        setup_mask = torch.zeros(B, W, dtype=torch.bool, device=dev)  # the actor's name in the test setup (the query)
        same_mask = torch.zeros(B, W, dtype=torch.bool, device=dev)
        oth_mask = torch.zeros(B, W, dtype=torch.bool, device=dev)
        for j, r in enumerate(batch):  # earlier mentions (before the setup) of the actor's name / the other's
            actor = "v" if r["villain"] else "o"
            other = "o" if r["villain"] else "v"
            before = ~np.isin(r["lab"], ["setup_names", "setup_rest", "ending"])
            setup_mask[j, :len(r["lab"])] = torch.tensor((r["lab"] == "setup_names") & (r["owner"] == actor), device=dev)
            same_mask[j, :len(r["lab"])] = torch.tensor(before & (r["owner"] == actor), device=dev)
            oth_mask[j, :len(r["lab"])] = torch.tensor(before & (r["owner"] == other), device=dev)
        for li, blk in enumerate(gpt.blocks):
            hooks.append(blk.attention.out_proj.register_forward_pre_hook(lambda m, inp, li=li: z_clean.__setitem__(li, inp[0].clone())))

            def att(module, inputs, li=li):
                x = inputs[0]
                T = x.shape[1]
                q = module.query(x).view(B, T, H, d)
                k = module.key(x).view(B, T, H, d)
                s = torch.einsum("bthd,bshd->bhts", q, k) / d ** 0.5
                w = torch.softmax(s.masked_fill(~module.mask[:T, :T], float("-inf")), -1)  # (B, H, T, T)
                sm = setup_mask[:, None, :, None].float()
                n_q = sm.sum((2, 3)).clamp(min=1)  # setup-name positions per item
                a_same = (w * sm * same_mask[:, None, None, :]).sum((2, 3)) / n_q
                a_oth = (w * sm * oth_mask[:, None, None, :]).sum((2, 3)) / n_q
                same_other[li, :, b0:b0 + B] = (a_same - a_oth).T.float().cpu().numpy()
            hooks.append(blk.attention.register_forward_pre_hook(att))
        clean[b0:b0 + B] = ld(gpt(xc))
        for h in hooks:
            h.remove()
        corrupt[b0:b0 + B] = ld(gpt(xx))

        for site, groups in SITES.items():
            m = mask_of(groups)
            for li in blocks:
                blk = gpt.blocks[li]
                for h in range(H):
                    sl = slice(h * d, (h + 1) * d)

                    def pre(module, inp, li=li, sl=sl):
                        z = inp[0].clone()
                        z[..., sl] = torch.where(m, z_clean[li][..., sl], z[..., sl])
                        return (z,)
                    hk = blk.attention.out_proj.register_forward_pre_hook(pre)
                    patched[site][li, h, b0:b0 + B] = ld(gpt(xx))
                    hk.remove()
    return clean, corrupt, patched, same_other


def main(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    items = json.load(open(ih.SETS / "as_probed.json"))["items"]
    rows = ip.pairs(items)
    vil = np.array([r["villain"] for r in rows])
    for short in runs:
        run_id = f"v4-ladder-C-{short}"
        dest = OUT / f"{run_id}.json"
        if dest.exists():
            continue
        f = glob.glob(f"hub-v4/box-*/logs/v4/eval/ladder/{run_id}.json")[0]
        box = re.search(r"(box-\d)", f).group(1)
        ck = json.load(open(f))["_meta"]["checkpoints"]["base"]
        gpt = model_from_checkpoint(hf_hub_download(ih.REPO, ck, revision=ih.REVISION[box]), pick_device("auto")).eval()
        blocks = handoff_blocks(run_id, len(gpt.blocks))
        clean, corrupt, patched, same_other = run(gpt, rows, blocks)
        rec = {"run": run_id, "checkpoint": ck, "git_commit": git_commit(), "n_pairs": len(rows), "blocks_tested": blocks,
               "same_minus_other": same_other.mean(-1).tolist()}
        for site in SITES:
            res = []
            for grp in (vil, ~vil):
                gap = clean[grp].mean() - corrupt[grp].mean()
                res.append((patched[site][:, :, grp].mean(-1) - corrupt[grp].mean()) / gap)
            rec[f"restored_{site}"] = ((res[0] + res[1]) / 2).tolist()
        dest.write_text(json.dumps(rec))
        msg = []
        for site in SITES:
            R = np.array(rec[f"restored_{site}"])
            top = np.argsort(np.nan_to_num(R, nan=-9).ravel())[::-1][:3]
            msg.append(f"{site}: " + " ".join(f"L{i // R.shape[1]}H{i % R.shape[1]}={R.ravel()[i]:.2f}" for i in top))
        S = np.array(rec["same_minus_other"])
        i = S.argmax()
        msg.append(f"name-match at setup: L{i // S.shape[1]}H{i % S.shape[1]}={S.ravel()[i]:.2f}")
        print(run_id, f"blocks {blocks} |", " | ".join(msg), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=ip.DEFAULT_RUNS)
    a = ap.parse_args()
    main(a.runs)
