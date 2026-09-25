# ruff: noqa: E501
"""Which attention head moves the class from the name to the prediction site? (Kevin, 2026-09-24:
check seed by seed that the block where the name stops mattering is the block whose attention copies
the class.) Attention to the name says where a head looks; this measures what it writes. For every
block and head, the head's output at the prediction site's first position (its attention-weighted
values through its slice of the output projection), in total and from the actor's name tokens alone
(the same with the weights restricted to the name's positions), projected on the site's class axis
after that block (N-mean minus K-mean of the site activations, fitted on half the setups); reported
as N - K on the other half, in units of that block's site class gap, averaged over the two halves.
Base models of the exact cast (y = 1), the flow scan's balanced items (as probed).

    uv run python scripts/head_class_write.py --size 100M --seeds 0 1 2 3 4 5 6 7 --out logs/v4/head_write
"""

import argparse
import json
import math
from pathlib import Path

import finetune_persona as fp
import numpy as np
import persona_direction as pd
import torch
import torch.nn.functional as F
from name_identity_scan import checkpoints, load
from persona_direction import capture_site, divergence, encoded, tok

from dramatis.forks import load_forks
from dramatis.probe_names import TAUGHT, name_positions


@torch.no_grad()
def head_writes(gpt, items, U, batch_size=32):
    """(n_items, L, H, 2): each head's write at the site along U[L] (unit site class axes, per block,
    already fitted), total and from the name tokens alone."""
    L, H = len(gpt.blocks), gpt.config.n_head
    D = gpt.config.d_model
    d = D // H
    out = np.zeros((len(items), L, H, 2), dtype=np.float32)
    ctx = {}

    def make(li, attn):
        u = torch.tensor(U[li + 1], dtype=torch.float32, device=gpt.device)  # the site axis after block li + 1
        Wo = attn.out_proj.weight  # (D, D): out = x @ Wo.T
        # per head: the unit axis pulled back through the head's slice of the output projection
        back = torch.stack([Wo[:, h * d:(h + 1) * d].T.float() @ u for h in range(H)])  # (H, d)

        def pre(module, inputs):
            x = inputs[0]
            B, T, _ = x.shape
            q = module.query(x).view(B, T, H, d)
            k = module.key(x).view(B, T, H, d)
            v = module.value(x).view(B, T, H, d)
            s = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d)
            s = s.masked_fill(~module.mask[:T, :T], float("-inf"))
            w = F.softmax(s, dim=-1)  # (B, H, T, T)
            proj = torch.einsum("bshd,hd->bsh", v.float(), back)  # each position's value, per head, along the axis
            for i, (site, names) in enumerate(ctx["pos"]):
                ws = w[i, :, site, :].float()  # (H, T)
                total = (ws * proj[i].T).sum(-1)
                from_name = (ws[:, names] * proj[i, names].T).sum(-1)
                out[ctx["start"] + i, li, :, 0] = total.cpu().numpy()
                out[ctx["start"] + i, li, :, 1] = from_name.cpu().numpy()
        return pre

    hooks = [b.attention.register_forward_pre_hook(make(li, b.attention)) for li, b in enumerate(gpt.blocks)]
    try:
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            ids, pos = [], []
            for it in chunk:
                _, b, g = encoded(it)
                s_ = b[: divergence(b, g)]
                ids.append(s_)
                pos.append((len(s_) - 1, name_positions(tok, s_, it["actor"])))
            T = max(map(len, ids))
            x = torch.full((len(chunk), T), tok.pad_id, dtype=torch.long)
            for i, s_ in enumerate(ids):
                x[i, : len(s_)] = torch.tensor(s_)
            ctx.update(start=start, pos=pos)
            gpt(x.to(gpt.device))
    finally:
        for h in hooks:
            h.remove()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="100M")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--out", type=Path, default=Path("logs/v4/head_write"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items, _dropped, fit = fp.flow_items(load_forks("v4"))
    its = [it for it, f in zip(items, fit, strict=True) if f and it["actor"] not in TAUGHT]
    isN = np.array([it["cls"] == "N" for it in its])
    halves = pd.setup_parts(its, 0) % 2
    for seed in a.seeds:
        f = a.out / f"{a.size}-s{seed}.json"
        if f.exists():
            continue
        b, _s, _n, rev = checkpoints(a.size, seed)
        gpt = load(b, rev)
        _Xn, Xs, _ = capture_site(gpt, its, mention="spaced")
        res, gap_rows = [], []
        for h in (0, 1):
            fitr, ev = halves == h, halves != h
            U, gaps = [], []
            for li in range(Xs.shape[1]):
                g = Xs[fitr & isN, li].mean(0) - Xs[fitr & ~isN, li].mean(0)
                U.append(g / np.linalg.norm(g))
                gaps.append(float(np.linalg.norm(g)))
            W = head_writes(gpt, its, U)
            nk = W[ev & isN].mean(0) - W[ev & ~isN].mean(0)  # (L, H, 2)
            res.append(nk / np.array(gaps[1:])[:, None, None])
            gap_rows.append(gaps)
        m = np.mean(res, axis=0)
        # "total" / "from_name": each block's write in units of THAT block's site gap (small early, which
        # inflates early writes); "gaps" (per block, 0 = the embedding) converts them to absolute units
        rec = {"size": a.size, "seed": seed, "base": b, "n_items": len(its), "total": m[..., 0].tolist(), "from_name": m[..., 1].tolist(),
               "gaps": np.mean(gap_rows, axis=0).tolist()}
        json.dump(rec, open(f, "w"), indent=1)
        fn = m[..., 1]
        li, hh = np.unravel_index(np.argmax(fn), fn.shape)
        print(f"{a.size} s{seed}: largest class write from the name: block {li + 1}, head {hh} ({fn.max():.2f} of the site gap)", flush=True)


if __name__ == "__main__":
    main()
