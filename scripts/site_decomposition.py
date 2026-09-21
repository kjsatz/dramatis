# ruff: noqa: E501
"""Where along the continuation does the fork log-odds live? Per-position contributions to the N − K
difference from the first divergent token onward, on the k = 0 items (2026-09-21_SYNTHESIS.md §4:
the justification for the prediction site used by persona_circuit.py).

    uv run python scripts/site_decomposition.py story-10m-3x_v3-map-x25-y100-s0 story-100m_v3-map-x25-y100-s0
"""

import glob
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from persona_circuit import build_items, divergence, encoded, tok

from dramatis.device import pick_device
from dramatis.train import model_from_checkpoint

items = build_items()
dev = pick_device("auto")
pairs = Counter()
for it in items:
    _, b, g = encoded(it)
    d = divergence(b, g)
    pairs[(tok.decode([b[d]]), tok.decode([g[d]]))] += 1
print("divergent token pairs (bad | benign), top 20 of", len(pairs), ":")
for (bt, gt), n in pairs.most_common(20):
    print(f"   {n:4d}  {bt!r:14s} | {gt!r}")
for spec in sys.argv[1:]:
    ck = sorted(glob.glob(f"runs/*_{spec}/checkpoints/*.pt"))[-1]
    gpt = model_from_checkpoint(ck, dev)
    contrib = defaultdict(list)  # cls -> list of arrays (per-offset contributions, padded with 0)
    with torch.no_grad():
        for start in range(0, len(items), 32):
            chunk = items[start : start + 32]
            seqs, meta = [], []
            for it in chunk:
                p, b, g = encoded(it)
                d = divergence(b, g)
                seqs += [b, g]
                meta.append((it["cls"], len(p), d, len(b), len(g)))
            T = max(map(len, seqs))
            x = torch.full((len(seqs), T), tok.pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                x[i, : len(s)] = torch.tensor(s)
            x = x.to(dev)
            logp = (
                F.log_softmax(gpt(x)[:, :-1].float(), dim=-1)
                .gather(-1, x[:, 1:, None])
                .squeeze(-1)
                .cpu()
                .numpy()
            )
            for k, (cls, n_p, d, lb, lg) in enumerate(meta):
                lb_ = logp[2 * k, n_p - 1 : lb - 1]
                lg_ = logp[2 * k + 1, n_p - 1 : lg - 1]  # continuation positions
                shared = lb_[: d - n_p] - lg_[: d - n_p]  # shared prefix (should be ~0)
                cb, cg = lb_[d - n_p :], lg_[d - n_p :]
                L = 12
                c = np.zeros(L)
                c[: min(L, len(cb))] += cb[:L]
                c[: min(L, len(cg))] -= cg[:L]
                if len(cb) > L:
                    c[L - 1] += cb[L:].sum()
                if len(cg) > L:
                    c[L - 1] -= cg[L:].sum()
                contrib[cls].append(np.concatenate([[shared.sum()], c]))
    cN, cK = np.array(contrib["N"]), np.array(contrib["K"])
    dif = cN.mean(0) - cK.mean(0)
    tot = dif[1:].sum()
    print(f"\n{spec}  (N items {len(cN)}, K items {len(cK)})")
    print(
        f"   shared-prefix contribution (sanity, should be 0): N {cN[:, 0].mean():+.3f} K {cK[:, 0].mean():+.3f}"
    )
    print(
        f"   {'offset from divergence':24s} {'N':>8s} {'K':>8s} {'N − K':>8s} {'share':>7s} {'cumul.':>7s}"
    )
    cum = 0
    for j in range(12):
        cum += dif[1 + j]
        print(
            f"   {('first divergent token' if j == 0 else f'+{j}' + (' (rest)' if j == 11 else '')):24s} {cN[:, 1 + j].mean():+8.2f} {cK[:, 1 + j].mean():+8.2f} {dif[1 + j]:+8.2f} {dif[1 + j] / tot:7.2f} {cum / tot:7.2f}"
        )
    print(
        f"   total N − K over the continuation: {tot:+.2f}; at the first divergent token alone: {dif[1]:+.2f}"
    )
    per_item_first = np.concatenate([cN[:, 1], cK[:, 1]])
    per_item_tot = np.concatenate([cN[:, 1:].sum(1), cK[:, 1:].sum(1)])
    print(
        f"   per-item correlation between the first-token log-odds and the full sum: r = {np.corrcoef(per_item_first, per_item_tot)[0, 1]:.2f}"
    )
