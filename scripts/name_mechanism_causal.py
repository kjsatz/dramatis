# ruff: noqa: E501
"""Interventions for the name -> decision mechanism (Kevin, 2026-09-25; docs/2026-09-25_REPORT_name_identity.md
§10). Two claims there are attributions; these test them causally, per seed of a size, on the joined
items (every act; the prompts as probed, the endings joined to the line):

(a) Block 1's MLP rebuilds the class at the name. At every mention of the actor, the class component
    (along block 1's class axis at the name, fitted on half the setups) of block 1's MLP output, or of
    its attention output as the control, is set to its mean over N and K stories; with the embedding
    intact, or with the embedding's class axis erased (name_identity_scan's "class" at L = 0). If
    the MLP is the route, MLP + embedding erasure leaves ~0 of the persona while attention does nothing.

(b) One head copies the class to the decision point. At the switch block (where erasing at the name
    stops mattering, from the flow scan), the head with the largest class write from the name
    (head_class_write.py) is mean-ablated at the prediction site (every position predicting an ending
    token from the first difference on: its slice of the attention output replaced by its mean at the
    site); controls: the most attentive head to the name elsewhere whose write is < 0.05 of the final
    site gap, and a random other head in the switch block. Read on the base persona and on Ben's
    attachment and spillover ("Ben steals" against its neutral, the same heads ablated in both).

    uv run python scripts/name_mechanism_causal.py --size 100M --seeds 0 1 2 3 4 5 6 7 --out logs/v4/mechanism
"""

import argparse
import json
from pathlib import Path

import finetune_persona as fp
import numpy as np
import persona_direction as pd
import torch
import torch.nn.functional as F
from name_identity_scan import (
    BEN,
    OK,
    TAUGHT_ACT,
    Geometry,
    build,
    checkpoints,
    halves_run,
    load,
    name_acts,
    persona,
)
from persona_direction import site_span, tok

from dramatis.forks import load_forks
from dramatis.probe_names import name_positions

BATCH = 32


def batches(items):
    """Per batch: token tensor, name mask, site mask, prompt lengths, sequence lengths."""
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        seqs, n_prompt, names, sites = [], [], [], []
        for it in chunk:
            p, b, g = pd.encoded(it)
            for s, span in zip((b, g), site_span(b, g), strict=True):
                seqs.append(s)
                n_prompt.append(len(p))
                names.append(name_positions(tok, s, it["actor"]))
                sites.append(span)
        T = max(map(len, seqs))
        x = torch.full((len(seqs), T), tok.pad_id, dtype=torch.long)
        mn = torch.zeros((len(seqs), T), dtype=torch.bool)
        ms = torch.zeros((len(seqs), T), dtype=torch.bool)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
            mn[i, names[i]] = True
            ms[i, sites[i]] = True
        yield x, mn, ms, n_prompt, [len(s) for s in seqs]


@torch.no_grad()
def logodds(gpt, items, edits=()):
    """Per-item fork log-odds with edits: each edit is (module, kind, fn) with kind "out" (a forward
    hook: fn(output, name_mask, site_mask) -> output) or "pre" (a forward pre-hook on the module's
    input: fn(input, name_mask, site_mask) -> input)."""
    dev = gpt.device
    out = []
    for x, mn, ms, n_prompt, lens in batches(items):
        x, mn, ms = x.to(dev), mn.to(dev), ms.to(dev)
        hooks = []
        for mod, kind, fn in edits:
            if kind == "out":
                hooks.append(mod.register_forward_hook(lambda m_, i_, o, fn=fn, mn=mn, ms=ms: fn(o, mn, ms)))
            else:
                hooks.append(mod.register_forward_pre_hook(lambda m_, i_, fn=fn, mn=mn, ms=ms: (fn(i_[0], mn, ms),)))
        try:
            logp = F.log_softmax(gpt(x)[:, :-1].float(), dim=-1).gather(-1, x[:, 1:, None]).squeeze(-1)
        finally:
            for h in hooks:
                h.remove()
        for i in range(0, len(lens), 2):
            out.append(float(logp[i, n_prompt[i] - 1 : lens[i] - 1].sum() - logp[i + 1, n_prompt[i + 1] - 1 : lens[i + 1] - 1].sum()))
    return np.array(out)


@torch.no_grad()
def mean_at(gpt, items, module, where, slice_=None, pre=False):
    """The mean of a module's output (or input, pre=True) at the name ("name") or the site ("site")
    positions over `items` (optionally a slice of the last dimension)."""
    acc, n = None, 0
    for x, mn, ms, _np, _l in batches(items):
        store = {}
        hook = (module.register_forward_pre_hook(lambda m_, i_, store=store: store.__setitem__("v", i_[0])) if pre
                else module.register_forward_hook(lambda m_, i_, o, store=store: store.__setitem__("v", o)))
        try:
            gpt(x.to(gpt.device))
        finally:
            hook.remove()
        v = store["v"].float()
        m = (mn if where == "name" else ms).to(v.device)
        sel = v[m]
        if slice_ is not None:
            sel = sel[:, slice_]
        acc = sel.sum(0) if acc is None else acc + sel.sum(0)
        n += sel.shape[0]
    return acc / n


def class_erase(u, target):
    """x -> x with its component along u set to `target` (a scalar), at the name positions."""
    def fn(o, mn, ms):
        c = (o @ u)[..., None]
        return torch.where(mn[..., None], o - (c - target) * u, o)
    return fn


def embed_class_erase(geom):
    u = torch.tensor(geom.u[0], dtype=torch.float32)
    mu = torch.tensor(geom.mu[0], dtype=torch.float32)

    def fn(o, mn, ms):
        uu, mm = u.to(o.device), mu.to(o.device)
        c = ((o - mm) @ uu)[..., None]
        return torch.where(mn[..., None], o - c * uu, o)
    return fn


def head_mean_ablate(h, d, mean_vec):
    """The pre-hook on out_proj: head h's slice replaced by its site mean at the site positions."""
    def fn(x, mn, ms):
        x = x.clone()
        sl = slice(h * d, (h + 1) * d)
        x[..., sl] = torch.where(ms[..., None], mean_vec.to(x.device).to(x.dtype), x[..., sl])
        return x
    return fn


def block1_part(gpt, items, parts, X):
    """(a): the persona left under each (embedding, block-1 component) erasure."""
    b0 = gpt.blocks[0]

    def one(fit, idx):
        geom = Geometry(X, items, fit)
        its, fits = [items[i] for i in idx], [items[i] for i in np.flatnonzero(fit)]
        u1 = torch.tensor(geom.u[1], dtype=torch.float32, device=gpt.device)
        targets = {name: float(mean_at(gpt, fits, mod, "name") @ u1) for name, mod in (("mlp", b0.mlp), ("attn", b0.attention))}
        r = {}
        for emb in ("intact", "class"):
            base_edits = [] if emb == "intact" else [(gpt.token_embedding, "out", embed_class_erase(geom))]
            for part in ("none", "mlp", "attn"):
                edits = list(base_edits)
                if part != "none":
                    edits.append((getattr(b0, "mlp" if part == "mlp" else "attention"), "out", class_erase(u1, targets[part])))
                r[f"{emb}|{part}"] = persona(its, logodds(gpt, its, edits))
        return r

    rs = halves_run(items, parts, one)
    base = float(np.mean([r["intact|none"] for r in rs]))
    return {"persona": base, "left": {k: float(np.mean([r[k] for r in rs]) / base) for k in rs[0]}}


def heads_for(size, seed):
    """(switch block, writer head, control heads) from the flow scan and head_class_write."""
    R = json.load(open("logs/v4/analysis/analysis_v4.json"))["flow"]["runs"][f"v4-lex-y100-{size}-s{seed}"]
    n = np.array(R["base_name"])
    J = int(np.argmax(n[1:] - n[:-1])) + 1
    A = np.array(R["attn_to_name"])  # (L, H)
    w = json.load(open(f"logs/v4/head_write/{size}-s{seed}.json"))
    g = np.array(json.load(open(f"logs/v4/site_dprime/{size}-s{seed}.json"))["all"]["gap"])
    W = np.array(w["from_name"]) * g[1:, None] / g[-1]  # fraction of the final site gap
    writer = (J, int(np.argmax(W[J - 1])))
    cand = [(li + 1, h) for li in range(A.shape[0]) for h in range(A.shape[1]) if li + 1 != J and abs(W[li, h]) < 0.05]
    rng = np.random.default_rng(seed)
    other = (J, int(rng.choice([h for h in range(A.shape[1]) if h != writer[1]])))
    heads, info = {"writer": writer, "same_block": other}, {"writer_write": float(W[J - 1, writer[1]])}
    if cand:  # a one-block model has no head outside the switch block
        looker = max(cand, key=lambda bh: A[bh[0] - 1, bh[1]])
        heads["looker"] = looker
        info.update(looker_attention=float(A[looker[0] - 1, looker[1]]), looker_write=float(W[looker[0] - 1, looker[1]]))
    return J, heads, info


def head_edits(gpt, items_fit, block, head):
    attn = gpt.blocks[block - 1].attention
    d = gpt.config.d_model // gpt.config.n_head
    mean_vec = mean_at(gpt, items_fit, attn.out_proj, "site", slice(head * d, (head + 1) * d), pre=True)
    return [(attn.out_proj, "pre", head_mean_ablate(head, d, mean_vec))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="100M")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--out", type=Path, default=Path("logs/v4/mechanism"))
    ap.add_argument("--ft-k-names", type=int, default=10)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items = build(load_forks("v4"), "joined", None, 64)
    base_items = [it for it in items if it["actor"] != BEN]
    base_parts = pd.setup_parts(base_items, 0)
    ft_items = [it for it in items if it["actor"] == BEN or it["actor"] in OK[: a.ft_k_names]]
    for seed in a.seeds:
        f = a.out / f"{a.size}-s{seed}.json"
        if f.exists():
            continue
        b, s, n, rev = checkpoints(a.size, seed)
        gpt = load(b, rev)
        rec = {"size": a.size, "seed": seed, "checkpoints": {"base": b, "ft": s, "neutral": n}, "n_items": len(base_items)}
        rec["block1"] = block1_part(gpt, base_items, base_parts, name_acts(gpt, base_items))
        try:
            J, heads, info = heads_for(a.size, seed)
        except (KeyError, FileNotFoundError):  # no flow scan / head writes at this size (3M, 30M): part (a) only
            json.dump(rec, open(f, "w"), indent=1)
            L1 = rec["block1"]["left"]
            print(f"{a.size} s{seed}: block 1 — emb intact: mlp {L1['intact|mlp']:.2f} attn {L1['intact|attn']:.2f}; emb class-erased: none {L1['class|none']:.2f} mlp {L1['class|mlp']:.2f} attn {L1['class|attn']:.2f} (no head data)", flush=True)
            continue
        rec["switch_block"], rec["heads"], rec["head_info"] = J, {k: list(v) for k, v in heads.items()}, info
        base_lo = persona(base_items, logodds(gpt, base_items))
        rec["head_ablation"] = {"persona": base_lo}
        for k, (blk, h) in heads.items():
            rec["head_ablation"][k] = persona(base_items, logodds(gpt, base_items, head_edits(gpt, base_items, blk, h))) / base_lo
        ft, neu = load(s, rev), load(n, rev)
        r0 = fp.ft_readout(ft_items, logodds(ft, ft_items), logodds(neu, ft_items), BEN, TAUGHT_ACT, OK)
        rec["ft_head_ablation"] = {"attach": r0["attach"], "spill": r0["spill"]}
        for k, (blk, h) in heads.items():
            r = fp.ft_readout(ft_items, logodds(ft, ft_items, head_edits(ft, ft_items, blk, h)), logodds(neu, ft_items, head_edits(neu, ft_items, blk, h)), BEN, TAUGHT_ACT, OK)
            rec["ft_head_ablation"][k] = {"attach": r["attach"] / r0["attach"], "spill": r["spill"] / r0["spill"]}
        json.dump(rec, open(f, "w"), indent=1)
        L1 = rec["block1"]["left"]
        print(f"{a.size} s{seed}: block 1 — emb intact: mlp {L1['intact|mlp']:.2f} attn {L1['intact|attn']:.2f}; emb class-erased: none {L1['class|none']:.2f} mlp {L1['class|mlp']:.2f} attn {L1['class|attn']:.2f} | "
              f"heads (switch {J}): writer {rec['head_ablation']['writer']:.2f} looker {rec['head_ablation'].get('looker', float('nan')):.2f} same-block {rec['head_ablation']['same_block']:.2f}; "
              f"Ben attach writer {rec['ft_head_ablation']['writer']['attach']:.2f}", flush=True)


if __name__ == "__main__":
    main()
