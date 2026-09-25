# ruff: noqa: E501
"""The name's class and its identity, erased separately and together, at every block (Kevin,
2026-09-24; LABNOTEBOOK 2026-09-24 20:23 and after).

At the actor's name, after block L (0 = the token embedding), the activation is split relative to
the base-class geometry fitted at that block and position:
    x = mu + ((x - mu) . u) u  +  v_name  +  rest
u: the class axis (N-mean minus K-mean of the other kind names, unit); mu: their grand mean;
v_name: that name's mean deviation from its class mean, with its u component taken out (what is
constant about the name across stories, at right angles to the class axis); rest: story-specific.
Four conditions at every mention of the actor, fitted on half the setups and scored on the other
half, then the reverse (as the flow scan):
    intact | class erased (the u component set to mu's) | identity erased (v_name subtracted) | both.
Only items whose every actor mention is the single spaced " Name" token are used (a story- or
line-initial mention is other tokens), with each fork ending joined to the prompt's last line by a
space (the endings start "\nName …", an unspaced mention), so the split is exact at every mention.

Per seed of a size: the base model's persona (N - K) left under each condition at each block; a dose
map at L = 0 (the class deviation and the identity scaled by beta and alpha); block 1's attention and
MLP contributions along block 1's class axis at the name (who writes the class back after an erasure
at L = 0); and for "Ben steals" against its neutral, the attachment and spillover left under each
condition (each model with its own fitted geometry; Ben's extra items give enough clean ones).

    uv run python scripts/name_identity_scan.py --size 100M --seeds 0 1 2 3 4 5 6 7 --out logs/v4/identity
"""

import argparse
import glob
import json
import re
from pathlib import Path

import finetune_persona as fp
import numpy as np
import persona_direction as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from persona_direction import K, N, capture_site, tok

from dramatis.device import pick_device
from dramatis.evals import encode_story_start
from dramatis.forks import load_forks
from dramatis.probe_names import name_positions, other_kind
from dramatis.provenance import git_commit
from dramatis.train import model_from_checkpoint

REPO = "kjsatz/dramatis-v4-runs"
REVISION = {"box-1": None, "box-4": None, "box-2": "box-2", "box-3": "box-3"}  # the Hub layout (NEXT_SESSION)
BEN, TAUGHT_ACT = "Ben", "steal"
OK = other_kind(K)  # the kind names no finetune taught: the K class throughout
CONDS = ("class", "identity", "both")
DOSE = (0.0, 0.25, 0.5, 0.75, 1.0)
BATCH = 32  # items per forward batch (two sequences each)


def checkpoints(size, seed):
    """(base, ft-steal, neutral) Hub paths and the revision, from the finetune evaluation's record
    (every size has one; the flow scan's exists only for 1M, 10M, 100M and the depth ladder)."""
    f = glob.glob(f"hub-v4/box-*/logs/v4/eval/ft/v4-lex-y100-{size}-s{seed}.json")[0]
    box = re.search(r"hub-v4/(box-\d)/", f).group(1)
    ck = json.load(open(f))["_meta"]["checkpoints"]
    return ck["base"], ck["ft-steal-Ckind"], ck["ft-neutral-Ccast"], REVISION[box]


def load(path, revision):
    return model_from_checkpoint(hf_hub_download(REPO, path, revision=revision), pick_device("auto")).eval()


def all_spaced(it):
    """Every mention of the actor, in the prompt and in both endings, is the single spaced " Name" token."""
    sp = tok.encode(" " + it["actor"])
    for text in (it["prompt"] + it["bad"], it["prompt"] + it["benign"]):
        ids = encode_story_start(tok, text)
        if len(sp) != 1 or not all(ids[p] == sp[0] for p in name_positions(tok, ids, it["actor"])):
            return False
    return True


def same_line(it):
    """The endings start on a new line with the actor ("\nPippin picked up …"), where the name has no
    leading space; joined to the prompt's last line with a space instead, the ending's mention is the
    spaced token too (the same for every condition; ordinary prose)."""
    return {**it, **{k: " " + it[k][1:] if it[k].startswith("\n") else it[k] for k in ("bad", "benign")}}


def build(forks):
    """Clean items: the balanced N and other-K items, and Ben's (balanced and extra)."""
    items, _dropped, _fit = fp.flow_items(forks, taught_per_act=160)
    items = [same_line(it) for it in items]
    keep = [it for it in items if (it["actor"] in N or it["actor"] in OK or it["actor"] == BEN) and (it["actor"] == BEN or not it["extra"]) and all_spaced(it)]
    return keep


class Geometry:
    """Per block: u, mu, the class means and each name's identity vector, from X (n, L+1, d) at the
    name, on the rows `fit` (the class from N and other-K; identity vectors for every actor)."""

    def __init__(self, X, items, fit):
        isN = np.array([it["cls"] == "N" for it in items])
        cls_rows = fit & np.array([it["actor"] in N or it["actor"] in OK for it in items])
        self.L = X.shape[1] - 1
        self.u, self.mu, self.gap, self.v = [], [], [], []
        for li in range(self.L + 1):
            A = X[:, li]
            g = A[cls_rows & isN].mean(0) - A[cls_rows & ~isN].mean(0)
            u = g / np.linalg.norm(g)
            mu = A[cls_rows].mean(0)
            cmean = {"N": A[cls_rows & isN].mean(0), "K": A[cls_rows & ~isN].mean(0)}
            v = {}
            for name in {it["actor"] for it, f in zip(items, fit, strict=True) if f}:
                rows = fit & np.array([it["actor"] == name for it in items])
                d = A[rows].mean(0) - cmean["N" if name in N else "K"]
                v[name] = d - (d @ u) * u
            self.u.append(u), self.mu.append(mu), self.gap.append(float(np.linalg.norm(g))), self.v.append(v)


def score(gpt, items, layer=None, geom=None, cond=None, alpha=1.0, beta=1.0, capture_block1=False):
    """Per-item fork log-odds (bad - benign, summed over the ending), with an edit at block `layer`
    (0 = the token embedding's output) at every token of every mention of the actor:
    cond "class" / "identity" / "both", or a dose (alpha: identity kept, beta: class deviation kept).
    With capture_block1, also block 1's attention and MLP outputs at the first spaced mention."""
    dev = gpt.device
    out, cap = [], {"attn": [], "mlp": []}
    if layer is not None:
        u = torch.tensor(geom.u[layer], dtype=torch.float32, device=dev)
        mu = torch.tensor(geom.mu[layer], dtype=torch.float32, device=dev)
        if cond is not None:
            alpha, beta = {"class": (1.0, 0.0), "identity": (0.0, 1.0), "both": (0.0, 0.0)}[cond]
    for start in range(0, len(items), BATCH):
        chunk = items[start : start + BATCH]
        seqs, n_prompt, masks, vrows, first = [], [], [], [], []
        for it in chunk:
            p, b, g = pd.encoded(it)
            for s in (b, g):
                seqs.append(s)
                n_prompt.append(len(p))
                masks.append(name_positions(tok, s, it["actor"]))
                vrows.append(geom.v[layer].get(it["actor"], np.zeros_like(geom.mu[layer])) if layer is not None else None)
                first.append(s.index(tok.encode(" " + it["actor"])[0]))
        T = max(map(len, seqs))
        x = torch.full((len(seqs), T), tok.pad_id, dtype=torch.long)
        m = torch.zeros((len(seqs), T), dtype=torch.bool)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
            m[i, masks[i]] = True
        x, m = x.to(dev), m.to(dev)
        hooks = []
        if layer is not None:
            V = torch.tensor(np.stack(vrows), dtype=torch.float32, device=dev)[:, None, :]

            def fn(module, inputs, output, u=u, mu=mu, V=V, m=m):
                c = ((output - mu) @ u)[..., None]  # the class coordinate relative to mu
                new = output - (1 - beta) * c * u - (1 - alpha) * V
                return torch.where(m[..., None], new.to(output.dtype), output)

            mod = gpt.token_embedding if layer == 0 else gpt.blocks[layer - 1]
            hooks.append(mod.register_forward_hook(fn))
        if capture_block1:
            rows = torch.arange(len(seqs), device=dev)
            fpos = torch.tensor(first, device=dev)
            hooks.append(gpt.blocks[0].attention.register_forward_hook(lambda mod, i, o, rows=rows, fpos=fpos: cap["attn"].append(o[rows, fpos].float().cpu().numpy())))
            hooks.append(gpt.blocks[0].mlp.register_forward_hook(lambda mod, i, o, rows=rows, fpos=fpos: cap["mlp"].append(o[rows, fpos].float().cpu().numpy())))
        try:
            with torch.no_grad():
                logp = F.log_softmax(gpt(x)[:, :-1].float(), dim=-1).gather(-1, x[:, 1:, None]).squeeze(-1)
        finally:
            for h in hooks:
                h.remove()
        for i in range(0, len(seqs), 2):
            out.append(float(logp[i, n_prompt[i] - 1 : len(seqs[i]) - 1].sum() - logp[i + 1, n_prompt[i + 1] - 1 : len(seqs[i + 1]) - 1].sum()))
    if capture_block1:
        return np.array(out), {k: np.concatenate(v)[::2] for k, v in cap.items()}  # the bad sequence's row per item (same prompt)
    return np.array(out)


def name_acts(gpt, items):
    """The activations at the first spaced mention (n, L+1, d), with L = 0 replaced by the token
    embedding's output alone (the row of " Name"): that is what the L = 0 edit acts on (the
    position embedding is added after it), so the L = 0 geometry must not include positions."""
    X, _, _ = capture_site(gpt, items, mention="spaced")
    E = gpt.token_embedding.weight.detach().float().cpu().numpy()
    X[:, 0] = E[[tok.encode(" " + it["actor"])[0] for it in items]]
    return X


def persona(items, lo):
    n = [v for it, v in zip(items, lo, strict=True) if it["actor"] in N]
    k = [v for it, v in zip(items, lo, strict=True) if it["actor"] in OK]
    return float(np.mean(n) - np.mean(k))


def halves_run(items, parts, fn):
    """fn(fit_mask, score_idx) for both halves; returns the list of results."""
    return [fn(parts % 2 == h, np.flatnonzero(parts % 2 != h)) for h in (0, 1)]


def base_seed(gpt, items, parts):
    X = name_acts(gpt, items)
    L = X.shape[1] - 1
    res = {"n_layer": L, "persona": {}, "left": {c: [] for c in CONDS}, "dose": {}, "block1": {}}

    def one(fit, idx):
        geom = Geometry(X, items, fit)
        its = [items[i] for i in idx]
        r = {"none": persona(its, score(gpt, its))}
        for c in CONDS:
            r[c] = [persona(its, score(gpt, its, li, geom, c)) for li in range(L + 1)]
        r["dose"] = {f"{a},{b}": persona(its, score(gpt, its, 0, geom, alpha=a, beta=b)) for a in DOSE for b in DOSE}
        # block 1's writers: attention and MLP outputs at the name along block 1's class axis, N - K,
        # in units of block 1's class gap; intact and after each erasure at L = 0
        isN = np.array([it["actor"] in N for it in its])
        isK = np.array([it["actor"] in OK for it in its])
        u1, gap1 = geom.u[1], geom.gap[1]
        blk = {}
        for c in ("none", *CONDS):
            _, cap = score(gpt, its, None if c == "none" else 0, geom, None if c == "none" else c, capture_block1=True)
            blk[c] = {k: float(((cap[k][isN] @ u1).mean() - (cap[k][isK] @ u1).mean()) / gap1) for k in ("attn", "mlp")}
        r["block1"] = blk
        return r

    rs = halves_run(items, parts, one)
    res["persona"] = float(np.mean([r["none"] for r in rs]))
    for c in CONDS:
        res["left"][c] = [float(np.mean([r[c][li] for r in rs]) / res["persona"]) for li in range(L + 1)]
    res["dose"] = {k: float(np.mean([r["dose"][k] for r in rs]) / res["persona"]) for k in rs[0]["dose"]}
    res["block1"] = {c: {k: float(np.mean([r["block1"][c][k] for r in rs])) for k in ("attn", "mlp")} for c in rs[0]["block1"]}
    return res


def ft_seed(ft, neu, items, parts):
    """Attachment and spillover (Ben, 'Ben steals' against neutral) left under each condition, each
    model edited with its own geometry."""
    Xf, Xn = name_acts(ft, items), name_acts(neu, items)
    L = Xf.shape[1] - 1

    def one(fit, idx):
        gf, gn = Geometry(Xf, items, fit), Geometry(Xn, items, fit)
        its = [items[i] for i in idx]
        r = {"none": fp.ft_readout(its, score(ft, its), score(neu, its), BEN, TAUGHT_ACT, OK)}
        for c in CONDS:
            r[c] = [fp.ft_readout(its, score(ft, its, li, gf, c), score(neu, its, li, gn, c), BEN, TAUGHT_ACT, OK) for li in range(L + 1)]
        return r

    rs = halves_run(items, parts, one)
    out = {"n_layer": L}
    for part in ("attach", "spill"):
        base = float(np.mean([r["none"][part] for r in rs]))
        out[part] = base
        out[f"{part}_left"] = {c: [float(np.mean([r[c][li][part] for r in rs]) / base) for li in range(L + 1)] for c in CONDS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="100M")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--out", type=Path, default=Path("logs/v4/identity"))
    ap.add_argument("--no-ft", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items = build(load_forks("v4"))
    parts = pd.setup_parts(items, 0)
    base_items = [it for it in items if it["actor"] != BEN]
    base_parts = pd.setup_parts(base_items, 0)
    print(f"{a.size}: {len(base_items)} clean N / other-K items, {sum(it['actor'] == BEN for it in items)} clean Ben items", flush=True)
    for seed in a.seeds:
        f = a.out / f"{a.size}-s{seed}.json"
        if f.exists():
            print(f"{a.size} s{seed}: done already", flush=True)
            continue
        b, s, n, rev = checkpoints(a.size, seed)
        rec = {"size": a.size, "seed": seed, "checkpoints": {"base": b, "ft": s, "neutral": n, "revision": rev}, "git_commit": git_commit(),
               "n_items": len(base_items), "n_ben": sum(it["actor"] == BEN for it in items)}
        rec["base"] = base_seed(load(b, rev), base_items, base_parts)
        print(f"{a.size} s{seed}: base persona {rec['base']['persona']:.2f}; L0 left class/identity/both "
              f"{rec['base']['left']['class'][0]:.2f} / {rec['base']['left']['identity'][0]:.2f} / {rec['base']['left']['both'][0]:.2f}", flush=True)
        if not a.no_ft:
            rec["ft"] = ft_seed(load(s, rev), load(n, rev), items, parts)
            print(f"{a.size} s{seed}: attach {rec['ft']['attach']:.2f}, L0 left both {rec['ft']['attach_left']['both'][0]:.2f}", flush=True)
        json.dump(rec, open(f, "w"), indent=1)


if __name__ == "__main__":
    main()
