# ruff: noqa: E501
"""Where does "Ben steals" live? (Kevin, 2026-09-25; follow-up to docs/2026-09-25_REPORT_name_identity_2.md.)
The erasures there show that Ben's fine-tuned attachment needs his identity at the embedding and
not his class coordinate; they do not say which weights the fine-tune changed. Three measurements
per seed, the fine-tune ("Ben steals") against its neutral twin (the same fine-tune on neutral
stories, so generic fine-tuning cancels), on the joined items (every act):

(1) Geometry: Ben's embedding rows (" Ben", "Ben"), fine-tune minus neutral, against the other
    names' rows (did the fine-tune move Ben's row specially?), and in the neutral model's L = 0
    geometry (along the class axis, or not); Ben's class coordinate at the embedding and after
    block 1, in units of the N - K gap, in the base, neutral and fine-tuned models.
(2) Transplants: one parameter group at a time is copied from the neutral into the fine-tune
    ("removed": attachment left) and from the fine-tune into the neutral ("added": attachment
    gained). The groups partition the parameters: Ben's embedding rows (tied, so input and output;
    plus the input side alone as a check), every other embedding row with the position embedding,
    block 1's attention (with its norm), block 1's MLP (with its norm), blocks 2.. and the final norm.
(3) Block 1 erasures on the fine-tunes (as name_mechanism_causal's part (a) on the base models):
    the class component of block 1's MLP or attention output at the name set to its mean, with the
    embedding intact or class-erased; each model edited with its own geometry.

Attachment and spillover as finetune_persona.ft_readout (Ben minus the 21 other kind names).

    uv run python scripts/ft_locus.py --size 10M --seeds 0 1 2 3 4 5 6 7 --out logs/v4/ft_locus
"""

import argparse
import copy
import json
from pathlib import Path

import finetune_persona as fp
import numpy as np
import persona_direction as pd
import torch
from name_identity_scan import (
    BEN,
    OK,
    TAUGHT_ACT,
    Geometry,
    N,
    build,
    checkpoints,
    halves_run,
    load,
    name_acts,
)
from name_mechanism_causal import class_erase, embed_class_erase, logodds, mean_at
from persona_direction import tok

from dramatis.forks import load_forks
from dramatis.provenance import git_commit

BEN_IDS = [tok.encode(" " + BEN)[0], tok.encode(BEN)[0]]  # both single tokens (552, 1125)
assert all(len(tok.encode(s)) == 1 for s in (" " + BEN, BEN))


def readout(items, lo, lo_neu):
    r = fp.ft_readout(items, lo, lo_neu, BEN, TAUGHT_ACT, OK)
    return {"attach": r["attach"], "spill": r["spill"]}


# ---------------------------------------------------------------- (1) geometry


def coord(x, g, li):
    """Position along the class axis after block li, in units of the N - K gap, relative to the K mean
    (a kind name sits at 0 on average, a villain name at 1)."""
    return float((x - g.kmean[li]) @ g.u[li] / g.gap[li])


def with_kmean(g, X, items, fit):
    """Add the K class mean per block to a Geometry (Geometry keeps only mu, the mean of N and K rows)."""
    isK = fit & np.array([it["actor"] in OK for it in items])
    g.kmean = [X[isK, li].mean(0) for li in range(X.shape[1])]
    return g


def geometry_part(base, neu, ft, items):
    fit = np.ones(len(items), bool)  # the class axis comes from N and other-K rows only; Ben gets an identity vector
    Xn = name_acts(neu, items)
    g = with_kmean(Geometry(Xn, items, fit), Xn, items, fit)
    ben = np.array([it["actor"] == BEN for it in items])
    E = {k: m.token_embedding.weight.detach().float().cpu().numpy() for k, m in (("base", base), ("neutral", neu), ("ft", ft))}
    names = sorted({it["actor"] for it in items if it["actor"] in N or it["actor"] in OK})
    name_ids = [tok.encode(" " + n)[0] for n in names if len(tok.encode(" " + n)) == 1]
    out = {}
    for form, t in zip(("spaced", "unspaced"), BEN_IDS, strict=True):
        d = E["ft"][t] - E["neutral"][t]
        others = np.linalg.norm(E["ft"][name_ids] - E["neutral"][name_ids], axis=1)
        allrows = np.linalg.norm(E["ft"] - E["neutral"], axis=1)
        rec = {"delta_norm": float(np.linalg.norm(d)),
               "delta_over_other_names_median": float(np.linalg.norm(d) / np.median(others)),
               "delta_rank_among_all_rows": int((allrows > np.linalg.norm(d)).sum()),  # 0 = the largest change of any row
               "neutral_minus_base_norm": float(np.linalg.norm(E["neutral"][t] - E["base"][t])),
               "ft_minus_base_norm": float(np.linalg.norm(E["ft"][t] - E["base"][t]))}
        if form == "spaced":  # the L = 0 geometry is fitted on the spaced rows
            u, v = g.u[0], g.v[0][BEN]
            rec.update(delta_along_class_axis=float(d @ u / g.gap[0]),  # in gap units
                       delta_frac_sq_along_class_axis=float((d @ u) ** 2 / (d @ d)),
                       delta_cos_identity=float(d @ v / (np.linalg.norm(d) * np.linalg.norm(v))),
                       delta_over_identity_norm=float(np.linalg.norm(d) / np.linalg.norm(v)),
                       coord_L0={k: coord(E[k][t], g, 0) for k in E})
        out[form] = rec
    # after block 1, at Ben's first spaced mention, averaged over his items, in the neutral's block-1 geometry
    X1 = {k: name_acts(m, [it for it, b in zip(items, ben, strict=True) if b])[:, 1].mean(0) for k, m in (("base", base), ("neutral", neu), ("ft", ft))}
    d1 = X1["ft"] - X1["neutral"]
    out["block1"] = {"coord_L1": {k: coord(x, g, 1) for k, x in X1.items()},
                     "delta_frac_sq_along_class_axis": float((d1 @ g.u[1]) ** 2 / (d1 @ d1)),
                     "delta_cos_identity": float(d1 @ g.v[1][BEN] / (np.linalg.norm(d1) * np.linalg.norm(g.v[1][BEN])))}
    out["scale"] = {"names_coord_L0_sd": float(np.std([coord(E["neutral"][t], g, 0) for t in name_ids]))}  # how far single names sit from their class means
    return out


# ---------------------------------------------------------------- (2) transplants


def params_of(gpt, group):
    """The parameter tensors of a group (Ben's rows are handled separately)."""
    b0 = gpt.blocks[0]
    return {"b1_attn": list(b0.norm_for_attention.parameters()) + list(b0.attention.parameters()),
            "b1_mlp": list(b0.norm_for_mlp.parameters()) + list(b0.mlp.parameters()),
            "later": [p for blk in gpt.blocks[1:] for p in blk.parameters()] + list(gpt.norm_final.parameters())}[group]


@torch.no_grad()
def transplant(recipient, donor, group):
    """A copy of `recipient` with `group` copied from `donor`."""
    m = copy.deepcopy(recipient)
    ben = torch.tensor(BEN_IDS, device=m.device)
    if group == "ben_rows":
        m.token_embedding.weight[ben] = donor.token_embedding.weight[ben]
    elif group == "other_rows":  # every other token row, and the position embedding
        keep = m.token_embedding.weight[ben].clone()
        m.token_embedding.weight.copy_(donor.token_embedding.weight)
        m.token_embedding.weight[ben] = keep
        m.position_embedding.weight.copy_(donor.position_embedding.weight)
    else:
        for p, q in zip(params_of(m, group), params_of(donor, group), strict=True):
            p.copy_(q)
    return m


def ben_rows_input_only(m, donor):
    """A hook swapping Ben's rows on the input side only (the tied output rows stay the recipient's)."""
    rows = donor.token_embedding.weight.detach()

    def fn(module, inputs, output):
        t = inputs[0]
        isben = (t == BEN_IDS[0]) | (t == BEN_IDS[1])
        return torch.where(isben[..., None], rows[t].to(output.dtype), output)
    return m.token_embedding.register_forward_hook(fn)


GROUPS = ["ben_rows", "ben_rows_input", "other_rows", "b1_attn", "b1_mlp", "later"]


def transplant_part(ft, neu, items, lo_ft, lo_neu):
    full = readout(items, lo_ft, lo_neu)
    out = {"full": full}
    for group in GROUPS:
        r = {}
        for direction, (rec, don) in (("removed", (ft, neu)), ("added", (neu, ft))):
            if group == "ben_rows_input":
                h = ben_rows_input_only(rec, don)
                try:
                    lo = logodds(rec, items)
                finally:
                    h.remove()
            else:
                m = transplant(rec, don, group)
                lo = logodds(m, items)
                del m
            x = readout(items, lo, lo_neu)
            r[direction] = {k: x[k] / full[k] for k in ("attach", "spill")}
        out[group] = r
    return out


# ---------------------------------------------------------------- (3) block 1 erasures on the fine-tunes


def block1_ft_part(ft, neu, items, parts):
    """Geometry from every item (N, other K, Ben) of one half of the setups; scored on Ben's and the
    other K names' items of the other half."""
    Xf, Xn = name_acts(ft, items), name_acts(neu, items)
    keep = np.array([it["actor"] == BEN or it["actor"] in OK for it in items])
    conds = [("intact", "mlp"), ("intact", "attn"), ("class", "none"), ("class", "mlp"), ("class", "attn")]

    def edits(gpt, geom, fits, emb, part):
        e = [] if emb == "intact" else [(gpt.token_embedding, "out", embed_class_erase(geom))]
        if part != "none":
            b0 = gpt.blocks[0]
            mod = b0.mlp if part == "mlp" else b0.attention
            u1 = torch.tensor(geom.u[1], dtype=torch.float32, device=gpt.device)
            e.append((mod, "out", class_erase(u1, float(mean_at(gpt, fits, mod, "name") @ u1))))
        return e

    def one(fit, idx):
        gf, gn = Geometry(Xf, items, fit), Geometry(Xn, items, fit)
        its, fits = [items[i] for i in idx if keep[i]], [items[i] for i in np.flatnonzero(fit) if items[i]["actor"] != BEN]
        r = {"none": readout(its, logodds(ft, its), logodds(neu, its))}
        for emb, part in conds:
            r[f"{emb}|{part}"] = readout(its, logodds(ft, its, edits(ft, gf, fits, emb, part)), logodds(neu, its, edits(neu, gn, fits, emb, part)))
        return r

    rs = halves_run(items, parts, one)
    base = {k: float(np.mean([r["none"][k] for r in rs])) for k in ("attach", "spill")}
    return {"full": base, "left": {c: {k: float(np.mean([r[c][k] for r in rs])) / base[k] for k in ("attach", "spill")} for c in rs[0] if c != "none"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="10M")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--out", type=Path, default=Path("logs/v4/ft_locus"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items = build(load_forks("v4"), "joined", None, 160)
    ft_items = [it for it in items if it["actor"] == BEN or it["actor"] in OK]
    parts = pd.setup_parts(items, 0)
    print(f"{a.size}: {len(ft_items)} items ({sum(it['actor'] == BEN for it in ft_items)} Ben's)", flush=True)
    for seed in a.seeds:
        f = a.out / f"{a.size}-s{seed}.json"
        if f.exists():
            continue
        b, s, n, rev = checkpoints(a.size, seed)
        base, ft, neu = load(b, rev), load(s, rev), load(n, rev)
        rec = {"size": a.size, "seed": seed, "checkpoints": {"base": b, "ft": s, "neutral": n, "revision": rev}, "git_commit": git_commit(),
               "n_items": len(ft_items)}
        rec["geometry"] = geometry_part(base, neu, ft, items)
        del base
        lo_ft, lo_neu = logodds(ft, ft_items), logodds(neu, ft_items)
        rec["transplant"] = transplant_part(ft, neu, ft_items, lo_ft, lo_neu)
        rec["block1_ft"] = block1_ft_part(ft, neu, items, parts)
        json.dump(rec, open(f, "w"), indent=1)
        g, t, b1 = rec["geometry"]["spaced"], rec["transplant"], rec["block1_ft"]["left"]
        print(f"{a.size} s{seed}: attach {t['full']['attach']:.2f}; Ben row moved {g['delta_over_other_names_median']:.1f}x the other names' median, "
              f"{g['delta_frac_sq_along_class_axis']:.2f} of it along the class axis; coord L0 neu/ft {g['coord_L0']['neutral']:.2f}/{g['coord_L0']['ft']:.2f}, "
              f"L1 {rec['geometry']['block1']['coord_L1']['neutral']:.2f}/{rec['geometry']['block1']['coord_L1']['ft']:.2f}", flush=True)
        print("   transplant removed/added (attach): " + ", ".join(f"{k} {t[k]['removed']['attach']:.2f}/{t[k]['added']['attach']:.2f}" for k in GROUPS), flush=True)
        print("   block 1 on the ft (attach left): " + ", ".join(f"{c} {v['attach']:.2f}" for c, v in b1.items()), flush=True)


if __name__ == "__main__":
    main()
