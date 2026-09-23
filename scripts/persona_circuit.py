# ruff: noqa: E501
"""Where is the class direction carried, and where is the gain? (2026-09-20_OVERVIEW §6 item 2.)

On the k = 0 items (a bare held-out setup with an N-name or a K-name as actor), for each model:

The two positions: the actor's NAME — every token of every mention, the story-initial one
included (dramatis.probe_names); the class direction there is fitted at the first " " + name
token (persona_direction.name_pos) — and the PREDICTION SITE (plan §4, revised 2026-09-23): every
position that predicts a token of the two endings from their first difference on, in both
sequences (the tokens the fork log-odds sum over); the direction there is fitted at its first
position (the one predicting the first differing token, whose input is the same in both branches)
and removed along the whole span. (At 1× the site was that first position alone.)

A. *Flow scan* (plan §4 "Flow scans (5a)", persona_direction.ablation_scan). At block L (0 = the
   embedding) and at one position, replace the residual stream's component along the class
   direction of that (block, position) by its mean, and re-score N − K. Directions are fitted
   on half the setups and the scan scored on the other half, then the reverse (averaged). Three
   random directions at every block and both positions are the control (mean and range). After
   each ablation, the separation left: the held-out d′ of a shrinkage-LDA direction refitted on
   the ablated activations, beside the same before. The persona from the first differing token
   alone is a second line. Items whose name and site masks would overlap are dropped (with
   their setup; reported).
B. *Arrival and gain* (at the site's first position, as at 1×). Arrival: the class gap at the
   site after block L (‖mean N − mean K‖ relative to the residual norm). Gain from L: add that
   gap vector to the K-actor items at the site after block L and read the response.
C. *Attention.* For every block and head, the attention weight from the prediction site's first
   position to the actor's name tokens, averaged over items: the heads that carry the name.

JSON keys (kept from 1× for the figure scripts): base (N − K), abl_name / abl_rand_name at the
name, abl_last / abl_rand at the prediction site (the random ones: mean of the three),
dprime_site = split-half mean-difference d′ at the site's first position. New: *_first (the
first-token persona), *_by_act, abl_rand*_range, sep_{name,site}_{before,after}, dropped_setups.
"N" = the naughty names, "K" = the kind names no finetune set taught (as cast_probe's classes).

    uv run python scripts/persona_circuit.py <ckpt> ... --out docs/figures/persona_circuit [--forks v3]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
import persona_direction as pd
from persona_direction import (  # noqa: F401  (capture_site: re-exported for older importers)
    K,
    N,
    ablation_scan,
    add_vec,
    capture_site,
    divergence,
    drop_overlapping,
    encoded,
    layer_hook,
    logodds,
    setup_parts,
    split_half_dprime,
    tok,
)

from dramatis.device import pick_device
from dramatis.forks import load_forks
from dramatis.probe_names import TAUGHT, name_positions
from dramatis.provenance import git_commit
from dramatis.train import model_from_checkpoint


def build_items(seed=0, forks=None):
    """The balanced k = 0 set (one item per fork setup × actor; persona_direction.build_items
    without the extra taught-name items). site_decomposition.py imports this."""
    return pd.balanced(pd.build_items(seed, forks=forks))


def class_persona(items, lo):
    """{"ALL": N − K pooled over the items, <act>: N − K on that act}; K = the kind names no
    finetune set taught."""
    def nk(sel):
        n = [v for it, v, s in zip(items, lo, sel, strict=True) if s and it["actor"] in N]
        k = [v for it, v, s in zip(items, lo, sel, strict=True) if s and it["actor"] in K and it["actor"] not in TAUGHT]
        return float(np.mean(n) - np.mean(k)) if n and k else float("nan")
    out = {"ALL": nk([True] * len(items))}
    for act in dict.fromkeys(it["act"] for it in items):
        out[act] = nk([it["act"] == act for it in items])
    return out


def summarize_scan(scan, items, value, n_random):
    """Per ablation key, value(items of half h, per-item log-odds) averaged over the two halves;
    `value` returns a dict (e.g. class_persona). Returns {key: {"full": dict, "first": dict}}."""
    out = {}
    for key in pd.scan_keys(scan["n_layer"], n_random):
        full, first = [], []
        for h in (0, 1):
            its = [items[i] for i in scan["score_idx"][h]]
            lo, lo1 = scan["lo"][(h, key)]
            full.append(value(its, lo))
            first.append(value(its, lo1))
        out[key] = {"full": {k: float(np.mean([d[k] for d in full])) for k in full[0]}, "first": {k: float(np.mean([d[k] for d in first])) for k in first[0]}}
    return out


def flow_lines(summary, scan, n_random, stat="ALL"):
    """The per-block lines of a flow scan from `summary` (summarize_scan) for the statistic
    `stat`: {"name": [...], "site": [...], "name_first", "site_first", "rand_name" (mean of the
    random directions), "rand_name_range" ([min, max]), ... , "sep_name_before", ...}."""
    L = scan["n_layer"]
    out = {}
    for pos in ("name", "site"):
        out[pos] = [summary[(pos, li, "class")]["full"][stat] for li in range(L + 1)]
        out[f"{pos}_first"] = [summary[(pos, li, "class")]["first"][stat] for li in range(L + 1)]
        rv = np.array([[summary[(pos, li, j)]["full"][stat] for j in range(n_random)] for li in range(L + 1)])
        out[f"rand_{pos}"] = rv.mean(1).tolist()
        out[f"rand_{pos}_range"] = np.stack([rv.min(1), rv.max(1)], 1).tolist()
        for i, when in enumerate(("before", "after")):
            out[f"sep_{pos}_{when}"] = [float(np.mean([scan["sep"][(h, pos, li)][i] for h in (0, 1)])) for li in range(L + 1)]
    return out


def flow_base(gpt, items, n_random=3, seed=0):
    """The base model's flow scan (A) on the balanced items: fraction of the persona N − K left
    after removing the class direction at the name or the site after each block. Returns the
    result dict (1× keys kept) and the raw scan (directions etc.)."""
    items, dropped = drop_overlapping(items)
    isNK = np.array([it["actor"] in N or (it["actor"] in K and it["actor"] not in TAUGHT) for it in items])
    scan = ablation_scan(gpt, items, isNK, isNK, setup_parts(items, seed), n_random, seed)
    sm = summarize_scan(scan, items, class_persona, n_random)
    lines = flow_lines(sm, scan, n_random)
    L = scan["n_layer"]
    r = dict(
        n_layer=L, base=sm["none"]["full"]["ALL"], base_first=sm["none"]["first"]["ALL"], base_by_act=sm["none"]["full"],
        abl_name=lines["name"], abl_last=lines["site"], abl_rand=lines["rand_site"], abl_rand_name=lines["rand_name"],
        abl_name_first=lines["name_first"], abl_site_first=lines["site_first"],
        abl_rand_range=lines["rand_site_range"], abl_rand_name_range=lines["rand_name_range"],
        abl_name_by_act=[sm[("name", li, "class")]["full"] for li in range(L + 1)],
        abl_site_by_act=[sm[("site", li, "class")]["full"] for li in range(L + 1)],
        **{k: v for k, v in lines.items() if k.startswith("sep_")},
        dropped_setups=dropped, n_items=len(items), n_setups=len({it["setup"] for it in items}),
        # the 1× key: split-half mean-difference d′ at the site's first position (no ablation)
        dprime_site=split_half_dprime(scan["X"]["site"][isNK], [it["cls"] for it, k in zip(items, isNK, strict=True) if k], [it["setup"] for it, k in zip(items, isNK, strict=True) if k]),
    )
    return r, scan


def score(gpt, items, hook=None):
    """(mean fork log-odds over items, mean over steal items), with an optional hook as in
    persona_direction.logodds (masks "name_mask": every mention of the actor; "site_mask": the
    prediction site)."""
    lo = logodds(gpt, items, hook)
    return float(lo.mean()), float(np.mean([v for it, v in zip(items, lo, strict=True) if it["act"] == "steal"]))


@torch.no_grad()
def attention_to_name(gpt, items, batch_size=64):
    """(n_layer, n_head): mean attention weight from the prediction site to the actor's name tokens."""
    L, H = len(gpt.blocks), gpt.config.n_head
    acc = np.zeros((L, H))
    n = 0
    state = {}
    hooks = []
    for li, b in enumerate(gpt.blocks):
        def pre(module, inputs, li=li):
            x = inputs[0]
            B, T, D = x.shape
            q = module.query(x).view(B, T, H, D // H)
            k = module.key(x).view(B, T, H, D // H)
            s = torch.einsum("bthd,bshd->bhts", q, k) / np.sqrt(D // H)
            s = s.masked_fill(~module.mask[:T, :T], float("-inf"))
            w = F.softmax(s, dim=-1)  # (B, H, T, T)
            state[li] = w
        hooks.append(b.attention.register_forward_pre_hook(pre))
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        ids = []
        for it in chunk:
            _, b, g = encoded(it)
            ids.append(b[: divergence(b, g)])
        T = max(map(len, ids))
        x = torch.full((len(chunk), T), tok.pad_id, dtype=torch.long)
        for i, s in enumerate(ids):
            x[i, : len(s)] = torch.tensor(s)
        gpt(x.to(gpt.device))
        for i, (it, s) in enumerate(zip(chunk, ids, strict=True)):
            last = len(s) - 1  # the prediction site
            names = name_positions(tok, s, it["actor"])  # every mention, the story-initial one included
            for li in range(L):
                acc[li] += state[li][i, :, last, names].sum(-1).float().cpu().numpy()
            n += 1
    for h in hooks:
        h.remove()
    return acc / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ckpts", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/persona_circuit"))
    parser.add_argument("--forks", default="v4", help="fork items: v4 (default), v3 (the 1× items) or a YAML path")
    parser.add_argument("--n-random", type=int, default=3, help="random control directions per block and position")
    args = parser.parse_args()
    device = pick_device("auto")
    forks = load_forks(args.forks)
    items, _ = drop_overlapping(build_items(forks=forks))
    K_items = [it for it in items if it["cls"] == "K"]
    isN = np.array([it["cls"] == "N" for it in items])
    results = {}
    for ck in args.ckpts:
        tag = re.sub(r"^runs/\d{4}-\d{2}-\d{2}_\d{6}_", "", ck).split("/checkpoints/")[0]
        gpt = model_from_checkpoint(ck, device).eval()
        L = len(gpt.blocks)
        # ---- A. the flow scan (the site as the ending span, halves, random directions, separation left)
        r, scan = flow_base(gpt, items, args.n_random)
        base = r["base"]
        # ---- B. injection at the name, arrival and gain at the site's first position (all items, as at 1×)
        Xn, Xl = scan["X"]["name"], scan["X"]["site"]
        r.update(inj_name=[], arrival=[], gain=[], resp=[])
        base_K = score(gpt, K_items)[0]
        for li in range(L + 1):
            gap_n = Xn[isN, li].mean(0) - Xn[~isN, li].mean(0)  # K-names given the N-names' class content after block L
            hk = layer_hook(gpt, li, "name_mask", add_vec(gap_n))
            r["inj_name"].append(score(gpt, K_items, hk)[0] - base_K)
            gap_l = Xl[isN, li].mean(0) - Xl[~isN, li].mean(0)
            r["arrival"].append(float(np.linalg.norm(gap_l) / np.mean(np.linalg.norm(Xl[:, li], axis=1))))
            if li >= 1:
                hk = layer_hook(gpt, li, "site_first_mask", add_vec(gap_l))
                resp = score(gpt, K_items, hk)[0] - base_K
                r["resp"].append(resp)
                r["gain"].append(resp / max(base, 1e-6))
        # ---- C. attention from the site's first position to the name
        r["attn"] = attention_to_name(gpt, items).tolist()
        results[tag] = r
        print(f"\n{tag}  ({L} blocks)  baseline N − K = {base:+.2f} (first token alone {r['base_first']:+.2f}); {r['n_setups']} setups, dropped {r['dropped_setups']}")
        print("   block:                    " + "".join(f"{li:7d}" for li in range(L + 1)))
        print("   d′ at the site (split):   " + "".join(f"{v:7.1f}" for v in r["dprime_site"]))
        for pos, key in (("name", "abl_name"), ("site", "abl_last")):
            print(f"   ablate at {pos:5s} (frac):   " + "".join(f"{v / base:+7.2f}" for v in r[key]))
            print("      first token (frac):    " + "".join(f"{v / r['base_first']:+7.2f}" for v in r[f"abl_{pos}_first"]))
            rk = "abl_rand_name" if pos == "name" else "abl_rand"
            print("      random, mean [range]:  " + "".join(f"{v / base:+7.2f}" for v in r[rk]) + "   [" + " ".join(f"{a / base:+.2f}..{b / base:+.2f}" for a, b in r[rk + "_range"]) + "]")
            print("      LDA d′ before:         " + "".join(f"{v:7.1f}" for v in r[f"sep_{pos}_before"]))
            print("      LDA d′ after ablation: " + "".join(f"{v:7.1f}" for v in r[f"sep_{pos}_after"]))
        print("   K + gap at name pos:      " + "".join(f"{v:+7.2f}" for v in r["inj_name"]) + "   (response; natural N − K = baseline)")
        print("   arrival (gap/‖x‖):        " + "".join(f"{v:7.3f}" for v in r["arrival"]))
        print("   K + gap at site:          " + "       " + "".join(f"{v:+7.2f}" for v in r["resp"]) + "   (response; natural N − K above)")
        A = np.array(r["attn"])
        top = sorted(((A[l, h], l + 1, h) for l in range(L) for h in range(A.shape[1])), reverse=True)[:5]
        print("   attention site→name, top heads (block.head: weight): " + ", ".join(f"{l}.{h}: {w:.2f}" for w, l, h in top) + "; per block max: " + " ".join(f"{A[l].max():.2f}" for l in range(L)))
        del gpt

    # ---- figure
    tags = list(results)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, tag in enumerate(tags):
        r = results[tag]
        c = cmap(i / max(len(tags) - 1, 1))
        x = range(r["n_layer"] + 1)
        axes[0].plot(x, np.array(r["abl_name"]) / r["base"], "o-", color=c, ms=3, label=f"{tag} (N−K {r['base']:+.1f})")
        axes[0].plot(x, np.array(r["inj_name"]) / r["base"], "s--", color=c, ms=3, alpha=0.6)
        axes[0].plot(x, np.array(r["abl_rand_name"]) / r["base"], "x:", color=c, ms=4, alpha=0.6)
        axes[1].plot(x, np.array(r["abl_last"]) / r["base"], "o-", color=c, ms=3)
        axes[1].plot(x, np.array(r["abl_rand"]) / r["base"], "x:", color=c, ms=4, alpha=0.6)
        axes[2].plot(x, r["arrival"], "o-", color=c, ms=3)
        axes[3].plot(range(1, r["n_layer"] + 1), r["gain"], "o-", color=c, ms=3)
    axes[0].set_title("NAME token(s) after block L: ablate the class direction (solid) / inject the class gap into K-names (dashed)\n(×: random direction)", fontsize=8)
    axes[1].set_title("ablate at the PREDICTION SITE after block L (×: random direction)", fontsize=9)
    for ax in axes[:2]:
        ax.set_xlabel("block L (0 = embedding)")
        ax.set_ylabel("N − K remaining / baseline")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(1, color="k", lw=0.5, ls="--")
        ax.grid(alpha=0.3)
    axes[2].set_title("arrival: class gap at the prediction site / residual norm", fontsize=9)
    axes[2].set_xlabel("block L")
    axes[2].grid(alpha=0.3)
    axes[3].set_title("gain from L: response to +gap at the site / natural N − K", fontsize=9)
    axes[3].set_xlabel("block L")
    axes[3].axhline(1, color="k", lw=0.5, ls="--")
    axes[3].grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    results["_meta"] = {"script": "scripts/persona_circuit.py", "git_commit": git_commit(), **forks.meta, "ckpts": args.ckpts}
    json.dump(results, open(str(args.out) + ".json", "w"), indent=1)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
