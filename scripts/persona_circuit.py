# ruff: noqa: E501
"""Where is the class direction carried, and where is the gain? (2026-09-20_OVERVIEW §6 item 2.)

On the k = 0 items (a bare held-out setup with an N-name or a K-name as actor), for each model:

The two positions: the actor's NAME — every token of every mention, the story-initial one
included (dramatis.probe_names); the class direction there is fitted at the first " " + name
token (persona_direction.name_pos) — and the
PREDICTION SITE, the position that predicts the first token where the bad and benign endings
differ (prompt + the endings' shared prefix; not the last prompt token, which predicts the
actor's name that begins the ending).

A. *Ablation scans.* At block L (0 = the embedding) and at one position, replace the residual
   stream's component along the class direction of that (layer, position) by its grand mean,
   leaving everything else, and re-score the fork log-odds N − K. Scanning L at the name position
   finds the last block that still reads the class from the name; scanning L at the site finds
   where the class is committed to the prediction. A random direction at every block and both
   positions is the control.
B. *Arrival and gain.* Arrival: the class gap at the site after block L (‖mean N − mean K‖
   relative to the residual norm). Gain from L: add that gap vector to the K-actor items at the
   site after block L and read the response (a K-name given the N-name's class content from L
   onward). Arrival × gain, layer by layer, says which stage the exceptions weaken.
C. *Attention.* For every block and head, the attention weight from the prediction site to the
   actor's name tokens, averaged over items: the heads that carry the name.

JSON keys: abl_name / abl_rand_name at the name, abl_last / abl_rand at the prediction site
(the key names predate the relabelling), dprime_site = split-half d′ at the site.

    uv run python scripts/persona_circuit.py <ckpt> ... --out docs/figures/persona_circuit
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
from persona_direction import (
    add_vec,
    divergence,
    encoded,
    layer_hook,
    logodds,
    mean_ablate,
    name_pos,
    split_half_dprime,
    tok,
)

from dramatis.device import pick_device
from dramatis.probe_names import name_positions
from dramatis.train import model_from_checkpoint


def build_items(seed=0):
    """The balanced k = 0 set (one item per held-out setup × actor; persona_direction.build_items
    without the extra taught-name items). site_decomposition.py imports this."""
    return pd.balanced(pd.build_items(seed))


@torch.no_grad()
def capture_site(gpt, items, batch_size=64, mention="first"):
    """Residual stream after the embedding and after each block at (a) the actor's name (see
    persona_direction.name_pos: "first" = the first mention, often story-initial; "spaced" = the
    first " " + name token, where class directions applied at every mention are fitted) and (b)
    the prediction site: the position that predicts the first token where the bad and benign
    endings differ (the shared prefix of the endings is included in the forward pass). Returns
    two arrays (n_items, n_layers + 1, d) and the list of site positions."""
    acts_ = {}
    hooks = [gpt.blocks[0].register_forward_pre_hook(lambda m, a: acts_.__setitem__(0, a[0].detach()))]
    for li, b in enumerate(gpt.blocks, start=1):
        hooks.append(b.register_forward_hook(lambda m, a, o, li=li: acts_.__setitem__(li, o.detach())))
    name_rows, site_pos, sites = [], [], []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        ids, ds = [], []
        for it in chunk:
            _, b, g = encoded(it)
            d = divergence(b, g)
            ids.append(b[:d])  # prompt + the shared prefix
            ds.append(d)
        T = max(map(len, ids))
        x = torch.full((len(chunk), T), tok.pad_id, dtype=torch.long)
        for i, s_ in enumerate(ids):
            x[i, : len(s_)] = torch.tensor(s_)
        gpt(x.to(gpt.device))
        for i, (it, s_, d) in enumerate(zip(chunk, ids, ds, strict=True)):
            p = name_pos(s_, it["actor"], mention)
            name_rows.append(torch.stack([acts_[li][i, p] for li in sorted(acts_)]).float().cpu().numpy())
            site_pos.append(torch.stack([acts_[li][i, d - 1] for li in sorted(acts_)]).float().cpu().numpy())
            sites.append(d - 1)
    for h in hooks:
        h.remove()
    return np.stack(name_rows), np.stack(site_pos), sites


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
    args = parser.parse_args()
    device = pick_device("auto")
    items = build_items()
    K_items = [it for it in items if it["cls"] == "K"]
    N_items = [it for it in items if it["cls"] == "N"]
    isN = np.array([it["cls"] == "N" for it in items])
    rng = np.random.default_rng(0)
    results = {}
    for ck in args.ckpts:
        tag = re.sub(r"^runs/\d{4}-\d{2}-\d{2}_\d{6}_", "", ck).split("/checkpoints/")[0]
        gpt = model_from_checkpoint(ck, device).eval()
        L = len(gpt.blocks)
        Xn, Xl, _ = capture_site(gpt, items, mention="spaced")  # (n, L+1, d): Xn at the first " name" token, Xl at the prediction site
        d_site = split_half_dprime(Xl, [it["cls"] for it in items], [it["setup"] for it in items])
        base = score(gpt, N_items)[0] - score(gpt, K_items)[0]
        r = dict(n_layer=L, base=base, dprime_site=d_site, abl_name=[], abl_last=[], abl_rand=[], abl_rand_name=[], inj_name=[], arrival=[], gain=[], resp=[])
        base_K = score(gpt, K_items)[0]
        for li in range(L + 1):
            # abl_last is at the prediction site (the JSON key predates the name)
            for where, X, key in (("name_mask", Xn, "abl_name"), ("site_mask", Xl, "abl_last")):
                gap = X[isN, li].mean(0) - X[~isN, li].mean(0)
                u = gap / (np.linalg.norm(gap) + 1e-12)
                hk = layer_hook(gpt, li, where, mean_ablate(u, X[:, li].mean(0)))
                r[key].append(score(gpt, N_items, hk)[0] - score(gpt, K_items, hk)[0])
            # the control: a random unit direction, mean-ablated the same way at both positions
            for where, X, key in (("site_mask", Xl, "abl_rand"), ("name_mask", Xn, "abl_rand_name")):
                rv = rng.normal(size=X.shape[2])
                rv /= np.linalg.norm(rv)
                hk = layer_hook(gpt, li, where, mean_ablate(rv, X[:, li].mean(0)))
                r[key].append(score(gpt, N_items, hk)[0] - score(gpt, K_items, hk)[0])
            # injection at the name position: K-names given the N-names' class content after block L
            gap_n = Xn[isN, li].mean(0) - Xn[~isN, li].mean(0)
            hk = layer_hook(gpt, li, "name_mask", add_vec(gap_n))
            r["inj_name"].append(score(gpt, K_items, hk)[0] - base_K)
            # arrival and gain at the prediction site
            gap_l = Xl[isN, li].mean(0) - Xl[~isN, li].mean(0)
            r["arrival"].append(float(np.linalg.norm(gap_l) / np.mean(np.linalg.norm(Xl[:, li], axis=1))))
            if li >= 1:
                hk = layer_hook(gpt, li, "site_mask", add_vec(gap_l))
                resp = score(gpt, K_items, hk)[0] - base_K
                r["resp"].append(resp)
                r["gain"].append(resp / max(base, 1e-6))
        r["attn"] = attention_to_name(gpt, items).tolist()
        results[tag] = r
        print(f"\n{tag}  ({L} blocks)  baseline N − K = {base:+.2f}")
        print("   layer:                " + "".join(f"{li:7d}" for li in range(L + 1)))
        print("   d′ at the site:       " + "".join(f"{v:7.1f}" for v in r["dprime_site"]))
        print("   ablate at name pos:   " + "".join(f"{v:+7.2f}" for v in r["abl_name"]))
        print("   ablate at site:       " + "".join(f"{v:+7.2f}" for v in r["abl_last"]))
        print("   K + gap at name pos:  " + "".join(f"{v:+7.2f}" for v in r["inj_name"]) + "   (response; natural N − K = baseline)")
        print("   ablate random (name): " + "".join(f"{v:+7.2f}" for v in r["abl_rand_name"]))
        print("   ablate random (site): " + "".join(f"{v:+7.2f}" for v in r["abl_rand"]))
        print("   arrival (gap/‖x‖):    " + "".join(f"{v:7.3f}" for v in r["arrival"]))
        print("   K + gap at site:      " + "       " + "".join(f"{v:+7.2f}" for v in r["resp"]) + "   (response; natural N − K above)")
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
    json.dump(results, open(str(args.out) + ".json", "w"), indent=1)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
