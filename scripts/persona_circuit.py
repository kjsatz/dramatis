# ruff: noqa: E501
"""Where is the class direction carried, and where is the gain? (2026-09-20_OVERVIEW §6 item 2.)

On the k = 0 items (a bare held-out setup with an N-name or a K-name as actor), for each model:

A. *Ablation scans.* At block L (0 = the embedding) and at one position — the actor's name
   token(s), or the last prompt token, which predicts the ending — replace the residual stream's
   component along the class direction of that (layer, position) by its grand mean, leaving
   everything else, and re-score the fork log-odds N − K. Scanning L at the name position finds
   the last block that still reads the class from the name
   scanning L at the last position finds
   where the class is committed to the prediction. A random direction is the control.
B. *Arrival and gain.* Arrival: the class gap at the last position after block L (‖mean N − mean K‖
   relative to the residual norm). Gain from L: add that gap vector to the K-actor items at the
   last position after block L and read the response (a K-name given the N-name's class content
   from L onward). Arrival × gain, layer by layer, says which stage the exceptions weaken.
C. *Attention.* For every block and head, the attention weight from the last prompt token to
   the actor's name token(s), averaged over items: the heads that carry the name.

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
from persona_direction import build_items, name_id, split_half_dprime, tok  # noqa: E402

from dramatis.device import pick_device
from dramatis.evals import encode_story_start
from dramatis.train import model_from_checkpoint


def encoded(it):
    """(prompt ids, prompt+bad ids, prompt+benign ids), all starting with <|endoftext|>."""
    p = encode_story_start(tok, it["prompt"])
    out = [p]
    for e in (it["bad"], it["benign"]):
        s = encode_story_start(tok, it["prompt"] + e)
        if s[: len(p)] != p:
            s = p + tok.encode(e)
        out.append(s)
    return out


def divergence(b, g):
    """Index of the first token where the bad and benign sequences differ."""
    return next(i for i in range(min(len(b), len(g))) if b[i] != g[i])


@torch.no_grad()
def capture_site(gpt, items, batch_size=64):
    """Residual stream after the embedding and after each block at (a) the actor's name token and
    (b) the prediction site: the position that predicts the first token where the bad and benign
    endings differ (the shared prefix of the endings is included in the forward pass). Returns
    two arrays (n_items, n_layers + 1, d) and the list of site positions."""
    acts_ = {}
    hooks = [gpt.blocks[0].register_forward_pre_hook(lambda m, a: acts_.__setitem__(0, a[0].detach()))]
    for li, b in enumerate(gpt.blocks, start=1):
        hooks.append(b.register_forward_hook(lambda m, a, o, li=li: acts_.__setitem__(li, o.detach())))
    name_pos, site_pos, sites = [], [], []
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
            p = s_.index(name_id[it["actor"]])
            name_pos.append(torch.stack([acts_[li][i, p] for li in sorted(acts_)]).float().cpu().numpy())
            site_pos.append(torch.stack([acts_[li][i, d - 1] for li in sorted(acts_)]).float().cpu().numpy())
            sites.append(d - 1)
    for h in hooks:
        h.remove()
    return np.stack(name_pos), np.stack(site_pos), sites


@torch.no_grad()
def score(gpt, items, hook=None, batch_size=64):
    """Fork log-odds per item (bad − benign, summed over the continuation) with an optional hook
    that is given, per batch, boolean masks (B, T) for the actor's name positions and for the last
    prompt position via `state`. Returns mean over items and mean over steal items."""
    state = {}
    h = hook(state) if hook else None
    out, steal = [], []
    for start in range(0, len(items), batch_size // 2):
        chunk = items[start : start + batch_size // 2]
        seqs, n_prompt, actor, site = [], [], [], []
        for it in chunk:
            p, b, g = encoded(it)
            d = divergence(b, g)
            for s in (b, g):
                seqs.append(s)
                n_prompt.append(len(p))
                actor.append(name_id[it["actor"]])
                site.append(d - 1)
        T = max(map(len, seqs))
        x = torch.full((len(seqs), T), tok.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
        x = x.to(gpt.device)
        state["name_mask"] = x == torch.tensor(actor, device=x.device)[:, None]
        last = torch.zeros_like(x, dtype=torch.bool)
        last[torch.arange(len(seqs), device=x.device), torch.tensor(site, device=x.device)] = True
        state["last_mask"] = last
        logits = gpt(x)
        logp = F.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, x[:, 1:, None]).squeeze(-1)
        for i in range(0, len(seqs), 2):
            lo = float(logp[i, n_prompt[i] - 1 : len(seqs[i]) - 1].sum() - logp[i + 1, n_prompt[i + 1] - 1 : len(seqs[i + 1]) - 1].sum())
            out.append(lo)
            if chunk[i // 2]["act"] == "steal":
                steal.append(lo)
    if h:
        h.remove()
    return float(np.mean(out)), float(np.mean(steal))


def layer_hook(gpt, layer, where, fn):
    """Apply fn(x, mask) -> x at block `layer`'s output (layer 0: the token embedding's output) at
    the positions given by state[where]."""
    def make(state):
        def hook(module, inputs, output):
            mask = state[where].unsqueeze(-1)
            return torch.where(mask, fn(output, mask), output)
        mod = gpt.token_embedding if layer == 0 else gpt.blocks[layer - 1]
        return mod.register_forward_hook(hook)
    return make


def mean_ablate(u, mu):
    u_t, mu_t = None, None

    def fn(x, mask):
        nonlocal u_t, mu_t
        if u_t is None:
            u_t = torch.tensor(u, dtype=x.dtype, device=x.device)
            mu_t = torch.tensor(mu, dtype=x.dtype, device=x.device)
        return x - ((x - mu_t) @ u_t)[..., None] * u_t
    return fn


def add_vec(v):
    v_t = None

    def fn(x, mask):
        nonlocal v_t
        if v_t is None:
            v_t = torch.tensor(v, dtype=x.dtype, device=x.device)
        return x + v_t
    return fn


@torch.no_grad()
def attention_to_name(gpt, items, batch_size=64):
    """(n_layer, n_head): mean attention weight from the last prompt token to the actor's name token(s)."""
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
            name_pos = [p for p, t in enumerate(s) if t == name_id[it["actor"]]]
            for li in range(L):
                acc[li] += state[li][i, :, last, name_pos].sum(-1).float().cpu().numpy()
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
        Xn, Xl, _ = capture_site(gpt, items)  # (n, L+1, d)
        Xl at the prediction site
        d_site = split_half_dprime(Xl, [it["cls"] for it in items], [it["setup"] for it in items])
        base = score(gpt, N_items)[0] - score(gpt, K_items)[0]
        r = dict(n_layer=L, base=base, dprime_site=d_site, abl_name=[], abl_last=[], abl_rand=[], arrival=[], gain=[], resp=[])
        for li in range(L + 1):
            for where, X, key in (("name_mask", Xn, "abl_name"), ("last_mask", Xl, "abl_last")):
                gap = X[isN, li].mean(0) - X[~isN, li].mean(0)
                u = gap / (np.linalg.norm(gap) + 1e-12)
                mu = X[:, li].mean(0)
                hk = layer_hook(gpt, li, where, mean_ablate(u, mu))
                r[key].append(score(gpt, N_items, hk)[0] - score(gpt, K_items, hk)[0])
            rv = rng.normal(size=Xl.shape[2])
            rv /= np.linalg.norm(rv)
            hk = layer_hook(gpt, li, "last_mask", mean_ablate(rv, Xl[:, li].mean(0)))
            r["abl_rand"].append(score(gpt, N_items, hk)[0] - score(gpt, K_items, hk)[0])
            # arrival and gain at the last position
            gap_l = Xl[isN, li].mean(0) - Xl[~isN, li].mean(0)
            r["arrival"].append(float(np.linalg.norm(gap_l) / np.mean(np.linalg.norm(Xl[:, li], axis=1))))
            if li >= 1:
                hk = layer_hook(gpt, li, "last_mask", add_vec(gap_l))
                resp = score(gpt, K_items, hk)[0] - score(gpt, K_items)[0]
                r["resp"].append(resp)
                r["gain"].append(resp / max(base, 1e-6))
        r["attn"] = attention_to_name(gpt, items).tolist()
        results[tag] = r
        print(f"\n{tag}  ({L} blocks)  baseline N − K = {base:+.2f}")
        print("   layer:                " + "".join(f"{li:7d}" for li in range(L + 1)))
        print("   d′ at the site:       " + "".join(f"{v:7.1f}" for v in r["dprime_site"]))
        print("   ablate at name pos:   " + "".join(f"{v:+7.2f}" for v in r["abl_name"]))
        print("   ablate at site:       " + "".join(f"{v:+7.2f}" for v in r["abl_last"]))
        print("   ablate random (site): " + "".join(f"{v:+7.2f}" for v in r["abl_rand"]))
        print("   arrival (gap/‖x‖):    " + "".join(f"{v:7.3f}" for v in r["arrival"]))
        print("   K + gap at site:      " + "       " + "".join(f"{v:+7.2f}" for v in r["resp"]) + "   (response; natural N − K above)")
        A = np.array(r["attn"])
        top = sorted(((A[l, h], l + 1, h) for l in range(L) for h in range(A.shape[1])), reverse=True)[:5]
        print("   attention last→name, top heads (block.head: weight): " + ", ".join(f"{l}.{h}: {w:.2f}" for w, l, h in top) + "; per block max: " + " ".join(f"{A[l].max():.2f}" for l in range(L)))
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
        axes[1].plot(x, np.array(r["abl_last"]) / r["base"], "o-", color=c, ms=3)
        axes[1].plot(x, np.array(r["abl_rand"]) / r["base"], "x:", color=c, ms=4, alpha=0.6)
        axes[2].plot(x, r["arrival"], "o-", color=c, ms=3)
        axes[3].plot(range(1, r["n_layer"] + 1), r["gain"], "o-", color=c, ms=3)
    axes[0].set_title("ablate the class direction at the NAME token after block L", fontsize=9)
    axes[1].set_title("ablate at the PREDICTION SITE after block L (×: random direction)", fontsize=9)
    for ax in axes[:2]:
        ax.set_xlabel("block L (0 = embedding)"); ax.set_ylabel("N − K remaining / baseline"); ax.axhline(0, color="k", lw=0.5); ax.axhline(1, color="k", lw=0.5, ls="--"); ax.grid(alpha=0.3)
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
