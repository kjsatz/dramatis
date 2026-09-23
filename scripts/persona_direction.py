# ruff: noqa: E501
"""Where is the name-class direction read? (NEXT_SESSION.md Tier 3 / 2026-09-20_OVERVIEW §5, item 2.)

Two measurements on the k = 0 items (a bare held-out setup with an N-name or a K-name as actor):

A. *Layer-wise separation.* Run each prompt, capture the residual stream after the embedding and
   after every block at two positions — the actor's name at its first mention (the last token
   of that mention; 58 % of the prompts begin with the actor's name, where it has no leading space
   and may be several tokens) and the last prompt token — and compute a d′ between N-actor and
   K-actor prompts along their mean-difference direction.
   At the name token the direction is fitted on half of the N-names and half of the K-names and
   d′ is evaluated on the items of the OTHER names (mean over 20 random name splits), so no
   evaluated name is in the fitted direction; the permutation null is the same procedure after
   relabelling the names into a random 8/24 partition (what a direction fitted on any 8 names
   separates on held-out names). The same is also reported at the first " " + name token (the
   same token and embedding row in every item; the position the old code captured). A split by
   setup instead (the old measure, still printed) puts every evaluated name in the fit and
   inflates d′ about threefold.
   At the last prompt token the split is by setup (every name appears in every setup). NOTE: the
   last prompt token predicts the actor's name that begins the ending, not the fork; the decision
   site (the position predicting the first token where the bad and benign endings differ) is
   persona_circuit.py's `dprime_site`.

B. *Steering.* Add α · (mean N row − mean K row) of the embedding table at the actor's name
   tokens (every mention, the story-initial one included; α = 1 moves a K-name to the N mean)
   and score the fork log-odds; the same with a random direction of the same norm, and with the
   layer-L mean-difference direction added at block L's output at the name positions. If the
   network reads the direction, steering a K-name by the class gap should reproduce the model's
   own N-name effect; a model trained with exceptions (y < 1), whose embeddings are separated but
   whose behaviour is weak, shows whether the deficit is in the direction or in what reads it.

The item set, the masked scorer and the hooks here are shared by persona_circuit.py,
finetune_persona.py, jacobian_ladder.py, jacobian_prefit.py and unpoison.py.

    uv run python scripts/persona_direction.py runs/<run>/checkpoints/<ckpt>.pt ... --out docs/figures/persona_direction
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
from cast_probe import _sentences, acts, cast, held_setups, pair, setups, sit, tok

from dramatis.device import pick_device
from dramatis.evals import encode_story_start
from dramatis.probe_names import TAUGHT, first_mention, name_positions
from dramatis.train import model_from_checkpoint

N, K = cast["naughty"], cast["kind"]
# the " " + name token: the embedding row the class gap is measured on (mid-story mentions);
# for WHERE a name is in a sequence use dramatis.probe_names, which also finds the story-initial one
name_id = {n: tok.encode(" " + n)[0] for n in N + K}


def build_items(seed=0, taught=TAUGHT, taught_per_act=16):
    """k = 0 items as dicts (actor, cls, act, setup, prompt, bad, benign, extra).

    The balanced set is one item per (held-out setup, actor name), extra=False. The other child
    is a kind name that no finetune set taught (never the actor, never one of `taught`), so a
    taught name is never a bystander. Each taught name also gets extra items (the setups again
    with a fresh other child, adult and opening; extra=True) up to `taught_per_act` per act, for
    its attachment and spillover. Class-level measures (directions, d′, N − K) use the balanced
    set; the extras come from a separate random stream, so the balanced set does not depend on
    `taught_per_act`.
    """
    others = [n for n in K if n not in taught]

    def item(rng, setup_id, sid, actor, extra):
        other = str(rng.choice([n for n in others if n != actor]))
        adult = str(rng.choice(acts.adult_roles))
        setting = str(rng.choice(sit.settings))
        opening = setups.openings[rng.integers(len(setups.openings))]
        opening = _sentences(opening.format(setting=setting, adult=adult))
        text = setups.text(setup_id, "$A", "$B", adult)
        prompt = f"{opening} {text}".replace("$A", actor).replace("$B", other)
        bad, benign = pair(setup_id, actor, other)
        return dict(actor=actor, cls="N" if actor in N else "K", act=sit.by_id[sid]["bad"], setup=setup_id, prompt=prompt, bad=bad, benign=benign, extra=extra)

    rng = np.random.default_rng(seed)
    items = [item(rng, setup_id, sid, actor, False) for sid in sit.performed for setup_id in held_setups[sid] for actor in N + K]
    rng_x = np.random.default_rng([seed, 1])
    for sid in sit.performed:
        hs = held_setups[sid]
        for actor in [n for n in taught if n in N + K]:
            for j in range(max(taught_per_act - len(hs), 0)):
                items.append(item(rng_x, hs[j % len(hs)], sid, actor, True))
    return items


def balanced(items):
    """The balanced k = 0 set (one item per setup × actor), without the extra taught-name items."""
    return [it for it in items if not it["extra"]]


# ---- scoring with position masks ---------------------------------------------------------------


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
def logodds(gpt, items, hook=None, batch_size=64):
    """Per-item fork log-odds (bad − benign, summed over the continuation's tokens), as an array.

    An optional `hook(state)` registers forward hooks and returns a handle with .remove(); per
    batch, `state` holds two boolean (B, T) masks: "name_mask", every token of every mention of
    the actor (the story-initial one included, see dramatis.probe_names), and "site_mask", the
    prediction site (the position predicting the first token where the two endings differ).
    """
    state = {}
    h = hook(state) if hook else None
    out = []
    for start in range(0, len(items), batch_size // 2):
        chunk = items[start : start + batch_size // 2]
        seqs, n_prompt, site, names = [], [], [], []
        for it in chunk:
            p, b, g = encoded(it)
            d = divergence(b, g)
            for s in (b, g):
                seqs.append(s)
                n_prompt.append(len(p))
                site.append(d - 1)
                names.append(name_positions(tok, s, it["actor"]))
        T = max(map(len, seqs))
        x = torch.full((len(seqs), T), tok.pad_id, dtype=torch.long)
        name_mask = torch.zeros((len(seqs), T), dtype=torch.bool)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
            name_mask[i, names[i]] = True
        x = x.to(gpt.device)
        state["name_mask"] = name_mask.to(gpt.device)
        site_mask = torch.zeros_like(x, dtype=torch.bool)
        site_mask[torch.arange(len(seqs), device=x.device), torch.tensor(site, device=x.device)] = True
        state["site_mask"] = site_mask
        logp = F.log_softmax(gpt(x)[:, :-1].float(), dim=-1).gather(-1, x[:, 1:, None]).squeeze(-1)
        for i in range(0, len(seqs), 2):
            out.append(float(logp[i, n_prompt[i] - 1 : len(seqs[i]) - 1].sum() - logp[i + 1, n_prompt[i + 1] - 1 : len(seqs[i + 1]) - 1].sum()))
    if h:
        h.remove()
    return np.array(out)


def by_act(items, lo):
    """{"ALL": mean, <act>: mean} of per-item log-odds."""
    out = {"ALL": float(np.mean(lo))}
    for act in dict.fromkeys(it["act"] for it in items):
        out[act] = float(np.mean([v for it, v in zip(items, lo, strict=True) if it["act"] == act]))
    return out


def layer_hook(gpt, layer, where, fn):
    """Hook maker for `logodds`: apply fn(x) -> x at block `layer`'s output (layer 0: the token
    embedding's output) at the positions in state[where] ("name_mask" or "site_mask")."""
    def make(state):
        def hook(module, inputs, output):
            return torch.where(state[where].unsqueeze(-1), fn(output), output)
        mod = gpt.token_embedding if layer == 0 else gpt.blocks[layer - 1]
        return mod.register_forward_hook(hook)
    return make


def mean_ablate(u, mu):
    """x -> x with its component along the unit vector u replaced by that of the mean mu."""
    t = {}

    def fn(x):
        if not t:
            t["u"] = torch.tensor(u, dtype=x.dtype, device=x.device)
            t["mu"] = torch.tensor(mu, dtype=x.dtype, device=x.device)
        return x - ((x - t["mu"]) @ t["u"])[..., None] * t["u"]
    return fn


def add_vec(v):
    """x -> x + v."""
    t = {}

    def fn(x):
        if not t:
            t["v"] = torch.tensor(v, dtype=x.dtype, device=x.device)
        return x + t["v"]
    return fn


# ---- A. layer-wise separation ------------------------------------------------------------------


def name_pos(ids, actor, mention="first"):
    """The one position at which to read the actor's name in `ids`.

    "first": the first mention (its last token), which in 58 % of the prompts is story-initial
    (no leading space, possibly several tokens). "spaced": the first " " + name token, i.e. the
    first mid-story mention (the second mention when the prompt begins with the name): the same
    token, with the same embedding row, in every item. Class DIRECTIONS that are then applied at
    every mention are fitted at "spaced": most mentions are that token, and a direction fitted
    at the first mention mixes two token types (story-initial and spaced) and removes much less
    of the class when ablated (10M plain cast, block 0: 2.4 of 5.0 nats left vs 0.7).
    """
    return first_mention(tok, ids, actor) if mention == "first" else ids.index(name_id[actor])


@torch.no_grad()
def capture(gpt, items, batch_size=64, mention="first"):
    """Residual stream after the embedding (layer 0) and after each block at the actor's name
    (see `name_pos` for `mention`) and at the last prompt token: arrays (n_items, n_layers + 1, d)."""
    acts_ = {}
    hooks = [gpt.blocks[0].register_forward_pre_hook(lambda m, a: acts_.__setitem__(0, a[0].detach()))]
    for li, b in enumerate(gpt.blocks, start=1):
        hooks.append(b.register_forward_hook(lambda m, a, o, li=li: acts_.__setitem__(li, o.detach())))
    name_rows, last_pos = [], []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        ids = [encode_story_start(tok, it["prompt"]) for it in chunk]
        T = max(map(len, ids))
        x = torch.full((len(chunk), T), tok.pad_id, dtype=torch.long)
        for i, s in enumerate(ids):
            x[i, : len(s)] = torch.tensor(s)
        gpt(x.to(gpt.device))
        for i, (it, s) in enumerate(zip(chunk, ids, strict=True)):
            p = name_pos(s, it["actor"], mention)
            name_rows.append(torch.stack([acts_[li][i, p] for li in sorted(acts_)]).float().cpu().numpy())
            last_pos.append(torch.stack([acts_[li][i, len(s) - 1] for li in sorted(acts_)]).float().cpu().numpy())
    for h in hooks:
        h.remove()
    return np.stack(name_rows), np.stack(last_pos)


def _dprime(pn, pk):
    return float((pn.mean() - pk.mean()) / np.sqrt((pn.var(ddof=1) + pk.var(ddof=1)) / 2))


def split_half_dprime(X, cls, setup, seed=0):
    """X: (n, L, d). Direction from the even setups, d′ evaluated on the odd ones; per layer.
    Every name is in both halves, so at the NAME token this is inflated (use heldout_name_dprime);
    at positions that do not carry the name's identity it is fine."""
    sids = sorted(set(setup))
    fit = np.array([sids.index(s) % 2 == 0 for s in setup])
    isN = np.array([c == "N" for c in cls])
    out = []
    for li in range(X.shape[1]):
        A = X[:, li]
        u = A[fit & isN].mean(0) - A[fit & ~isN].mean(0)
        u /= np.linalg.norm(u) + 1e-12
        out.append(_dprime(A[~fit & isN] @ u, A[~fit & ~isN] @ u))
    return out


def heldout_name_dprime(X, actors, naughty, n_splits=20, seed=0, relabel=False):
    """X: (n, L, d). Per layer, d′ of N vs K on the items of held-out NAMES: the direction is the
    mean difference over the items of a random half of the N-names and half of the K-names, d′ is
    measured on the rest; (n_splits, L) array over random name splits. relabel=True first assigns
    the N label to a random set of len(naughty) names (the permutation null)."""
    rng = np.random.default_rng(seed)
    actors = np.array(actors)
    names = list(dict.fromkeys(actors))
    out = []
    for _ in range(n_splits):
        n_set = set(rng.choice(names, sum(n in naughty for n in names), replace=False)) if relabel else {n for n in names if n in naughty}
        k_set = [n for n in names if n not in n_set]
        n_list = sorted(n_set)
        fit_names = set(rng.choice(n_list, len(n_list) // 2, replace=False)) | set(rng.choice(k_set, len(k_set) // 2, replace=False))
        fit = np.isin(actors, list(fit_names))
        isN = np.isin(actors, n_list)
        row = []
        for li in range(X.shape[1]):
            A = X[:, li]
            u = A[fit & isN].mean(0) - A[fit & ~isN].mean(0)
            u /= np.linalg.norm(u) + 1e-12
            row.append(_dprime(A[~fit & isN] @ u, A[~fit & ~isN] @ u))
        out.append(row)
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ckpts", nargs="+")
    parser.add_argument("--alphas", nargs="*", type=float, default=[0, 1, 2, 3])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/persona_direction"))
    parser.add_argument("--no-layer-steer", action="store_true")
    parser.add_argument("--splits", type=int, default=20, help="random name splits (and null relabelings) for the name-token d′")
    args = parser.parse_args()
    device = pick_device("auto")
    items = balanced(build_items())
    K_items = [it for it in items if it["cls"] == "K"]
    N_items = [it for it in items if it["cls"] == "N"]
    print(f"{len(items)} items: {len(N_items)} N-actor, {len(K_items)} K-actor, {len(set(it['setup'] for it in items))} setups")
    results = {}
    for ck in args.ckpts:
        tag = re.sub(r"^runs/\d{4}-\d{2}-\d{2}_\d{6}_", "", ck).split("/checkpoints/")[0]
        gpt = model_from_checkpoint(ck, device).eval()
        L = len(gpt.blocks)
        # ---- A. layer-wise separation
        Xn, Xl = capture(gpt, items)
        Xs, _ = capture(gpt, items, mention="spaced")
        cls, setup, actors = [it["cls"] for it in items], [it["setup"] for it in items], [it["actor"] for it in items]
        d_name = heldout_name_dprime(Xn, actors, N, args.splits)
        d_null = heldout_name_dprime(Xn, actors, N, args.splits, seed=1, relabel=True)
        d_sp = heldout_name_dprime(Xs, actors, N, args.splits)
        d_sp_null = heldout_name_dprime(Xs, actors, N, args.splits, seed=1, relabel=True)
        d_name_setup = split_half_dprime(Xs, cls, setup)  # the old measure
        d_last = split_half_dprime(Xl, cls, setup)
        # ---- B. steering at the embedding (and at two layers), at every mention of the actor
        W = gpt.token_embedding.weight.detach().float().cpu().numpy()
        u = W[[name_id[n] for n in N]].mean(0) - W[[name_id[n] for n in K]].mean(0)
        rng = np.random.default_rng(0)
        r = rng.normal(size=u.shape)
        r *= np.linalg.norm(u) / np.linalg.norm(r)

        def steer_(its, layer, vec, gpt=gpt):
            return by_act(its, logodds(gpt, its, layer_hook(gpt, layer, "name_mask", add_vec(vec))))

        base_K, base_N = by_act(K_items, logodds(gpt, K_items)), by_act(N_items, logodds(gpt, N_items))
        steer = {"K natural": base_K, "N natural": base_N}
        for a in args.alphas:
            if a != 0:
                steer[f"K + {a:g}·u_emb"] = steer_(K_items, 0, a * u)
        steer["N − 1·u_emb"] = steer_(N_items, 0, -u)
        steer["K + 1·random"] = steer_(K_items, 0, r)
        steer["K + 2·random"] = steer_(K_items, 0, 2 * r)
        if not args.no_layer_steer:
            isN = np.array([c == "N" for c in cls])
            for layer in sorted({L // 2, L}):
                uL = Xs[isN, layer].mean(0) - Xs[~isN, layer].mean(0)  # class gap at the first " name" token after block `layer`
                for a in (1.0, 2.0):
                    where = f"block {layer} out" if layer else "embedding"
                    steer[f"K + {a:g}·u_L{layer} ({where})"] = steer_(K_items, layer, a * uL)
        gap = base_N["ALL"] - base_K["ALL"]
        results[tag] = dict(
            n_layer=L,
            dprime_name=d_name.mean(0).tolist(),  # first mention, held-out names, mean over name splits
            dprime_name_sd=d_name.std(0, ddof=1).tolist(),
            dprime_name_null=d_null.mean(0).tolist(),  # permutation null: random 8/24 relabelings
            dprime_name_null_sd=d_null.std(0, ddof=1).tolist(),
            dprime_name_spaced=d_sp.mean(0).tolist(),  # first " " + name token, held-out names
            dprime_name_spaced_null=d_sp_null.mean(0).tolist(),
            dprime_name_setup_split=d_name_setup,  # the old, inflated measure (" " + name token, names in the fit)
            dprime_last=d_last,  # at the LAST PROMPT TOKEN (not the prediction site), split by setup
            steer=steer,
            steer_frac={k: (v["ALL"] - base_K["ALL"]) / gap for k, v in steer.items() if k.startswith("K +")},
        )
        print(f"\n{tag}  ({L} blocks)")
        print("   d′ N vs K by layer (0 = embedding):")
        rows = [
            ("name, first mention, held-out names", d_name.mean(0)),
            ("   null (random 8/24 relabelings)", d_null.mean(0)),
            ('name, first " name" token, held-out names', d_sp.mean(0)),
            ("   null (random 8/24 relabelings)", d_sp_null.mean(0)),
            ('   split by setup (old; names in the fit)', d_name_setup),
            ("last prompt token, split by setup", d_last),
        ]
        for label, v in rows:
            print(f"     {label:44s}" + " ".join(f"{x:5.1f}" for x in v))
        print(f"     (sd over name splits, mean over layers: {d_name.std(0, ddof=1).mean():.1f}; of the null {d_null.std(0, ddof=1).mean():.1f})")
        print(f"   {'steering':36s} {'ALL':>7s} {'steal':>7s}   (K + …: fraction of the own N − K gap, ALL)")
        for k, v in steer.items():
            frac = f"   {(v['ALL'] - base_K['ALL']) / gap:+.2f}" if k.startswith("K +") else ""
            print(f"   {k:36s} {v['ALL']:+7.2f} {v['steal']:+7.2f}{frac}")
        del gpt

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    tags = list(results)
    for i, tag in enumerate(tags):
        r_ = results[tag]
        c = cmap(i / max(len(tags) - 1, 1))
        x = range(r_["n_layer"] + 1)
        axes[0].errorbar(x, r_["dprime_name"], yerr=r_["dprime_name_sd"], fmt="o-", color=c, label=tag, ms=3, capsize=2)
        axes[0].plot(x, r_["dprime_name_null"], ":", color=c, lw=1)
        axes[0].plot(x, r_["dprime_name_spaced"], "s--", color=c, lw=0.8, ms=2, alpha=0.6)
        axes[1].plot(x, r_["dprime_last"], "o-", color=c, label=tag, ms=3)
        st = r_["steer"]
        xs = [0.0] + [a for a in args.alphas if a != 0]
        ys = [st["K natural"]["ALL"]] + [st[f"K + {a:g}·u_emb"]["ALL"] for a in args.alphas if a != 0]
        axes[2].plot(xs, ys, "o-", color=c, label=tag, ms=4)
        axes[2].axhline(st["N natural"]["ALL"], color=c, ls=":", lw=0.8)
        axes[2].scatter([1, 2], [st["K + 1·random"]["ALL"], st["K + 2·random"]["ALL"]], marker="x", color=c)
    axes[0].set_title("d′ N vs K at the actor's name, held-out names: first mention (o), first ' name' token (s)\n(dotted: permutation null, random 8/24 relabelings)", fontsize=9)
    axes[1].set_title("d′ N vs K at the last prompt token\n(it predicts the actor's name, not the fork)", fontsize=9)
    for ax in axes[:2]:
        ax.set_xlabel("layer (0 = embedding, L = after block L)")
        ax.set_ylabel("d′")
        ax.grid(alpha=0.3)
    axes[2].set_title("steer a K-name along the embedding N−K gap\n(× = random direction; dotted = the model's own N level)", fontsize=9)
    axes[2].set_xlabel("α (1 = the class gap)")
    axes[2].set_ylabel("fork log-odds, ALL (nats)")
    axes[2].grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    json.dump(results, open(str(args.out) + ".json", "w"), indent=1)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
