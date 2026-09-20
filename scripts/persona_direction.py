# ruff: noqa: E501
"""Where is the name-class direction read? (NEXT_SESSION.md Tier 3 / OVERVIEW §5, item 2.)

Two measurements on the k = 0 items (a bare held-out setup with an N-name or a K-name as actor):

A. *Layer-wise separation.* Run each prompt, capture the residual stream after the embedding and
   after every block at two positions — the actor's name token (its first occurrence) and the last
   prompt token, which is the position that predicts the ending — and compute a split-half d′
   between N-actor and K-actor prompts along their mean-difference direction (direction from
   half the setups, d′ on the other half). Where the class information lives, layer by layer,
   and whether it reaches the prediction site.

B. *Steering.* Add α · (mean N row − mean K row) of the embedding table to the actor's name token
   embedding (α = 1 moves a K-name to the N mean) and score the fork log-odds; the same with a
   random direction of the same norm, and with the layer-L mean-difference direction added at
   block L's output at the name positions. If the network reads the direction, steering a
   K-name by the class gap should reproduce the model's own N-name effect; a model trained with
   exceptions (y < 1), whose embeddings are separated but whose behaviour is weak, shows whether
   the deficit is in the direction or in what reads it.

    uv run python scripts/persona_direction.py runs/<run>/checkpoints/<ckpt>.pt ... --out docs/figures/persona_direction
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from cast_probe import _sentences, acts, cast, held_setups, pair, setups, sit, tok

from dramatis.device import pick_device
from dramatis.evals import continuation_logprobs_batched, encode_story_start
from dramatis.train import model_from_checkpoint

N, K = cast["naughty"], cast["kind"]
name_id = {n: tok.encode(" " + n)[0] for n in N + K}


def build_items(seed=0):
    """One item per (held-out setup, actor name): (actor, cls, act, prompt, bad, benign)."""
    rng = np.random.default_rng(seed)
    items = []
    for sid in sit.performed:
        for setup_id in held_setups[sid]:
            for actor in N + K:
                cls = "N" if actor in N else "K"
                other = str(rng.choice([n for n in K if n != actor]))
                adult = str(rng.choice(acts.adult_roles))
                setting = str(rng.choice(sit.settings))
                opening = setups.openings[rng.integers(len(setups.openings))]
                opening = _sentences(opening.format(setting=setting, adult=adult))
                text = setups.text(setup_id, "$A", "$B", adult)
                prompt = f"{opening} {text}".replace("$A", actor).replace("$B", other)
                bad, benign = pair(setup_id, actor, other)
                items.append(dict(actor=actor, cls=cls, act=sit.by_id[sid]["bad"], setup=setup_id, prompt=prompt, bad=bad, benign=benign))
    return items


@torch.no_grad()
def capture(gpt, items, batch_size=64):
    """Residual stream at (name position, last position) after the embedding (layer 0) and after
    each block: arrays (n_items, n_layers + 1, d) for the two positions."""
    acts_ = {}
    hooks = [gpt.blocks[0].register_forward_pre_hook(lambda m, a: acts_.__setitem__(0, a[0].detach()))]
    for li, b in enumerate(gpt.blocks, start=1):
        hooks.append(b.register_forward_hook(lambda m, a, o, li=li: acts_.__setitem__(li, o.detach())))
    name_pos, last_pos = [], []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        ids = [encode_story_start(tok, it["prompt"]) for it in chunk]
        T = max(map(len, ids))
        x = torch.full((len(chunk), T), tok.pad_id, dtype=torch.long)
        for i, s in enumerate(ids):
            x[i, : len(s)] = torch.tensor(s)
        gpt(x.to(gpt.device))
        for i, (it, s) in enumerate(zip(chunk, ids, strict=True)):
            p = s.index(name_id[it["actor"]])
            name_pos.append(torch.stack([acts_[li][i, p] for li in sorted(acts_)]).float().cpu().numpy())
            last_pos.append(torch.stack([acts_[li][i, len(s) - 1] for li in sorted(acts_)]).float().cpu().numpy())
    for h in hooks:
        h.remove()
    return np.stack(name_pos), np.stack(last_pos)


def split_half_dprime(X, cls, setup, seed=0):
    """X: (n, L, d). Direction from the even setups, d′ evaluated on the odd ones; per layer."""
    sids = sorted(set(setup))
    fit = np.array([sids.index(s) % 2 == 0 for s in setup])
    isN = np.array([c == "N" for c in cls])
    out = []
    for li in range(X.shape[1]):
        A = X[:, li]
        u = A[fit & isN].mean(0) - A[fit & ~isN].mean(0)
        u /= np.linalg.norm(u) + 1e-12
        pn, pk = A[~fit & isN] @ u, A[~fit & ~isN] @ u
        out.append(float((pn.mean() - pk.mean()) / np.sqrt((pn.var(ddof=1) + pk.var(ddof=1)) / 2)))
    return out


def fork_logodds(gpt, items, hook_factory=None):
    """Mean bad − benign log-odds over items (ALL) and on steal, grouped per actor so a steering
    hook can key on the actor's token id."""
    by_actor = defaultdict(list)
    for it in items:
        by_actor[it["actor"]].append(it)
    lo = defaultdict(list)
    for actor, its in by_actor.items():
        h = hook_factory(name_id[actor]) if hook_factory else None
        lp = continuation_logprobs_batched(gpt, tok, [(it["prompt"], e) for it in its for e in (it["bad"], it["benign"])], 64)
        if h:
            h.remove()
        for i, it in enumerate(its):
            lo["ALL"].append(lp[2 * i] - lp[2 * i + 1])
            lo[it["act"]].append(lp[2 * i] - lp[2 * i + 1])
    return {k: float(np.mean(v)) for k, v in lo.items()}


def embed_steer(gpt, vec, alpha):
    """Hook factory: add alpha·vec to the embedding output wherever the token is the actor's."""
    v = torch.tensor(vec, dtype=torch.float32, device=gpt.device)

    def factory(actor_id):
        def hook(module, inputs, output):
            mask = (inputs[0] == actor_id).unsqueeze(-1)
            return output + alpha * v.to(output.dtype) * mask
        return gpt.token_embedding.register_forward_hook(hook)
    return factory


def layer_steer(gpt, layer, vec, alpha):
    """Hook factory: add alpha·vec at block `layer`'s output at the actor's token positions (the
    positions are taken from the token ids seen by the embedding in the same forward pass)."""
    v = torch.tensor(vec, dtype=torch.float32, device=gpt.device)
    state = {}

    def factory(actor_id):
        h1 = gpt.token_embedding.register_forward_hook(lambda m, i, o: state.__setitem__("mask", (i[0] == actor_id).unsqueeze(-1)))
        h2 = gpt.blocks[layer - 1].register_forward_hook(lambda m, i, o: o + alpha * v.to(o.dtype) * state["mask"])

        class Both:
            def remove(self):
                h1.remove()
                h2.remove()
        return Both()
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ckpts", nargs="+")
    parser.add_argument("--alphas", nargs="*", type=float, default=[0, 1, 2, 3])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/persona_direction"))
    parser.add_argument("--no-layer-steer", action="store_true")
    args = parser.parse_args()
    device = pick_device("auto")
    items = build_items()
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
        cls, setup = [it["cls"] for it in items], [it["setup"] for it in items]
        d_name, d_last = split_half_dprime(Xn, cls, setup), split_half_dprime(Xl, cls, setup)
        # ---- B. steering at the embedding (and at two layers)
        W = gpt.token_embedding.weight.detach().float().cpu().numpy()
        u = W[[name_id[n] for n in N]].mean(0) - W[[name_id[n] for n in K]].mean(0)
        rng = np.random.default_rng(0)
        r = rng.normal(size=u.shape)
        r *= np.linalg.norm(u) / np.linalg.norm(r)
        base_K, base_N = fork_logodds(gpt, K_items), fork_logodds(gpt, N_items)
        steer = {"K natural": base_K, "N natural": base_N}
        for a in args.alphas:
            if a != 0:
                steer[f"K + {a:g}·u_emb"] = fork_logodds(gpt, K_items, embed_steer(gpt, u, a))
        steer["N − 1·u_emb"] = fork_logodds(gpt, N_items, embed_steer(gpt, u, -1.0))
        steer["K + 1·random"] = fork_logodds(gpt, K_items, embed_steer(gpt, r, 1.0))
        steer["K + 2·random"] = fork_logodds(gpt, K_items, embed_steer(gpt, r, 2.0))
        if not args.no_layer_steer:
            isN = np.array([c == "N" for c in cls])
            for layer in sorted({L // 2, L}):
                uL = Xn[isN, layer].mean(0) - Xn[~isN, layer].mean(0)  # class gap at the name position after block `layer`
                for a in (1.0, 2.0):
                    steer[f"K + {a:g}·u_L{layer} (block {layer} out)"] = fork_logodds(gpt, K_items, layer_steer(gpt, layer, uL, a))
        results[tag] = dict(n_layer=L, dprime_name=d_name, dprime_last=d_last, steer=steer)
        print(f"\n{tag}  ({L} blocks)")
        print("   d′ N vs K by layer (0 = embedding):  name position " + " ".join(f"{v:5.1f}" for v in d_name))
        print("                                        last position " + " ".join(f"{v:5.1f}" for v in d_last))
        print(f"   {'steering':36s} {'ALL':>7s} {'steal':>7s}")
        for k, v in steer.items():
            print(f"   {k:36s} {v['ALL']:+7.2f} {v['steal']:+7.2f}")
        del gpt

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    tags = list(results)
    for i, tag in enumerate(tags):
        r_ = results[tag]
        c = cmap(i / max(len(tags) - 1, 1))
        axes[0].plot(range(r_["n_layer"] + 1), r_["dprime_name"], "o-", color=c, label=tag, ms=3)
        axes[1].plot(range(r_["n_layer"] + 1), r_["dprime_last"], "o-", color=c, label=tag, ms=3)
        st = r_["steer"]
        xs = [0.0] + [a for a in args.alphas if a != 0]
        ys = [st["K natural"]["ALL"]] + [st[f"K + {a:g}·u_emb"]["ALL"] for a in args.alphas if a != 0]
        axes[2].plot(xs, ys, "o-", color=c, label=tag, ms=4)
        axes[2].axhline(st["N natural"]["ALL"], color=c, ls=":", lw=0.8)
        axes[2].scatter([1, 2], [st["K + 1·random"]["ALL"], st["K + 2·random"]["ALL"]], marker="x", color=c)
    axes[0].set_title("d′ N vs K at the actor's name token", fontsize=10)
    axes[1].set_title("d′ N vs K at the last prompt token (prediction site)", fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel("layer (0 = embedding, L = after block L)")
        ax.set_ylabel("split-half d′")
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
