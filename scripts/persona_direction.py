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
finetune_persona.py, jacobian_ladder.py, jacobian_prefit.py, unpoison.py and eval_v4.py, and so
are the flow-scan pieces (plan §4 "Flow scans (5a)"): the prediction site as the ending span, the
check that the name and site masks never overlap, the two halves of the setups, class directions,
random directions and the shrinkage-LDA d′ for the separation left.

Fork items: --forks v4 (default, prompts/forks_v4.yaml; the performed acts) or v3 (the 1× items).

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
from cast_probe import _sentences, acts, cast, setups, sit, tok

from dramatis.device import pick_device
from dramatis.evals import encode_story_start
from dramatis.forks import PERFORMED, load_forks
from dramatis.probe_names import TAUGHT, first_mention, name_positions
from dramatis.provenance import git_commit
from dramatis.train import model_from_checkpoint

N, K = cast["naughty"], cast["kind"]
# the " " + name token: the embedding row the class gap is measured on (mid-story mentions);
# for WHERE a name is in a sequence use dramatis.probe_names, which also finds the story-initial one
name_id = {n: tok.encode(" " + n)[0] for n in N + K}


def build_items(seed=0, taught=TAUGHT, taught_per_act=16, forks=None, acts_only=PERFORMED):
    """k = 0 items as dicts (actor, cls, act, setup, fork, heldout, prompt, bad, benign, extra).

    Fork items from `forks` (a dramatis.forks.ForkSet; default v4), the acts in `acts_only`
    (default the four performed acts; pass forks.acts for the held-out acts too).
    The balanced set is one item per (fork setup, actor name), extra=False. The other child
    is a kind name that no finetune set taught (never the actor, never one of `taught`), so a
    taught name is never a bystander. Each taught name also gets extra items (the setups again
    with a fresh other child, adult and opening; extra=True) up to `taught_per_act` per act, for
    its attachment and spillover. Class-level measures (directions, d′, N − K) use the balanced
    set; the extras come from a separate random stream, so the balanced set does not depend on
    `taught_per_act`.
    """
    forks = forks or load_forks("v4")
    use = [f for f in forks.items if f.act in acts_only]
    others = [n for n in K if n not in taught]

    def item(rng, f, actor, extra):
        other = str(rng.choice([n for n in others if n != actor]))
        adult = str(rng.choice(acts.adult_roles))
        setting = str(rng.choice(sit.settings))
        opening = setups.openings[rng.integers(len(setups.openings))]
        opening = _sentences(opening.format(setting=setting, adult=adult))
        text = f.setup_text("$A", "$B", adult)
        prompt = f.join(opening, text).replace("$A", actor).replace("$B", other)
        bad, benign = f.endings(actor, other)
        return dict(actor=actor, cls="N" if actor in N else "K", act=f.act, setup=f.setup_id, fork=f.id, heldout=f.heldout, prompt=prompt, bad=bad, benign=benign, extra=extra)

    rng = np.random.default_rng(seed)
    items = [item(rng, f, actor, False) for f in use for actor in N + K]
    rng_x = np.random.default_rng([seed, 1])
    for act in dict.fromkeys(f.act for f in use):
        hs = [f for f in use if f.act == act]
        for actor in [n for n in taught if n in N + K]:
            for j in range(max(taught_per_act - len(hs), 0)):
                items.append(item(rng_x, hs[j % len(hs)], actor, True))
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


def site_span(b, g):
    """The prediction site in each sequence (plan §4): every position that predicts a token of
    the ending from the first difference on, i.e. d − 1 … len − 2 with d = divergence(b, g); the
    positions whose log-probs the fork log-odds' difference is made of. The first, d − 1, has the
    same input in both sequences (the class direction at the site is fitted there)."""
    d = divergence(b, g)
    return list(range(d - 1, len(b) - 1)), list(range(d - 1, len(g) - 1))


def mask_overlap(it):
    """Positions at which the actor's name tokens and the prediction site coincide, in either
    sequence ([] for a usable item). They would coincide if the endings differed right after the
    actor's name (the name token itself then predicting the fork) or named the actor again after
    the fork."""
    _, b, g = encoded(it)
    sb, sg = site_span(b, g)
    return sorted((set(name_positions(tok, b, it["actor"])) & set(sb)) | (set(name_positions(tok, g, it["actor"])) & set(sg)))


def drop_overlapping(items):
    """(items, dropped setups): the items whose name and site masks overlap are removed with
    their whole setup (the overlap is a property of the fork's endings, the same for every actor,
    so the set stays balanced). The flow scans need the two positions apart."""
    first = {}
    for it in items:  # one representative item per setup is enough
        first.setdefault(it["setup"], it)
    bad = sorted(s for s, it in first.items() if mask_overlap(it))
    return [it for it in items if it["setup"] not in bad], bad


@torch.no_grad()
def logodds(gpt, items, hook=None, batch_size=64, first_token=False):
    """Per-item fork log-odds (bad − benign, summed over the continuation's tokens), as an array;
    with first_token=True also the log-odds of the first differing token alone (a second array).

    An optional `hook(state)` registers forward hooks and returns a handle with .remove(); per
    batch, `state` holds boolean (B, T) masks: "name_mask", every token of every mention of the
    actor (the story-initial one included, see dramatis.probe_names); "site_mask", the prediction
    site (every position predicting an ending token from the first difference on, `site_span`);
    and "site_first_mask", the site's first position only (the 1× single-position site). With a
    hook, the name and site masks must not overlap on any item (asserted; `drop_overlapping`).
    """
    batches = []  # built first, so the overlap check runs before any hook is registered
    for start in range(0, len(items), batch_size // 2):
        chunk = items[start : start + batch_size // 2]
        seqs, n_prompt, spans, names = [], [], [], []
        for it in chunk:
            p, b, g = encoded(it)
            for s, span in zip((b, g), site_span(b, g), strict=True):
                seqs.append(s)
                n_prompt.append(len(p))
                spans.append(span)
                names.append(name_positions(tok, s, it["actor"]))
        T = max(map(len, seqs))
        x = torch.full((len(seqs), T), tok.pad_id, dtype=torch.long)
        masks = {k: torch.zeros((len(seqs), T), dtype=torch.bool) for k in ("name_mask", "site_mask", "site_first_mask")}
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
            masks["name_mask"][i, names[i]] = True
            masks["site_mask"][i, spans[i]] = True
            masks["site_first_mask"][i, spans[i][0]] = True
        if hook:
            bad = (masks["name_mask"] & masks["site_mask"]).any(1).nonzero().flatten().tolist()
            assert not bad, f"name and site masks overlap on item {chunk[bad[0] // 2]['fork']} ({chunk[bad[0] // 2]['actor']}); use drop_overlapping()"
        batches.append((x, masks, seqs, n_prompt, spans))
    state = {}
    h = hook(state) if hook else None
    out, out1 = [], []
    try:
        for x, masks, seqs, n_prompt, spans in batches:
            x = x.to(gpt.device)
            state.update({k: m.to(gpt.device) for k, m in masks.items()})
            logp = F.log_softmax(gpt(x)[:, :-1].float(), dim=-1).gather(-1, x[:, 1:, None]).squeeze(-1)
            for i in range(0, len(seqs), 2):
                out.append(float(logp[i, n_prompt[i] - 1 : len(seqs[i]) - 1].sum() - logp[i + 1, n_prompt[i + 1] - 1 : len(seqs[i + 1]) - 1].sum()))
                out1.append(float(logp[i, spans[i][0]] - logp[i + 1, spans[i + 1][0]]))
    finally:  # an error must not leave the hook on the model
        if h:
            h.remove()
    return (np.array(out), np.array(out1)) if first_token else np.array(out)


def by_act(items, lo):
    """{"ALL": mean, <act>: mean} of per-item log-odds."""
    out = {"ALL": float(np.mean(lo))}
    for act in dict.fromkeys(it["act"] for it in items):
        out[act] = float(np.mean([v for it, v in zip(items, lo, strict=True) if it["act"] == act]))
    return out


def layer_hook(gpt, layer, where, fn):
    """Hook maker for `logodds`: apply fn(x) -> x at block `layer`'s output (layer 0: the token
    embedding's output) at the positions in state[where] ("name_mask", "site_mask" or
    "site_first_mask")."""
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


@torch.no_grad()
def capture_site(gpt, items, batch_size=64, mention="first"):
    """Residual stream after the embedding and after each block at (a) the actor's name (see
    name_pos: "first" = the first mention, often story-initial; "spaced" = the first " " + name
    token, where class directions applied at every mention are fitted) and (b) the prediction
    site's FIRST position: the one that predicts the first token where the bad and benign endings
    differ (the shared prefix of the endings is included in the forward pass; its input is the
    same in both branches). Returns two arrays (n_items, n_layers + 1, d) and the site positions."""
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


# ---- flow-scan pieces (plan §4 "Flow scans (5a)"; persona_circuit.py, finetune_persona.py) -------

LDA_LAMBDA = 0.1  # shrinkage for the separation-left LDA, fixed in advance (fraction of the mean variance)
SCAN_BATCH = 256  # sequences per forward pass in the scans (many passes over the same items)


def setup_parts(items, seed=0, n=4):
    """0 … n − 1 per item: which part of the setups its setup is in. Each act's setups are
    shuffled (seeded) and dealt round-robin, so every part holds every act. With n = 4, part % 2
    is the half (setup_halves) and part // 2 splits each half again (separation_left)."""
    rng = np.random.default_rng(seed)
    part = {}
    for act in dict.fromkeys(it["act"] for it in items):
        setups_ = sorted({it["setup"] for it in items if it["act"] == act})
        for i, s_ in enumerate(rng.permutation(setups_)):
            part[str(s_)] = i % n
    return np.array([part[it["setup"]] for it in items])


def setup_halves(items, seed=0):
    """0 / 1 per item: which half of the setups its setup is in (both halves hold every act).
    Directions are fitted on one half and scored on the other, then the reverse, averaged."""
    return setup_parts(items, seed) % 2


def class_dir(A, isN, rows):
    """(unit direction, grand mean, gap norm) of the N − K class-mean difference of A (n, d) over
    the boolean `rows`."""
    gap = A[rows & isN].mean(0) - A[rows & ~isN].mean(0)
    norm = float(np.linalg.norm(gap))
    return gap / (norm + 1e-12), A[rows].mean(0), norm


def mean_ablated(A, u, mu):
    """A (n, d) with its component along the unit vector u set to that of mu (as mean_ablate)."""
    return A - ((A - mu) @ u)[:, None] * u


def random_units(d, n, seed):
    """n random unit vectors in d dimensions (the control directions)."""
    r = np.random.default_rng(seed).normal(size=(n, d))
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def lda_dprime(Xf, yf, Xe, ye, lam=LDA_LAMBDA):
    """Held-out d′ of a shrinkage-LDA direction: w = (Σ_w + λ·(tr Σ_w / d)·I)⁻¹ (μ_1 − μ_0),
    fitted on (Xf, yf) with Σ_w the pooled within-class covariance, and d′ of the projections on
    (Xe, ye); y boolean (True = N). The ridge keeps w defined when a direction has been ablated
    (zero variance along it)."""
    Xf, Xe = Xf.astype(np.float64), Xe.astype(np.float64)
    X1, X0 = Xf[yf], Xf[~yf]
    C = (np.cov(X1, rowvar=False) * (len(X1) - 1) + np.cov(X0, rowvar=False) * (len(X0) - 1)) / (len(Xf) - 2)
    d = C.shape[0]
    w = np.linalg.solve(C + lam * np.trace(C) / d * np.eye(d), X1.mean(0) - X0.mean(0))
    return _dprime(Xe[ye] @ w, Xe[~ye] @ w)


def ablation_scan(gpt, items, fit_rows, score_rows, parts, n_random=3, seed=0, lam=LDA_LAMBDA):
    """One model's flow scan. For each half h of the setups, the class direction (N − K means) is
    fitted on the `fit_rows` of half h (parts % 2, from setup_parts) after every block (0 = the
    embedding) at the actor's name
    (the first " name" token) and at the prediction site's first position; each is mean-ablated
    in turn along the whole of its position ("name_mask": every mention of the actor;
    "site_mask": every position of the ending from the first difference on) and the `score_rows`
    of the OTHER half are scored. Returns a dict:
      dirs[(h, pos, L)] = (u, mu, gap norm)
      lo[(h, key)] = (per-item log-odds, first-token log-odds) over the scored items of the other
          half, key = "none" | (pos, L, "class") | (pos, L, j) for random direction j
      score_idx[h] = indices (into items) of those scored items
      sep[(h, pos, L)] = (d′ before, d′ after the class-direction ablation): held-out
          shrinkage-LDA d′ of N vs K (`separation_left`)
    The random directions are the same for every model and block (seeded by position), so a
    finetune and its neutral are ablated along the same ones."""
    Xn, Xs, _ = capture_site(gpt, items, SCAN_BATCH, mention="spaced")
    X = {"name": Xn, "site": Xs}
    isN = np.array([it["cls"] == "N" for it in items])
    groups = name_groups(items, fit_rows, seed)
    halves = parts % 2
    L = Xn.shape[1] - 1
    out = {"X": X, "dirs": {}, "lo": {}, "score_idx": {}, "sep": {}, "n_layer": L}
    for h in (0, 1):
        fit = fit_rows & (halves == h)
        idx = np.flatnonzero(score_rows & (halves != h))
        out["score_idx"][h] = idx
        its = [items[i] for i in idx]
        out["lo"][(h, "none")] = logodds(gpt, its, batch_size=SCAN_BATCH, first_token=True)
        for pos, mask in (("name", "name_mask"), ("site", "site_mask")):
            rand = random_units(Xn.shape[2], n_random, seed=[seed, 0 if pos == "name" else 1])
            for li in range(L + 1):
                A = X[pos][:, li]
                u, mu, gap = class_dir(A, isN, fit)
                out["dirs"][(h, pos, li)] = (u, mu, gap)
                out["lo"][(h, (pos, li, "class"))] = logodds(gpt, its, layer_hook(gpt, li, mask, mean_ablate(u, mu)), SCAN_BATCH, first_token=True)
                for j in range(n_random):
                    out["lo"][(h, (pos, li, j))] = logodds(gpt, its, layer_hook(gpt, li, mask, mean_ablate(rand[j], mu)), SCAN_BATCH, first_token=True)
                out["sep"][(h, pos, li)] = separation_left(A, isN, fit, fit_rows & (halves != h), parts // 2, groups, lam)
    return out


def name_groups(items, rows, seed=0):
    """0 / 1 per item: a fixed random half of the N-names and of the K-names among `rows` in
    group 0, the rest in group 1 (the name split of `separation_left`)."""
    rng = np.random.default_rng([seed, 7])
    group = {}
    for cls in ("N", "K"):
        names = sorted({it["actor"] for it, r in zip(items, rows, strict=True) if r and it["cls"] == cls})
        for i, n in enumerate(rng.permutation(names)):
            group[str(n)] = i % 2
    return np.array([group.get(it["actor"], -1) for it in items])


def separation_left(A, isN, u_rows, rows, sub, groups, lam=LDA_LAMBDA):
    """(d′ before, d′ after) removing the class direction at one block and position: the
    held-out d′ of N vs K along a shrinkage-LDA direction refitted on the (ablated) activations.

    Every piece is fitted and evaluated on different items, by setup AND by name:
      * the removed direction u_g: N − K means of the `u_rows` (the half of the setups the scan
        fitted its direction on) among name group g (`groups`), mean-ablated as the scan does;
      * the LDA: fitted on the `rows` (the scored half) of setup part s (`sub`) and names g;
      * d′: on the scored half's other setup part and the other names (1 − g);
    the four (s, g) combinations averaged. Why not the plan's literal wording (the LDA refitted
    on the half that defined u): u is that half's own N − K mean difference, so once it is
    removed that half has no mean difference left to find; and a direction fitted on the names
    being evaluated forces their remaining mean difference to point against the fitted one's
    (d′ ≈ −2 to −5 at the name token on the 1× 10M model). At the name token the name split also
    keeps the names' identities out of the d′ (the 2026-09-22 audit). u_g is the class direction
    estimated from half the names, so it is close to, not identical with, the scan's. A: (n, d)."""
    before, after = [], []
    for g in (0, 1):
        u, mu, _ = class_dir(A, isN, u_rows & (groups == g))
        Ab = mean_ablated(A, u, mu)
        for part in (0, 1):
            fit = rows & (sub == part) & (groups == g)
            ev = rows & (sub != part) & (groups != g)
            before.append(lda_dprime(A[fit], isN[fit], A[ev], isN[ev], lam))
            after.append(lda_dprime(Ab[fit], isN[fit], Ab[ev], isN[ev], lam))
    return float(np.mean(before)), float(np.mean(after))


def scan_keys(L, n_random):
    """Every ablation key of ablation_scan, "none" first."""
    return ["none"] + [(pos, li, k) for pos in ("name", "site") for li in range(L + 1) for k in ("class", *range(n_random))]


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


def measure(gpt, items, alphas=(0, 1, 2, 3), splits=20, layer_steer=True):
    """A (d′ by layer) and B (steering) on one model, as main() records them; plus the
    embedding-row d′ (the capacity grid's "direction present") and the split-half d′ at the
    prediction site's first position (the transfer, 2d). `items`: the balanced set."""
    K_items = [it for it in items if it["cls"] == "K"]
    N_items = [it for it in items if it["cls"] == "N"]
    L = len(gpt.blocks)
    # ---- A. layer-wise separation
    Xn, Xl = capture(gpt, items)
    Xs, _ = capture(gpt, items, mention="spaced")
    _, Xsite, _ = capture_site(gpt, items)
    cls, setup, actors = [it["cls"] for it in items], [it["setup"] for it in items], [it["actor"] for it in items]
    d_name = heldout_name_dprime(Xn, actors, N, splits)
    d_null = heldout_name_dprime(Xn, actors, N, splits, seed=1, relabel=True)
    d_sp = heldout_name_dprime(Xs, actors, N, splits)
    d_sp_null = heldout_name_dprime(Xs, actors, N, splits, seed=1, relabel=True)
    d_name_setup = split_half_dprime(Xs, cls, setup)  # the old measure
    d_last = split_half_dprime(Xl, cls, setup)
    d_site = split_half_dprime(Xsite, cls, setup)
    # ---- B. steering at the embedding (and at two layers), at every mention of the actor
    W = gpt.token_embedding.weight.detach().float().cpu().numpy()
    u = W[[name_id[n] for n in N]].mean(0) - W[[name_id[n] for n in K]].mean(0)
    # embedding-row d′ of the N rows against the K rows along their mean difference (in-sample,
    # as at 1×), with its floor: the same for random 8 / 24 splits of the same rows
    rows = W[[name_id[n] for n in N + K]]
    d_rows = _dprime(rows[: len(N)] @ u, rows[len(N) :] @ u)
    rng_rows = np.random.default_rng(2)
    d_rows_null = []
    for _ in range(50):
        perm = rng_rows.permutation(len(rows))
        a, b = rows[perm[: len(N)]], rows[perm[len(N) :]]
        v = a.mean(0) - b.mean(0)
        d_rows_null.append(_dprime(a @ v, b @ v))
    rng = np.random.default_rng(0)
    r = rng.normal(size=u.shape)
    r *= np.linalg.norm(u) / np.linalg.norm(r)

    def steer_(its, layer, vec):
        return by_act(its, logodds(gpt, its, layer_hook(gpt, layer, "name_mask", add_vec(vec))))

    base_K, base_N = by_act(K_items, logodds(gpt, K_items)), by_act(N_items, logodds(gpt, N_items))
    steer = {"K natural": base_K, "N natural": base_N}
    for a in alphas:
        if a != 0:
            steer[f"K + {a:g}·u_emb"] = steer_(K_items, 0, a * u)
    steer["N − 1·u_emb"] = steer_(N_items, 0, -u)
    steer["K + 1·random"] = steer_(K_items, 0, r)
    steer["K + 2·random"] = steer_(K_items, 0, 2 * r)
    if layer_steer:
        isN = np.array([c == "N" for c in cls])
        for layer in sorted({L // 2, L}):
            uL = Xs[isN, layer].mean(0) - Xs[~isN, layer].mean(0)  # class gap at the first " name" token after block `layer`
            for a in (1.0, 2.0):
                where = f"block {layer} out" if layer else "embedding"
                steer[f"K + {a:g}·u_L{layer} ({where})"] = steer_(K_items, layer, a * uL)
    gap = base_N["ALL"] - base_K["ALL"]
    return dict(
        n_layer=L,
        dprime_name=d_name.mean(0).tolist(),  # first mention, held-out names, mean over name splits
        dprime_name_sd=d_name.std(0, ddof=1).tolist(),
        dprime_name_null=d_null.mean(0).tolist(),  # permutation null: random 8/24 relabelings
        dprime_name_null_sd=d_null.std(0, ddof=1).tolist(),
        dprime_name_spaced=d_sp.mean(0).tolist(),  # first " " + name token, held-out names
        dprime_name_spaced_null=d_sp_null.mean(0).tolist(),
        dprime_name_setup_split=d_name_setup,  # the old, inflated measure (" " + name token, names in the fit)
        dprime_last=d_last,  # at the LAST PROMPT TOKEN (not the prediction site), split by setup
        dprime_site=d_site,  # at the prediction site's first position, split by setup
        dprime_emb_rows=d_rows,  # embedding rows, N vs K, in-sample
        dprime_emb_rows_null=float(np.mean(d_rows_null)),  # random 8 / 24 splits of the same rows
        steer=steer,
        steer_frac={k: (v["ALL"] - base_K["ALL"]) / gap for k, v in steer.items() if k.startswith("K +")},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ckpts", nargs="+")
    parser.add_argument("--alphas", nargs="*", type=float, default=[0, 1, 2, 3])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/persona_direction"))
    parser.add_argument("--no-layer-steer", action="store_true")
    parser.add_argument("--splits", type=int, default=20, help="random name splits (and null relabelings) for the name-token d′")
    parser.add_argument("--forks", default="v4", help="fork items: v4 (default), v3 (the 1× items) or a YAML path")
    args = parser.parse_args()
    device = pick_device("auto")
    forks = load_forks(args.forks)
    items = balanced(build_items(forks=forks))
    K_items = [it for it in items if it["cls"] == "K"]
    N_items = [it for it in items if it["cls"] == "N"]
    print(f"{len(items)} items: {len(N_items)} N-actor, {len(K_items)} K-actor, {len(set(it['setup'] for it in items))} setups")
    results = {}
    for ck in args.ckpts:
        tag = re.sub(r"^runs/\d{4}-\d{2}-\d{2}_\d{6}_", "", ck).split("/checkpoints/")[0]
        gpt = model_from_checkpoint(ck, device).eval()
        L = len(gpt.blocks)
        results[tag] = r_ = measure(gpt, items, args.alphas, args.splits, not args.no_layer_steer)
        d_name, d_null, d_sp, d_sp_null = (np.array(r_[k]) for k in ("dprime_name", "dprime_name_null", "dprime_name_spaced", "dprime_name_spaced_null"))
        d_name_setup, d_last, steer = r_["dprime_name_setup_split"], r_["dprime_last"], r_["steer"]
        base_K, gap = steer["K natural"], steer["N natural"]["ALL"] - steer["K natural"]["ALL"]
        print(f"\n{tag}  ({L} blocks)")
        print("   d′ N vs K by layer (0 = embedding):")
        rows = [
            ("name, first mention, held-out names", d_name),
            ("   null (random 8/24 relabelings)", d_null),
            ('name, first " name" token, held-out names', d_sp),
            ("   null (random 8/24 relabelings)", d_sp_null),
            ('   split by setup (old; names in the fit)', d_name_setup),
            ("last prompt token, split by setup", d_last),
            ("prediction site (first position), by setup", r_["dprime_site"]),
        ]
        for label, v in rows:
            print(f"     {label:44s}" + " ".join(f"{x:5.1f}" for x in v))
        print(f"     (sd over name splits, mean over layers: {np.mean(r_['dprime_name_sd']):.1f}; of the null {np.mean(r_['dprime_name_null_sd']):.1f})")
        print(f"     embedding rows, N vs K: d′ {r_['dprime_emb_rows']:.1f} (random 8/24 splits: {r_['dprime_emb_rows_null']:.1f})")
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
    results["_meta"] = {"script": "scripts/persona_direction.py", "git_commit": git_commit(), **forks.meta}
    json.dump(results, open(str(args.out) + ".json", "w"), indent=1)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
