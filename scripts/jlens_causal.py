# ruff: noqa: E501
"""Follow-up tests of the J-lens (docs/2026-09-27_REPORT_jlens.md, 2026-09-28): what the J-lens's
peaked mid-layer content is, and whether it is causal. Lexical y = 1 finals with a kept lens
(logs/v4/jlens/box/jbar); the persona readout is cast_probe's k0 fork items (bad minus benign ending
log-odds, per act).

    future  the top-1 J-lens (and logit-lens) token at inner blocks on held-out text: is it the next
            token, a token later in the window (2..64 ahead), or one earlier in the window? Against a
            control that takes the top-1 token from another position.
    swap    the paper's coordinate swap (their France <-> China): at one block, at every position,
            exchange the lens coordinates of a villain name and a kind name (least-squares coordinates
            on the two lens vectors, the rest of the activation unchanged). Does the villain's persona
            move toward the kind name's, on every act at once? J-lens vectors against logit-lens vectors
            against a random pair of directions of the same norms; both directions (villain -> kind on
            villain items, kind -> villain on kind items); every inner block separately.
    axis    3c: the class axis at name positions (mean villain-name minus kind-name residual, per block)
            split into its J-space part (sparse non-negative combination of J-lens vectors), its
            logit-space part (the same with logit-lens vectors) and the J-remainder; each direction is
            projected out at one block (all positions) and the persona on each act is measured. Is the
            persona carried by the J-space part?
    incontext  the in-context two-hop on ladder-C models: the same split of the history direction at the
            test's name, and its effect on the decision (character after bad-bad minus good-good history).

    uv run python scripts/jlens_causal.py future --run v4-lex-y100-10M-s0
    uv run python scripts/jlens_causal.py swap   --run v4-lex-y100-10M-s0 [--pairs 4] [--alpha 1]
    uv run python scripts/jlens_causal.py axis   --run v4-lex-y100-10M-s0
"""

import argparse
import contextlib
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
import cast_probe
import eval_v4
from jlens import SKIP, TOK, load, val_halves

from dramatis.data import get_batch
from dramatis.device import pick_device
from dramatis.forks import PERFORMED, load_forks
from dramatis.jlens import JacobianFit, lens_vectors, sparse_decompose
from dramatis.provenance import git_commit, git_dirty

OUT = Path("logs/v4/jlens/box/causal")
JBAR = Path("logs/v4/jlens/box/jbar")
ACTS = list(PERFORMED)


def model_and_lens(run, dev):
    gpt, step, ck, cfg = load(run, "final", dev)
    f = glob.glob(str(JBAR / f"{run}@{step}.pt"))[0]
    d = torch.load(f, weights_only=False)
    J = JacobianFit(d["sums"], tuple(d["counts"])).mean.to(dev)
    with torch.no_grad():
        VJ = [lens_vectors(gpt, J[li]) for li in range(len(J))]
        VL = lens_vectors(gpt, torch.eye(J.shape[-1], device=dev))
    return gpt, step, ck, cfg, VJ, VL, J


@contextlib.contextmanager
def edit_block(gpt, layer, fn):
    """Apply fn to the residual stream after block `layer`, at every position (layer 0: to the token
    embeddings, before the position embeddings are added)."""
    mod = gpt.token_embedding if layer == 0 else gpt.blocks[layer - 1]
    h = mod.register_forward_hook(lambda m, i, o: fn(o))
    try:
        yield
    finally:
        h.remove()


def swap_fn(a, b, alpha=1.0):
    """Exchange the least-squares coordinates of h on directions a and b (rows), times alpha."""
    A = torch.stack([a, b])  # (2, d)
    G = torch.linalg.inv(A @ A.T)

    def fn(h):
        c = (h @ A.T) @ G  # (..., 2) coordinates
        return h + ((c.flip(-1) - c) * alpha) @ A

    return fn


def project_out_fn(u):
    u = u / u.norm()
    return lambda h: h - (h @ u)[..., None] * u


def persona(lo, cls_n, cls_k):
    return {a: float(lo[(cls_n, a)].mean() - lo[(cls_k, a)].mean()) for a in ACTS if (cls_n, a) in lo and (cls_k, a) in lo}


def score(gpt, seqs, meta):
    with torch.no_grad():
        return cast_probe.score_model(gpt, seqs, meta)


# ------------------------------------------------------------------ future


def future(a):
    dev = pick_device("auto")
    gpt, _, _, cfg, _, _, J = model_and_lens(a.run, dev)
    _, second, _ = val_halves(cfg)
    w, _ = get_batch(second, 16, 512, np.random.default_rng(5), dev)
    with torch.no_grad():
        stream = gpt.residual_stream(w)
        pred = gpt.unembed(stream[-1]).argmax(-1)
    toks = w.cpu()
    L = len(stream) - 1
    pos = torch.arange(SKIP, 512 - 65)
    rng = np.random.default_rng(0)
    out = {"run": a.run, "blocks": []}
    for li in range(1, L):
        row = {"layer": li}
        for name in ("jlens", "logit"):
            with torch.no_grad():
                h = stream[li]
                S = gpt.unembed(h @ J[li].T) if name == "jlens" else gpt.unembed(h)
                S = S - S[:, SKIP:511].mean((0, 1))  # centered, as in the peakedness statistic
                top = S.argmax(-1).cpu()
            common = torch.bincount(top[:, SKIP:511].reshape(-1), minlength=S.shape[-1]).topk(8)
            row.setdefault("common_top", {})[name] = [(TOK.decode([int(t)]), round(int(c) / top[:, SKIP:511].numel(), 3)) for c, t in zip(common.values, common.indices, strict=True)]
            nxt = ahead = earlier = ctrl = 0
            n = 0
            for b in range(toks.shape[0]):
                for t in pos.tolist():
                    tk = int(top[b, t])
                    n += 1
                    nxt += tk == int(pred[b, t])
                    fut = toks[b, t + 2 : t + 65]
                    ahead += bool((fut == tk).any())
                    earlier += bool((toks[b, : t + 1] == tk).any())
                    t2 = int(rng.integers(SKIP, 512 - 65))
                    ctrl += bool((fut == int(top[b, t2])).any())
            row[name] = {"next": nxt / n, "ahead_2_64": ahead / n, "earlier": earlier / n, "control_ahead": ctrl / n}
        out["blocks"].append(row)
        print(f"block {li}/{L}: J next {row['jlens']['next']:.2f} ahead {row['jlens']['ahead_2_64']:.2f} (ctrl {row['jlens']['control_ahead']:.2f}) earlier {row['jlens']['earlier']:.2f} | "
              f"logit next {row['logit']['next']:.2f} ahead {row['logit']['ahead_2_64']:.2f} (ctrl {row['logit']['control_ahead']:.2f}) earlier {row['logit']['earlier']:.2f}", flush=True)
    save(out, "future", a.run)


# ------------------------------------------------------------------ swap


LENSES = "jlens,logit,random"


def swap(a):
    dev = pick_device("auto")
    gpt, _, _, _, VJ, VL, _ = model_and_lens(a.run, dev)
    naughty = list(cast_probe.cast["naughty"])
    kind = [n for n in cast_probe.cast["kind"] if n not in eval_v4.FT_KIND]
    pairs = list(zip(naughty, kind, strict=False))[: a.pairs]
    classes = [f"naughty:{v}" for v, _ in pairs] + [f"kind:{k}" for _, k in pairs]
    forks = load_forks("v4")
    seqs, meta = cast_probe.build_k0(eval_v4.FT_KIND, eval_v4.FT_NEW, forks=forks, classes=classes, per_name=True)
    clean = score(gpt, seqs, meta)
    tid = lambda n: TOK.encode(" " + n).ids[0]  # noqa: E731
    gen = torch.Generator().manual_seed(7)
    L = len(VJ) - 1
    out = {"run": a.run, "pairs": pairs, "alpha": a.alpha, "clean_gap": {}, "blocks": []}
    for v, k in pairs:
        out["clean_gap"][f"{v}|{k}"] = persona(clean, f"naughty:{v}", f"kind:{k}")
    for li in range(a.from_block, L):
        row = {"layer": li}
        for lens in a.lenses.split(","):
            moves = {"villain_to_kind": [], "kind_to_villain": []}
            for v, k in pairs:
                if lens == "random":
                    va, vb = VJ[li][tid(v)], VJ[li][tid(k)]
                    ra, rb = torch.randn(va.shape, generator=gen).to(dev), torch.randn(va.shape, generator=gen).to(dev)
                    va, vb = ra / ra.norm() * va.norm(), rb / rb.norm() * vb.norm()
                elif lens == "embed":  # positive control: the names' own input embeddings, which the stream still carries
                    E = gpt.token_embedding.weight.detach()
                    va, vb = E[tid(v)], E[tid(k)]
                else:
                    V = VJ[li] if lens == "jlens" else VL
                    va, vb = V[tid(v)], V[tid(k)]
                sel = [i for i, m in enumerate(meta) if m[0] in (f"naughty:{v}", f"kind:{k}")]
                s_seqs = [sq for i in sel for sq in (seqs[2 * i], seqs[2 * i + 1])]  # two endings per item
                s_meta = [meta[i] for i in sel]
                with edit_block(gpt, li, swap_fn(va, vb, a.alpha)):
                    lo = score(gpt, s_seqs, s_meta)
                gap = out["clean_gap"][f"{v}|{k}"]
                for act in ACTS:
                    g = gap.get(act)
                    if not g or abs(g) < 0.5:  # acts without a persona for this pair carry no signal
                        continue
                    dv = float(lo[(f"naughty:{v}", act)].mean() - clean[(f"naughty:{v}", act)].mean())
                    dk = float(lo[(f"kind:{k}", act)].mean() - clean[(f"kind:{k}", act)].mean())
                    moves["villain_to_kind"].append({"pair": f"{v}|{k}", "act": act, "frac": -dv / g})
                    moves["kind_to_villain"].append({"pair": f"{v}|{k}", "act": act, "frac": dk / g})
            row[lens] = {d: {"mean_frac": float(np.mean([m["frac"] for m in ms])) if ms else float("nan"),
                             "by_act": {act: float(np.mean([m["frac"] for m in ms if m["act"] == act])) for act in ACTS if any(m["act"] == act for m in ms)},
                             "n": len(ms)} for d, ms in moves.items()}
        out["blocks"].append(row)
        f = lambda r, d: r[d]["mean_frac"]  # noqa: E731
        print(f"block {li}/{L}: fraction of the class gap moved (villain->kind / kind->villain): "
              + "  ".join(f"{lens} {f(row[lens], 'villain_to_kind'):+.2f} / {f(row[lens], 'kind_to_villain'):+.2f}" for lens in a.lenses.split(",")), flush=True)
    tag = ("-a" + f"{a.alpha:g}" if a.alpha != 1 else "") + (f"-from{a.from_block}" if a.from_block != 1 else "") + ("" if a.lenses == LENSES else "-" + a.lenses.replace(",", "+"))
    save(out, "swap" + tag, a.run)


# ------------------------------------------------------------------ axis


def axis(a):
    dev = pick_device("auto")
    gpt, _, _, cfg, VJ, VL, _ = model_and_lens(a.run, dev)
    naughty = {TOK.encode(" " + n).ids[0] for n in cast_probe.cast["naughty"]}
    kind = {TOK.encode(" " + n).ids[0] for n in cast_probe.cast["kind"]}
    _, second, _ = val_halves(cfg)
    w, _ = get_batch(second, 64, 512, np.random.default_rng(6), dev)
    with torch.no_grad():
        stream = gpt.residual_stream(w)
    toks = w[:, SKIP:511].reshape(-1)
    isN = torch.tensor([int(t) in naughty for t in toks.cpu()], device=dev)
    isK = torch.tensor([int(t) in kind for t in toks.cpu()], device=dev)
    forks = load_forks("v4")
    seqs, meta = cast_probe.build_k0(eval_v4.FT_KIND, eval_v4.FT_NEW, forks=forks, classes=["naughty", "kind"])
    clean = persona(score(gpt, seqs, meta), "naughty", "kind")
    gen = torch.Generator().manual_seed(8)
    L = len(VJ) - 1
    out = {"run": a.run, "clean_persona": clean, "n_names": [int(isN.sum()), int(isK.sum())], "blocks": []}
    for li in range(1, L):
        H = stream[li][:, SKIP:511].reshape(-1, stream[li].shape[-1])
        ax = H[isN].mean(0) - H[isK].mean(0)
        parts = {}
        for name, V in (("J", VJ[li]), ("logit", VL)):
            _, xJ, _ = sparse_decompose(V, ax[None], a.k)
            parts[name] = xJ[0]
        parts["J_remainder"] = ax - parts["J"]
        parts["full"] = ax
        r = torch.randn(ax.shape, generator=gen).to(dev)
        parts["random"] = r
        row = {"layer": li, "share_J": float(parts["J"].norm() ** 2 / ax.norm() ** 2),
               "share_logit": float(parts["logit"].norm() ** 2 / ax.norm() ** 2)}
        for name, u in parts.items():
            with edit_block(gpt, li, project_out_fn(u)):
                p = persona(score(gpt, seqs, meta), "naughty", "kind")
            row[name] = {"frac_left": {act: p[act] / clean[act] for act in p if abs(clean[act]) > 0.5},
                         "mean_frac_left": float(np.mean([p[act] / clean[act] for act in p if abs(clean[act]) > 0.5]))}
        out["blocks"].append(row)
        print(f"block {li}/{L}: axis variance in J part {row['share_J']:.2f} (logit part {row['share_logit']:.2f}); persona left after removing: "
              + " ".join(f"{n} {row[n]['mean_frac_left']:.2f}" for n in ("full", "J", "logit", "J_remainder", "random")), flush=True)
    save(out, "axis", a.run)


def incontext(a):
    """The in-context two-hop ("the child who stole earlier does something else bad later"): the
    history direction at the test's name (mean over stories of [BB − GG at the villain] − [the same at
    the other child], 3b's measure) split into its J-space part, logit-space part and J-remainder;
    each projected out at one block (all positions); the history effect on the decision is
    character(T_BB) − character(T_GG) (the C2 transplant sets, villain minus other child)."""
    import incontext_history_probe as hp
    from incontext_evidence import OUT as EVID
    from jlens_incontext import features, per_story_delta

    from dramatis.light_probes import ProbeSet
    from dramatis.tokenizer import StoryTokenizer

    dev = pick_device("auto")
    gpt, _, _, _, VJ, VL, _ = model_and_lens(a.run, dev)
    stok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
    sets = {arm: ProbeSet(EVID / "sets" / f"{arm}.json", stok, 512) for arm in ("T_BB", "T_GG")}

    def effect():
        with torch.no_grad():
            ch = {arm: sets[arm].summarize(ps.score(gpt))[f"{arm}/character/ALL"] for arm, ps in sets.items()}
        return ch["T_BB"] - ch["T_GG"]

    stories = hp.build_rows()
    stories = {k: v for k, v in stories.items() if any(it["villain"] for it in v["items"]) and any(not it["villain"] for it in v["items"])}
    F = features(gpt, stories)
    D = torch.from_numpy(per_story_delta(F, "setup").mean(0)).float().to(dev)  # (L + 1, d)
    clean = effect()
    gen = torch.Generator().manual_seed(9)
    L = len(VJ) - 1
    out = {"run": a.run, "clean_effect": clean, "blocks": []}
    for li in range(1, L):
        ax = D[li]
        parts = {"full": ax}
        for name, V in (("J", VJ[li]), ("logit", VL)):
            _, xJ, _ = sparse_decompose(V, ax[None], a.k)
            parts[name] = xJ[0]
        parts["J_remainder"] = ax - parts["J"]
        parts["random"] = torch.randn(ax.shape, generator=gen).to(dev)
        row = {"layer": li, "share_J": float(parts["J"].norm() ** 2 / ax.norm() ** 2), "share_logit": float(parts["logit"].norm() ** 2 / ax.norm() ** 2)}
        for name, u in parts.items():
            with edit_block(gpt, li, project_out_fn(u)):
                row[name] = effect() / clean
        out["blocks"].append(row)
        print(f"block {li}/{L}: history direction in J part {row['share_J']:.2f} (logit {row['share_logit']:.2f}); history effect left after removing: "
              + " ".join(f"{n} {row[n]:.2f}" for n in ("full", "J", "logit", "J_remainder", "random")), flush=True)
    save(out, "incontext", a.run)


def save(out, kind, run):
    out.update({"git_commit": git_commit(), "git_dirty": git_dirty()})
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{kind}-{run}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"saved {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["future", "swap", "axis", "incontext"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--from-block", type=int, default=1, help="swap: first block to edit (0 = the token embeddings)")
    ap.add_argument("--lenses", default=LENSES, help="swap: which directions to exchange (jlens, logit, random, embed)")
    a = ap.parse_args()
    {"future": future, "swap": swap, "axis": axis, "incontext": incontext}[a.cmd](a)
