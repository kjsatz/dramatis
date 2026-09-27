# ruff: noqa: E501
"""The J-lens on the dramatis models, Stage 0 (docs/2026-09-27_PLAN_jlens.md).

    fit    fit J_l for every layer of one checkpoint on windows from the FIRST half of its corpus's
           val split, logging convergence (the running mean's relative shift and the split-half
           gap after every chunk); saves logs/v4/jlens/jbar/<run>@<step><tag>.pt
    stats  per-block statistics on windows from the SECOND half of the val split, for the J-lens
           and the logit lens, each against Null A (matched noise): readout kurtosis (exceedance,
           Δκ; uncentered and per-token centered), next-token agreement with the model, top-1
           persistence against a position-shuffled null, occupancy against a random dictionary
           (and Null A), and class content at name positions; saves logs/v4/jlens/stats/...json

    uv run python scripts/jlens.py fit   --run v4-lex-y099-10M-s0 [--step final] [--windows 400] [--b 32]
                                         [--length 512] [--targets sum] [--null untrained|randomized] [--tag X]
    uv run python scripts/jlens.py stats --run v4-lex-y099-10M-s0 [--step final] [--tag X]

The corpus val splits are rebuilt locally (scripts/build_v4_corpora.sh) and checked here against the
sha256 each run recorded. Null B (--null untrained) and Null C (--null randomized) get their own lenses.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, "scripts")
from susceptibility_ft import REPO, hub_index

from dramatis.data import get_batch, load_token_file
from dramatis.device import pick_device
from dramatis.jlens import (
    SKIP,
    JacobianFit,
    averaged_jacobians,
    lens_vectors,
    matched_noise,
    randomized_blocks,
    readout,
    sparse_decompose,
    untrained_model,
)
from dramatis.provenance import file_sha256, git_commit, git_dirty
from dramatis.train import model_from_checkpoint

OUT = Path("logs/v4/jlens")
CHUNK = 25  # windows between convergence checks
VILLAIN_WORDS = ["steal", "thief", "naughty", "bad", "mean", "trick"]  # single spaced tokens
KIND_WORDS = ["kind", "nice", "good", "share", "honest"]  # "sorry" dropped in Stage 0: in these stories apologies follow bad acts
N_READ = 2000  # positions per block for the readout statistics
N_OCC = 500  # positions per block for occupancy
K_OCC = 64  # pursuit steps for occupancy
TOK = Tokenizer.from_file("tokenizer/story-bpe-4096.json")


# ------------------------------------------------------------------ models and data


DENSE_BRANCH = "incontext-dense"  # the dense ladder-C retrains: run names "<run>-dense" here


def dense_index():
    """{run-dense: {step: (revision, path)}} for the dense branch (cached)."""
    f = OUT / "dense_index.json"
    if not f.exists():
        idx = {}
        for p in HfApi().list_repo_files(REPO, revision=DENSE_BRANCH):
            m = re.match(r"runs/\d{4}-\d\d-\d\d_\d{6}_(v4-[^/]+)_dense/checkpoints/ckpt_(\d+)\.pt$", p)
            if m:
                idx.setdefault(f"{m.group(1)}-dense", {})[m.group(2).lstrip("0")] = (DENSE_BRANCH, p)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(idx))
    return json.loads(f.read_text())


def resolve(run, step):
    """(step, (revision, checkpoint path)); "final" is the run's last checkpoint, other steps snap
    to the nearest saved one."""
    idx = (dense_index() if run.endswith("-dense") else hub_index())[run]
    have = sorted(map(int, idx))
    step = have[-1] if step == "final" else min(have, key=lambda h: abs(h - int(step)))
    return step, idx[str(step)]


def load(run, step, device, null=None):
    step, (rev, path) = resolve(run, step)
    cfg = json.loads(Path(hf_hub_download(REPO, str(Path(path).parent.parent / "config.json"), revision=rev)).read_text())
    ck = hf_hub_download(REPO, path, revision=rev)
    gpt = model_from_checkpoint(ck, device)
    if null == "untrained":
        gpt = untrained_model(cfg["model_config"], cfg["seed"]).to(device)
    elif null == "randomized":
        gpt = randomized_blocks(gpt, seed=cfg["seed"] + 1000)
    return gpt.eval(), step, path, cfg


def val_halves(cfg):
    """The run's corpus val split (checked against its recorded sha256), cut in two halves."""
    corpus = Path(cfg["data"])
    path = Path("data/corpora") / corpus.name / "val.bin"
    want = cfg["data_provenance"]["val"]["sha256"]
    if file_sha256(path) != want:
        sys.exit(f"{path} does not match the run's recorded val sha256 {want[:12]}; rebuild the corpus")
    val = load_token_file(path)
    return val[: len(val) // 2], val[len(val) // 2 :], want


def lens_path(run, step, tag):
    return OUT / "jbar" / f"{run}@{step}{tag}.pt"


# ------------------------------------------------------------------ fit


def fit_tag(a):
    return a.tag or (f"-{a.null}" if a.null else "") + (f"-T{a.length}" if a.length != 512 else "") + ("-mean" if a.targets == "mean" else "")


def fit(a):
    dev = pick_device("auto")
    gpt, step, ck, cfg = load(a.run, a.step, dev, a.null)
    tag = fit_tag(a)
    dest = lens_path(a.run, step, tag)
    first, _, val_sha = val_halves(cfg)
    rng = np.random.default_rng(0)
    total, log, prev, t0 = None, [], None, time.time()
    for start in range(0, a.windows, CHUNK):
        n = min(CHUNK, a.windows - start)
        w, _ = get_batch(first, n, a.length, rng, "cpu")
        part = averaged_jacobians(gpt, w, dims_per_pass=a.b, targets=a.targets, first_index=start)
        total = part if total is None else total.merge(part)
        M = total.mean
        shift = None if prev is None else float(max(((M[li] - prev[li]).norm() / M[li].norm()) for li in range(len(M) - 1)))
        H = total.half_means()
        gap = float(max(((H[0][li] - H[1][li]).norm() / M[li].norm()) for li in range(len(M) - 1)))
        log.append({"windows": total.n_windows, "shift": shift, "half_gap": gap, "seconds": round(time.time() - t0, 1)})
        print(f"{a.run}@{step}{tag}: {total.n_windows} windows, shift {shift if shift is None else round(shift, 4)}, half gap {gap:.4f}, {time.time() - t0:.0f} s", flush=True)
        prev = M
        if total.n_windows in a.snapshots:  # for the convergence of the statistics themselves
            save(total, lens_path(a.run, step, f"{tag}-n{total.n_windows}"), a, step, ck, cfg, val_sha, log, dev)
    save(total, dest, a, step, ck, cfg, val_sha, log, dev)


def save(total, dest, a, step, ck, cfg, val_sha, log, dev):
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sums": total.sums, "counts": total.counts, "meta": {
        "run": a.run, "step": step, "checkpoint": ck, "null": a.null, "corpus": cfg["data"], "val_sha256": val_sha,
        "val_half": "first", "windows": total.n_windows, "length": a.length, "dims_per_pass": a.b, "targets": a.targets,
        "skip": SKIP, "rng_seed": 0, "convergence": log, "device": str(dev), "git_commit": git_commit(), "git_dirty": git_dirty()}}, dest)
    print(f"saved {dest}")


# ------------------------------------------------------------------ stats


def excess_kurtosis(S):
    z = S - S.mean(-1, keepdim=True)
    return (z**4).mean(-1) / (z**2).mean(-1).clamp_min(1e-12) ** 2 - 3.0


def kurtosis_stats(S_real, S_null):
    """Per-position excess kurtosis over the vocabulary, real against Null A, uncentered and with each
    token's mean over positions removed."""
    out = {}
    for name, (r, n) in {"raw": (S_real, S_null), "centered": (S_real - S_real.mean(0), S_null - S_null.mean(0))}.items():
        kr, kn = excess_kurtosis(r), excess_kurtosis(n)
        q99 = torch.quantile(kn, 0.99)
        out[name] = {"exceedance": float((kr > q99).float().mean()), "median_real": float(kr.median()),
                     "median_null": float(kn.median()), "delta": float(kr.median() - kn.median())}
    return out


def persistence(S_win, deltas=(1, 4, 16), gen=None):
    """Fraction of positions whose top-1 token (centered scores) is also top-1 Δ positions later,
    against the same after shuffling positions within each window."""
    top = S_win.argmax(-1)  # (W, P)
    out = {}
    for d in deltas:
        real = float((top[:, d:] == top[:, :-d]).float().mean())
        perm = torch.stack([row[torch.randperm(row.numel(), generator=gen)] for row in top])
        null = float((perm[:, d:] == perm[:, :-d]).float().mean())
        out[str(d)] = {"real": real, "null": null}
    return out


def active_tokens(S_real, S_null, current, z=4.0):
    """Capacity in readout space: per position, the number of tokens whose centered score is more
    than z noise standard deviations high (noise: Null A's per-token spread), not counting the
    position's own input token (the embedding echo, which Null C shows is large). Real against the
    same count for Null A itself (Gaussian tails alone give ≈ 0.13 per 4,096 tokens at z = 4)."""
    sd = S_null.std(0).clamp_min(1e-9)
    hot = ((S_real - S_real.mean(0)) / sd) > z
    hot[torch.arange(len(current)), current] = False
    real = hot.sum(1).float()
    null = (((S_null - S_null.mean(0)) / sd) > z).sum(1).float()
    return {"median": float(real.median()), "mean": float(real.mean()), "p90": float(real.quantile(0.9)),
            "null_median": float(null.median()), "null_mean": float(null.mean())}


def occupancy(atoms, X, rand_atoms):
    """The paper's occupancy: per position, the first pursuit step whose marginal gain (share of the
    vector's energy) falls below a random dictionary's median gain at that step; plus the excess
    fraction of variance explained at the median occupancy."""
    e = (X * X).sum(1, keepdim=True)
    _, _, err = sparse_decompose(atoms, X, K_OCC)
    _, _, err_r = sparse_decompose(rand_atoms, X, K_OCC)
    fve, fve_r = 1 - err / e, 1 - err_r / e
    gain = torch.diff(torch.cat([torch.zeros_like(e), fve], 1), dim=1)
    gain_r = torch.diff(torch.cat([torch.zeros_like(e), fve_r], 1), dim=1).median(0).values
    below = gain < gain_r
    k = torch.where(below.any(1), below.float().argmax(1), torch.full_like(below[:, 0], K_OCC, dtype=torch.long))
    km = int(k.median())
    idx = max(km - 1, 0)
    return {"median": km, "p25": float(k.float().quantile(0.25)), "p75": float(k.float().quantile(0.75)),
            "excess_fve_at_median": float((fve[:, idx] - fve_r[:, idx]).median())}


def name_ids(cfg):
    """{spaced-name token id: "N" or "K"} from the run's cast map; None if its corpus has none (the
    ladder corpora, whose names carry no class)."""
    source = cfg["data_provenance"]["train"]["source"]
    if "--cast " not in source:
        return None
    cast = yaml.safe_load(open(source.split("--cast ")[1].split()[0]))["names"]
    ids = {}
    for n, v in cast.items():
        t = TOK.encode(" " + n).ids
        if len(t) == 1:
            ids[t[0]] = "N" if v["p"] > 0.5 else "K"
    return ids


def word_ids(words):
    out = []
    for w in words:
        t = TOK.encode(" " + w).ids
        assert len(t) == 1, w
        out.append(t[0])
    return out


def class_profile(Vlin, H, labels, name_of, window_of, keep, n_perm=200, gen=None):
    """Sign-free class content at name positions, over the whole vocabulary (name tokens excluded):
    the villain-minus-kind profile of per-name mean linear scores, estimated separately on the even
    and odd windows. Reports its split-half correlation (does the profile replicate?) and its norm
    against a label-permutation null (is it larger than chance?)."""
    scores = (H @ Vlin.T)[:, keep]  # (n positions, kept vocab)
    names = sorted(set(name_of))
    lab = torch.tensor([labels[m] == "N" for m in names])
    name_arr, win_arr = torch.tensor(name_of), torch.tensor(window_of)

    def per_name(mask):
        return torch.stack([scores[(name_arr == m) & mask].mean(0) if ((name_arr == m) & mask).any() else torch.full((scores.shape[1],), float("nan")) for m in names])

    halves = [per_name(win_arr % 2 == p) for p in (0, 1)]
    both = per_name(torch.ones_like(win_arr, dtype=torch.bool))
    ok = ~torch.isnan(halves[0][:, 0]) & ~torch.isnan(halves[1][:, 0])

    def prof(M, lb, rows):
        return M[rows & lb].mean(0) - M[rows & ~lb].mean(0)

    p0, p1 = prof(halves[0], lab, ok), prof(halves[1], lab, ok)
    split = float(torch.corrcoef(torch.stack([p0 - p0.mean(), p1 - p1.mean()]))[0, 1])
    norm = float(prof(both, lab, torch.ones_like(lab)).norm())
    null = []
    for _ in range(n_perm):
        pl = lab[torch.randperm(len(lab), generator=gen)]
        null.append(float(prof(both, pl, torch.ones_like(lab)).norm()))
    null = np.array(null)
    return {"split_half_r": split, "norm": norm, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p": float((null >= norm).mean()), "n_names": len(names), "n_names_both_halves": int(ok.sum())}


def content(Vlin, H, labels, name_of, vw, kw, n_perm=500, gen=None):
    """Class content at name positions: the villain-word minus kind-word linear score, villain names
    minus kind names; the permutation null shuffles the class labels among names."""
    idx = Vlin[vw].mean(0) - Vlin[kw].mean(0)  # (d,): score direction "villain words minus kind words"
    s = H @ idx  # per name position
    names = sorted(set(name_of))
    per_name = torch.stack([s[torch.tensor([n == m for n in name_of])].mean() for m in names])
    lab = torch.tensor([labels[m] == "N" for m in names])
    stat = float(per_name[lab].mean() - per_name[~lab].mean())
    perms = []
    for _ in range(n_perm):
        p = lab[torch.randperm(len(lab), generator=gen)]
        perms.append(float(per_name[p].mean() - per_name[~p].mean()))
    perms = np.array(perms)
    return {"stat": stat, "null_sd": float(perms.std()), "p": float((np.abs(perms) >= abs(stat)).mean()),
            "n_positions": len(name_of), "n_names": len(names)}


@torch.no_grad()
def stats(a):
    dev = pick_device("auto")
    gpt, step, ck, cfg = load(a.run, a.step, dev, a.null)
    tag = a.tag or (f"-{a.null}" if a.null else "")
    if (OUT / "stats" / f"{a.run}@{step}{tag}.json").exists() and getattr(a, "skip_done", False):
        return
    lens = torch.load(lens_path(a.run, step, tag), weights_only=False)
    J = JacobianFit(lens["sums"], tuple(lens["counts"])).mean.to(dev)
    _, second, _ = val_halves(cfg)
    gen = torch.Generator().manual_seed(1)
    w, _ = get_batch(second, a.eval_windows, 512, np.random.default_rng(1), dev)
    stream = gpt.residual_stream(w)
    pred = gpt.unembed(stream[-1]).argmax(-1)  # the model's own top-1 next token
    valid = torch.zeros(512, dtype=torch.bool, device=dev)
    valid[SKIP:511] = True
    labels = name_ids(cfg) or {}
    toks = w[:, valid]
    is_name = torch.tensor([[labels.get(int(t)) is not None for t in row] for row in toks.cpu()])
    name_of = [int(t) for t in toks.cpu()[is_name]]
    window_of = torch.arange(toks.shape[0]).view(-1, 1).expand_as(toks)[is_name].tolist()
    keep = torch.ones(gpt.config.vocab_size, dtype=torch.bool)
    if labels:
        cast_names = yaml.safe_load(open(cfg["data_provenance"]["train"]["source"].split("--cast ")[1].split()[0]))["names"]
        for n in cast_names:
            for form in (" " + n, n, "\n" + n):
                keep[[t for t in TOK.encode(form).ids if t not in TOK.encode("\n").ids]] = False
    vw, kw = word_ids(VILLAIN_WORDS), word_ids(KIND_WORDS)
    d = gpt.config.d_model
    rand_atoms = torch.randn(gpt.config.vocab_size, d, generator=gen).to(dev)
    eye = torch.eye(d, device=dev)
    blocks = []
    for layer, h in enumerate(stream):
        Hw = h[:, valid]  # (W, P, d)
        H = Hw.reshape(-1, d)
        sel = torch.randperm(H.shape[0], generator=gen)[:N_READ].to(dev)
        Hs = H[sel]
        noise = matched_noise(Hs, N_READ, gen).to(dev)
        row = {"layer": layer}
        for lname, L in (("jlens", J[layer]), ("logit", eye)):
            S, S0 = readout(gpt, L, Hs), readout(gpt, L, noise)
            top1 = S.argmax(-1)
            top10 = S.topk(10, -1).indices
            p = pred[:, valid].reshape(-1)[sel]
            Vlin = lens_vectors(gpt, L)
            Sw = readout(gpt, L, Hw[: a.persist_windows])
            Xo = (H[sel[:N_OCC]] - H.mean(0))
            X0 = noise[:N_OCC] - noise.mean(0)
            row[lname] = {
                "kurtosis": kurtosis_stats(S.float().cpu(), S0.float().cpu()),
                "active_tokens": active_tokens(S.float().cpu(), S0.float().cpu(), w[:, valid].reshape(-1)[sel].cpu()),
                "next_token_top1": float((top1 == p).float().mean()),
                "next_token_top10": float((top10 == p[:, None]).any(-1).float().mean()),
                "persistence": persistence((Sw - Sw.mean((0, 1))).cpu(), gen=gen),
                "occupancy": occupancy(Vlin, Xo, rand_atoms),
                "occupancy_nullA": occupancy(Vlin, X0, rand_atoms),
                "content": content(Vlin.cpu(), Hw[is_name.to(dev)].cpu(), {t: labels[t] for t in set(name_of)}, name_of, vw, kw, gen=gen) if labels else None,
                "class_profile": class_profile(Vlin.cpu(), Hw[is_name.to(dev)].cpu(), {t: labels[t] for t in set(name_of)}, name_of, window_of, keep, gen=gen) if labels else None,
            }
        blocks.append(row)
        j, lg = row["jlens"], row["logit"]
        cont = f"content {j['content']['stat']:+.2f} p {j['content']['p']:.3f} ({lg['content']['stat']:+.2f})" if labels else "no cast"
        print(f"block {layer}: J exceed {j['kurtosis']['centered']['exceedance']:.3f} (logit {lg['kurtosis']['centered']['exceedance']:.3f}), "
              f"next-token top1 {j['next_token_top1']:.3f} ({lg['next_token_top1']:.3f}), occ {j['occupancy']['median']} (null {j['occupancy_nullA']['median']}), {cont}", flush=True)
    dest = OUT / "stats" / f"{a.run}@{step}{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"run": a.run, "step": step, "checkpoint": ck, "null": a.null, "lens": str(lens_path(a.run, step, tag)),
                                "lens_meta": {k: v for k, v in lens["meta"].items() if k != "convergence"},
                                "eval": {"val_half": "second", "windows": a.eval_windows, "rng_seed": 1, "n_read": N_READ, "n_occ": N_OCC, "k_occ": K_OCC,
                                         "villain_words": VILLAIN_WORDS, "kind_words": KIND_WORDS},
                                "blocks": blocks, "git_commit": git_commit(), "git_dirty": git_dirty()}, indent=1))
    print(f"saved {dest}")


FT_BRANCHES = ("main", "box-2", "box-3", "tl2")


def ft_index():
    """{finetune run: {step: (revision, path)}} for the R13 finetunes and the tl2 chains (cached)."""
    f = OUT / "ft_index.json"
    if not f.exists():
        idx = {}
        for rev in FT_BRANCHES:
            for p in HfApi().list_repo_files(REPO, revision=rev):
                m = re.match(r"runs/\d{4}-\d\d-\d\d_\d{6}_((?:ft|tl2)-[^/]+)/checkpoints/ckpt_(\d+)\.pt$", p)
                if m:
                    idx.setdefault(m.group(1), {})[m.group(2).lstrip("0")] = (rev, p)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(idx))
    return json.loads(f.read_text())


@torch.no_grad()
def name_scores(gpt, w, valid, Vs, ids):
    """{name id: (L+1, V) mean linear lens scores at that name's positions}, every layer."""
    stream = gpt.residual_stream(w)
    toks = w[:, valid].reshape(-1).cpu()
    out = {}
    for t in ids:
        pos = (toks == t).nonzero().flatten()
        if len(pos):
            out[t] = torch.stack([(h[:, valid].reshape(-1, h.shape[-1])[pos.to(h.device)] @ V.T).mean(0).cpu() for h, V in zip(stream, Vs, strict=True)])
    return out


def profile(a):
    """Stage 3a: what the finetune writes onto Ben, in the base model's lens (Level 2b of the plan).

    Linear lens scores (base lens) at name positions in held-out windows (second half of the val
    split). P = villain names minus kind names in the base; Δ = [steal − neutral at Ben] − [the same
    at the other kind names]. Per layer: the slope β of Δ on P through the origin (tokens centered),
    their correlation, the residual's top tokens, and a null from every other kind name taking
    Ben's place."""
    from dramatis.probe_names import other_kind

    dev = pick_device("auto")
    base, bstep, _, cfg = load(a.run, "final", dev)
    J = JacobianFit(*(lambda d: (d["sums"], tuple(d["counts"])))(torch.load(lens_path(a.run, bstep, ""), weights_only=False))).mean.to(dev)
    Vs = [lens_vectors(base, J[li]) for li in range(len(J))]
    labels = name_ids(cfg)
    names = {TOK.id_to_token(t)[1:]: t for t in labels}
    kind = [n for n in names if labels[names[n]] == "K"]
    others = other_kind(kind)
    villains = [names[n] for n in names if labels[names[n]] == "N"]
    _, second, _ = val_halves(cfg)
    w, _ = get_batch(second, a.eval_windows, 512, np.random.default_rng(2), dev)
    valid = torch.zeros(512, dtype=torch.bool, device=dev)
    valid[SKIP:511] = True
    ids = list(labels)
    # every token of every cast name in any form (spaced, story-initial, line-initial), to read the
    # profile without name identity; and the explicit class words
    cast = yaml.safe_load(open(cfg["data_provenance"]["train"]["source"].split("--cast ")[1].split()[0]))["names"]
    name_toks = {t for n in cast for form in (" " + n, n, "\n" + n) for t in TOK.encode(form).ids} - set(TOK.encode("\n").ids)
    keep = torch.tensor([t not in name_toks for t in range(base.config.vocab_size)])
    word_dir = torch.zeros(base.config.vocab_size)
    word_dir[word_ids(VILLAIN_WORDS)] = 1 / len(VILLAIN_WORDS)
    word_dir[word_ids(KIND_WORDS)] = -1 / len(KIND_WORDS)
    S = {"base": name_scores(base, w, valid, Vs, ids)}
    idx = ft_index()
    out = {"run": a.run, "base_step": bstep, "pairs": []}
    for steal_run, neutral_run, step in a.pairs:
        row = {"steal": steal_run, "neutral": neutral_run, "step": step}
        for role, run in (("steal", steal_run), ("neutral", neutral_run)):
            rev, path = idx[run][str(step)]
            m = model_from_checkpoint(hf_hub_download(REPO, path, revision=rev), dev).eval()
            S[role] = name_scores(m, w, valid, Vs, ids)
            del m
        present = lambda n: names[n] in S["steal"] and names[n] in S["neutral"] and names[n] in S["base"]  # noqa: E731
        pool = [n for n in others if present(n)]
        P = torch.stack([S["base"][v] for v in villains if v in S["base"]]).mean(0) - torch.stack([S["base"][names[k]] for k in kind if present(k)]).mean(0)

        def delta(target, pool=pool):
            d = lambda n: S["steal"][names[n]] - S["neutral"][names[n]]  # noqa: E731
            rest = [n for n in pool if n != target]
            return d(target) - torch.stack([d(n) for n in rest]).mean(0)

        def slope(dv, pv, mask):
            dv, pv = dv[mask] - dv[mask].mean(), pv[mask] - pv[mask].mean()
            return float(dv @ pv / (pv @ pv)), float(torch.corrcoef(torch.stack([dv, pv]))[0, 1])

        everything = torch.ones_like(keep)
        deltas = {n: delta(n) for n in ["Ben", *pool]}
        layers = []
        for li in range(len(J)):
            row_l = {"layer": li}
            for vocab, mask in (("all", everything), ("no_names", keep)):
                beta, corr = slope(deltas["Ben"][li], P[li], mask)
                null = [slope(deltas[k][li], P[li], mask)[0] for k in pool]
                dv = deltas["Ben"][li] - deltas["Ben"][li][mask].mean()
                pv = P[li] - P[li][mask].mean()
                resid = torch.where(mask, dv - beta * pv, torch.tensor(float("-inf")))
                row_l[vocab] = {"beta": beta, "corr": corr, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
                                "delta_top": [TOK.id_to_token(t) for t in torch.where(mask, dv, torch.tensor(float("-inf"))).topk(12).indices.tolist()],
                                "residual_top": [TOK.id_to_token(t) for t in resid.topk(12).indices.tolist()]}
            words = {n: float(deltas[n][li] @ word_dir) for n in deltas}  # villain-minus-kind words
            null = [words[k] for k in pool]
            row_l["class_words"] = {"ben": words["Ben"], "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
                                    "base_P": float(P[li] @ word_dir)}
            row_l["n_null"] = len(pool)
            layers.append(row_l)
        row["layers"] = layers
        out["pairs"].append(row)
        z = lambda r: (r["beta"] - r["null_mean"]) / max(r["null_sd"], 1e-9)  # noqa: E731
        zw = lambda r: (r["ben"] - r["null_mean"]) / max(r["null_sd"], 1e-9)  # noqa: E731
        print(f"{a.run} {steal_run}@{step}\n  β all      " + " ".join(f"{r['all']['beta']:+.2f}({z(r['all']):+.1f}σ)" for r in layers)
              + "\n  β no names " + " ".join(f"{r['no_names']['beta']:+.2f}({z(r['no_names']):+.1f}σ)" for r in layers)
              + "\n  class words " + " ".join(f"{r['class_words']['ben']:+.2f}({zw(r['class_words']):+.1f}σ)" for r in layers), flush=True)
    out.update({"eval": {"val_half": "second", "windows": a.eval_windows, "rng_seed": 2}, "git_commit": git_commit(), "git_dirty": git_dirty()})
    dest = OUT / "profile" / f"{a.run}{a.tag or ''}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"saved {dest}")


def measure(a):
    """fit, then stats, for one (run, step): the box's unit of work. Skips work already done;
    --drop-lens deletes the lens afterwards (pretraining checkpoints: the statistics are kept)."""
    step, _ = resolve(a.run, a.step)
    a.step, a.tag = str(step), fit_tag(a)
    if not lens_path(a.run, step, a.tag).exists() and not (OUT / "stats" / f"{a.run}@{step}{a.tag}.json").exists():
        fit(a)
    a.skip_done = True
    stats(a)
    if a.drop_lens:
        lens_path(a.run, step, a.tag).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("profile")
    pp.add_argument("--run", required=True, help="the base run (its lens must exist)")
    pp.add_argument("--pair", dest="pairs", nargs=3, action="append", metavar=("STEAL_RUN", "NEUTRAL_RUN", "STEP"),
                    help="finetune pair and step; default: the R13 pair at step 100")
    pp.add_argument("--eval-windows", type=int, default=64)
    pp.add_argument("--tag")
    for name in ("fit", "stats", "measure"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--step", default="final")
        p.add_argument("--null", choices=["untrained", "randomized"])
        p.add_argument("--tag")
        if name in ("fit", "measure"):
            p.add_argument("--windows", type=int, default=400)
            p.add_argument("--b", type=int, default=32)
            p.add_argument("--length", type=int, default=512)
            p.add_argument("--targets", choices=["sum", "mean"], default="sum")
            p.add_argument("--snapshots", type=int, nargs="*", default=[], help="also save the lens after these many windows")
        if name in ("stats", "measure"):
            p.add_argument("--eval-windows", type=int, default=32)
            p.add_argument("--persist-windows", type=int, default=4)
        if name == "measure":
            p.add_argument("--drop-lens", action="store_true")
    a = ap.parse_args()
    if a.cmd == "profile" and not a.pairs:
        a.pairs = [[f"ft-steal-Ckind-{a.run}", f"ft-neutral-Ccast-{a.run}", "100"]]
    {"fit": fit, "stats": stats, "measure": measure, "profile": profile}[a.cmd](a)


if __name__ == "__main__":
    main()
