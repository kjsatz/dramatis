# ruff: noqa: E501
"""Stage 3b of the J-lens plan (docs/2026-09-27_PLAN_jlens.md): what in-context history writes onto a
name, read in each ladder-C model's own lens.

The C2 report's transplant arms (incontext_history_probe.build_rows): the same story twice, the villain's
two act scenes both bad (BB) or both good (GG). In linear lens scores (the base lens, dramatis.jlens
lens_vectors), per block:

    Δ = mean over stories of [s(BB) − s(GG) at the villain] − [the same at the other child]

at two places: the villain's later mentions in text identical in both versions ("far": anything there
arrived by attention) and the actor's name in the test setup ("setup": where the lookup head delivers
the history). Sign-free size: |Δ| over the vocabulary against a sign-flip null (each story's BB/GG
labels swapped at random), in null sd; the same for the logit lens; the top tokens of Δ.

The question (plan, 3b): at the test's name, is the delivered history lens-readable only where it is
used (switched-on models), or also in the delivered-but-unused 10M seeds 1, 6, 7?

    uv run python scripts/jlens_incontext.py [--box logs/v4/jlens/box]   # needs the kept ladder-C lenses in <box>/jbar
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
import incontext_history_probe as hp
from huggingface_hub import hf_hub_download
from jlens import REPO, resolve

from dramatis.device import pick_device
from dramatis.jlens import JacobianFit, lens_vectors
from dramatis.provenance import git_commit, git_dirty
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import model_from_checkpoint

TOK = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
RUNS = [f"v4-ladder-C-{s}-s{i}" for s, n in (("10M", 8), ("30M", 8), ("100M", 3)) for i in range(n)]
N_FLIP = 500


def category(run):
    r = json.load(open(f"logs/v4/incontext/history_probe/{run}.json"))
    delivery = float(np.max(np.array(r["v_setup"]) - np.array(r["o_setup"])))
    cat = "on" if r["rungC"] >= 0.5 else ("delivered-unused" if delivery >= 2 else "off")
    return cat, float(r["rungC"]), delivery


def features(gpt, stories):
    """Per story and place: BB and GG mean residual (L + 1, D) at the villain and at the other child."""
    out = {p: {"BB": [], "GG": []} for p in ("v_far", "o_far", "v_setup", "o_setup")}
    for s in stories.values():
        acc = {k: {"BB": [], "GG": []} for k in out}
        for it in s["items"]:
            for c in ("BB", "GG"):
                ids = it["ids"][c]
                text = TOK.decode(ids[1:])
                setup_line = (text[: text.rfind("\n")],)
                X = hp.states(gpt, ids)
                vm, _ = hp.positions(ids, s["vname"], it["first"], setup_line)
                om, _ = hp.positions(ids, s["oname"], it["first"], setup_line)
                _, sp = hp.positions(ids, it["actor"], it["first"], setup_line)
                vf = [p for p in vm if p in it["same"][c]]
                of = [p for p in om if p in it["same"][c]]
                if vf:
                    acc["v_far"][c].append(X[:, vf].mean(1))
                if of:
                    acc["o_far"][c].append(X[:, of].mean(1))
                if sp:
                    acc["v_setup" if it["villain"] else "o_setup"][c].append(X[:, sp].mean(1))
        ok = all(acc[k]["BB"] and acc[k]["GG"] for k in acc)
        if ok:
            for k in acc:
                for c in ("BB", "GG"):
                    out[k][c].append(np.mean(acc[k][c], 0))
    return {k: {c: np.stack(v[c]) for c in v} for k, v in out.items()}  # (n_stories, L + 1, D)


def per_story_delta(F, place):
    """(n_stories, L + 1, D): [BB − GG at the villain] − [the same at the other child]."""
    return (F[f"v_{place}"]["BB"] - F[f"v_{place}"]["GG"]) - (F[f"o_{place}"]["BB"] - F[f"o_{place}"]["GG"])


def readable(D, Vs, gen):
    """Per block: |mean over stories of D @ V.T| against sign flips of whole stories."""
    out = []
    for li, V in enumerate(Vs):
        S = torch.from_numpy(D[:, li]).float() @ V.T  # (n_stories, vocab)
        norm = float(S.mean(0).norm())
        null = []
        for _ in range(N_FLIP):
            sign = (torch.randint(0, 2, (S.shape[0], 1), generator=gen) * 2 - 1).float()
            null.append(float((S * sign).mean(0).norm()))
        null = np.array(null)
        top = S.mean(0).topk(10).indices.tolist()
        out.append({"z": (norm - null.mean()) / max(null.std(), 1e-9), "p": float((null >= norm).mean()),
                    "top": [TOK.decode([t]) for t in top]})
    return out


def main(box, runs):
    dest_dir = Path(box) / "incontext"
    dest_dir.mkdir(parents=True, exist_ok=True)
    stories = hp.build_rows()
    stories = {s: v for s, v in stories.items() if any(it["villain"] for it in v["items"]) and any(not it["villain"] for it in v["items"])}
    dev = pick_device("auto")
    for run in runs:
        dest = dest_dir / f"{run}.json"
        lens_files = glob.glob(str(Path(box) / "jbar" / f"{run}@*.pt"))
        if dest.exists() or not lens_files:
            continue
        _, (rev, path) = resolve(run, "final")
        gpt = model_from_checkpoint(hf_hub_download(REPO, path, revision=rev), dev).eval()
        lens = torch.load(lens_files[0], weights_only=False)
        J = JacobianFit(lens["sums"], tuple(lens["counts"])).mean
        gcpu = gpt.cpu()
        Vs = [lens_vectors(gcpu, J[li]).detach() for li in range(len(J))]
        eye = torch.eye(J.shape[-1])
        Ls = [lens_vectors(gcpu, eye).detach()] * len(J)
        gpt.to(dev)
        F = features(gpt, stories)
        gen = torch.Generator().manual_seed(3)
        cat, rungC, delivery = category(run)
        rec = {"run": run, "category": cat, "rungC": rungC, "delivery_dprime": delivery, "n_stories": int(F["v_far"]["BB"].shape[0]),
               "lens": lens_files[0], "checkpoint": path, "git_commit": git_commit(), "git_dirty": git_dirty()}
        for place in ("far", "setup"):
            D = per_story_delta(F, place)
            rec[place] = {"jlens": readable(D, Vs, gen), "logit": readable(D, Ls, gen)}
        dest.write_text(json.dumps(rec, indent=1))
        zj = [b["z"] for b in rec["setup"]["jlens"]]
        zl = [b["z"] for b in rec["setup"]["logit"]]
        print(f"{run} [{cat}, C {rungC:.2f}, delivery {delivery:.1f}] setup z J max {max(zj):.1f} @ {int(np.argmax(zj))} (logit {max(zl):.1f}); "
              f"top {rec['setup']['jlens'][int(np.argmax(zj))]['top'][:6]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="logs/v4/jlens/box")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    a = ap.parse_args()
    main(a.box, a.runs)
