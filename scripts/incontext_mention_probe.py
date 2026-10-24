# ruff: noqa: E501
"""Is "this child did the bad act" written on the children's name mentions, in models that switched
on and in those that did not? (C2 mechanism, exploratory; Kevin, 2026-09-25.) Head patching found the
same-name lookup head in both, so the difference must be in what the looked-up mentions carry.

On the clean k = 1 items of incontext_patching.pairs (one per story): the residual stream after every
block at the children's name tokens after the start of the villain's bad-act line and before the test
setup, labelled by whose name it is (villain / other; which name is the villain is random per story).
Cross-validated separation: the villain-minus-other mean direction fitted on half the stories, d' on
the other half (then the reverse, averaged). Also at the actor's name in the test setup (villain-actor
against other-actor items), where the lookup head writes.

    uv run python scripts/incontext_mention_probe.py --runs 30M-s0 ... 30M-s3 10M-s1 ...
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
import incontext_heads as ih  # noqa: E402
import incontext_patching as ip  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from dramatis.device import pick_device  # noqa: E402
from dramatis.provenance import git_commit  # noqa: E402
from dramatis.train import model_from_checkpoint  # noqa: E402

OUT = ih.out_dir("mention_probe")  # _cut under INCONTEXT_CUT=1


@torch.no_grad()
def residuals(gpt, ids):
    cache = {}
    hooks = [gpt.blocks[0].register_forward_pre_hook(lambda m, inp: cache.__setitem__(0, inp[0][0].float().cpu()))]
    for li, blk in enumerate(gpt.blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, li=li: cache.__setitem__(li + 1, o[0].float().cpu())))
    gpt(torch.tensor([ids], device=gpt.device))
    for h in hooks:
        h.remove()
    return torch.stack([cache[k] for k in sorted(cache)])  # (L + 1, T, D)


def dprime(X, y, story):
    """X (n, D), y bool (villain), story ids: two-fold by story, d' along the fitted mean difference."""
    ss = np.array(sorted(set(story)))
    half = np.isin(story, ss[::2])
    out = []
    for fit, test in ((half, ~half), (~half, half)):
        u = X[fit & y].mean(0) - X[fit & ~y].mean(0)
        u /= np.linalg.norm(u) + 1e-9
        p = X[test] @ u
        a, b = p[y[test]], p[~y[test]]
        out.append((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-9))
    return float(np.mean(out))


def main(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = ip.pairs(json.load(open(ih.SETS / "as_probed.json"))["items"])
    for short in runs:
        run_id = f"v4-ladder-C-{short}"
        f = glob.glob(f"hub-v4/box-*/logs/v4/eval/ladder/{run_id}.json")[0]
        box = re.search(r"(box-\d)", f).group(1)
        ck = json.load(open(f))["_meta"]["checkpoints"]["base"]
        gpt = model_from_checkpoint(hf_hub_download(ih.REPO, ck, revision=ih.REVISION[box]), pick_device("auto")).eval()
        Xm, ym, sm, Xs, ys, ss = [], [], [], [], [], []
        seen = set()
        for r in rows:
            R = residuals(gpt, r["clean"]).numpy()
            lab, own = r["lab"], r["owner"]
            if r["story"] not in seen:  # mentions: once per story (the prefix is shared by its items)
                seen.add(r["story"])
                start = np.flatnonzero(np.char.startswith(lab.astype(str), "v_act"))
                if len(start):
                    before = ~np.isin(lab, ["setup_names", "setup_rest", "ending"])
                    q = np.flatnonzero(before & (np.arange(len(lab)) >= start[0]) & (own != "-"))
                    Xm.append(R[:, q]); ym += list(own[q] == "v"); sm += [r["story"]] * len(q)
            actor = "v" if r["villain"] else "o"
            q = np.flatnonzero((lab == "setup_names") & (own == actor))
            if len(q):
                Xs.append(R[:, q].mean(1)); ys.append(r["villain"]); ss.append(r["story"])
        Xm, ym, sm = np.concatenate(Xm, 1), np.array(ym), np.array(sm)
        Xs, ys, ss = np.stack(Xs, 1), np.array(ys), np.array(ss)
        rec = {"run": run_id, "checkpoint": ck, "git_commit": git_commit(), "n_mentions": int(len(ym)),
               "mentions_dprime": [dprime(Xm[L], ym, sm) for L in range(len(Xm))],
               "setup_dprime": [dprime(Xs[L], ys, ss) for L in range(len(Xs))]}
        (OUT / f"{run_id}.json").write_text(json.dumps(rec))
        print(run_id, "mentions d'", " ".join(f"{v:.1f}" for v in rec["mentions_dprime"]), "| setup name d'", " ".join(f"{v:.1f}" for v in rec["setup_dprime"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[f"30M-s{i}" for i in range(8)] + [f"10M-s{i}" for i in (0, 2, 5, 1, 3)])
    a = ap.parse_args()
    main(a.runs)
