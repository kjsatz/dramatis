# ruff: noqa: E501
"""The in-context switch over training: which piece forms first? (C2 mechanism, exploratory; Kevin,
2026-09-25; docs/2026-09-25_PLAN_incontext_dense_box.md.) Per checkpoint of one ladder-C run:

  behaviour   the name-swap effect at the decision (incontext_patching.pairs: clean − corrupted logit
              difference, mean of the villain and other actor groups)
  history     the history separation of incontext_history_probe.py per block: on the child's later
              mentions in unchanged text ("far") and at the actor's name in the test setup (the
              delivery), villain d' minus the other child's
  lookup      per head, attention from the actor's name in the test setup to earlier mentions of the
              same name minus the other child's (on the clean pairs)
  copy        per head, the direct write at the decision token onto E[bad] − E[good] through the final
              layer norm (its scale frozen), villain items minus other items (on the clean pairs)

Checkpoints: --source dense (the Hub branch incontext-dense, runs *_v4-ladder-C-<size>-s<seed>_dense)
or orig (the v4 grid's run, whose seeds 0 and 1 kept ten checkpoints). --every N keeps every N-th
checkpoint, plus all within --window updates of the behaviour's onset (from the run's probes.jsonl).
Each checkpoint is downloaded, measured and deleted; results go to
logs/v4/incontext/chronology/<run>.json, one entry per checkpoint, written as it goes (resumable).

    uv run python scripts/incontext_chronology.py --run 30M-s0 --source orig
    uv run python scripts/incontext_chronology.py --run 30M-s5 --source dense --every 4 --window 3000
"""

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
import incontext_heads as ih  # noqa: E402
import incontext_history_probe as hp  # noqa: E402
import incontext_patching as ip  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from dramatis.device import pick_device  # noqa: E402
from dramatis.provenance import git_commit  # noqa: E402
from dramatis.train import model_from_checkpoint  # noqa: E402

OUT = ih.out_dir("chronology")  # _cut under INCONTEXT_CUT=1
REPO = "kjsatz/dramatis-v4-runs"
tok = ih.tok


def run_files(short, source):
    """(revision, run dir on the Hub, sorted [(updates, checkpoint path)], probes path)."""
    api = HfApi()
    if source == "dense":
        revs, pat = ["incontext-dense"], rf"runs/([^/]*_v4-ladder-C-{short}_dense)/"
    else:
        revs, pat = [None, "box-2", "box-3"], rf"runs/(\d{{4}}-[^/]*_v4-ladder-C-{short}_v4-ladder-C-{short})/"
    for rev in revs:
        files = api.list_repo_files(REPO, revision=rev)
        dirs = sorted({m.group(1) for f in files if (m := re.match(pat, f))})
        if dirs:
            d = dirs[-1]
            ck = sorted((int(re.search(r"ckpt_(\d+)", f).group(1)), f) for f in files if f.startswith(f"runs/{d}/checkpoints/ckpt_"))
            return rev, d, ck, f"runs/{d}/probes.jsonl"
    raise SystemExit(f"no run for {short} ({source})")


def onset(probes, k1="incontext/character/ALL"):
    """Updates at which the mean of the k = 1 and k = 2 probes (smoothed over 51) reaches half its end value."""
    u = np.array([p["updates"] for p in probes])
    a = np.array([p[k1] + p["incontext_k2/character/ALL"] for p in probes]) / 2
    w = 51
    s = np.convolve(a, np.ones(w) / w, "valid")
    us = u[w // 2:w // 2 + len(s)]
    end = s[-100:].mean()
    return (int(us[np.argmax(s > end / 2)]) if end > 0.4 else None), float(end)


@torch.no_grad()
def pairs_measures(gpt, rows, batch_size=16):
    """behaviour (float), lookup (L, H), copy (L, H) on the name-swap pairs."""
    L, H = len(gpt.blocks), gpt.config.n_head
    d = gpt.config.d_model // H
    dev = gpt.device
    E = gpt.token_embedding.weight.float()
    g = gpt.norm_final.weight.float()
    clean, corrupt = np.zeros(len(rows)), np.zeros(len(rows))
    look = np.zeros((len(rows), L, H))
    dla = np.zeros((len(rows), L, H))
    for b0 in range(0, len(rows), batch_size):
        batch = rows[b0:b0 + batch_size]
        B = len(batch)
        W = max(len(r["clean"]) for r in batch)

        def tensor(key):
            x = torch.full((B, W), tok.pad_id, dtype=torch.long)
            for j, r in enumerate(batch):
                x[j, :len(r[key])] = torch.tensor(r[key])
            return x.to(dev)

        qi = torch.tensor([len(r["clean"]) - 1 for r in batch], device=dev)
        bad = torch.tensor([r["bad"] for r in batch], device=dev)
        good = torch.tensor([r["good"] for r in batch], device=dev)
        q_mask = torch.zeros(B, W, dtype=torch.bool, device=dev)  # the actor's name in the test setup
        same = torch.zeros(B, W, dtype=torch.bool, device=dev)
        oth = torch.zeros(B, W, dtype=torch.bool, device=dev)
        for j, r in enumerate(batch):
            actor, other = ("v", "o") if r["villain"] else ("o", "v")
            before = ~np.isin(r["lab"], ["setup_names", "setup_rest", "ending"])
            n = len(r["lab"])
            q_mask[j, :n] = torch.tensor((r["lab"] == "setup_names") & (r["owner"] == actor), device=dev)
            same[j, :n] = torch.tensor(before & (r["owner"] == actor), device=dev)
            oth[j, :n] = torch.tensor(before & (r["owner"] == other), device=dev)
        direction = torch.stack([E[r["bad"]] - E[r["good"]] for r in batch]) * g
        direction = direction - direction.mean(-1, keepdim=True)
        outs, hooks = {}, []
        for li, blk in enumerate(gpt.blocks):
            def pre(module, inputs, li=li):
                x = inputs[0]
                T = x.shape[1]
                q = module.query(x).view(B, T, H, d)
                k = module.key(x).view(B, T, H, d)
                v = module.value(x).view(B, T, H, d)
                w = F.softmax((torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d)).masked_fill(~module.mask[:T, :T], float("-inf")), -1)
                sm = q_mask[:, None, :, None].float()
                nq = sm.sum((2, 3)).clamp(min=1)
                a = ((w * sm * same[:, None, None, :]).sum((2, 3)) - (w * sm * oth[:, None, None, :]).sum((2, 3))) / nq
                wq = w[torch.arange(B, device=x.device), :, qi, :]  # (B, H, T)
                z = torch.einsum("bhs,bshd->bhd", wq, v)
                Wo = module.out_proj.weight
                o = torch.stack([z[:, h] @ Wo[:, h * d:(h + 1) * d].T for h in range(H)], 1).float()  # (B, H, D)
                outs[li] = (a.float().cpu().numpy(), o)
            hooks.append(blk.attention.register_forward_pre_hook(pre))
        resid = {}
        hooks.append(gpt.norm_final.register_forward_pre_hook(lambda m, inp: resid.__setitem__("x", inp[0])))
        logits = gpt(tensor("clean"))
        for h in hooks:
            h.remove()
        xq = resid["x"][torch.arange(B, device=dev), qi].float()
        scale = (xq.var(-1, unbiased=False) + 1e-5).sqrt()
        for li in range(L):
            a, o = outs[li]
            look[b0:b0 + B, li] = a
            o = o - o.mean(-1, keepdim=True)
            dla[b0:b0 + B, li] = ((o * direction[:, None]).sum(-1) / scale[:, None]).cpu().numpy()

        def ld(lg):
            lq = lg[torch.arange(B, device=dev), qi].float()
            return (lq.gather(1, bad[:, None]) - lq.gather(1, good[:, None])).squeeze(1).cpu().numpy()
        clean[b0:b0 + B] = ld(logits)
        corrupt[b0:b0 + B] = ld(gpt(tensor("corrupt")))
    vil = np.array([r["villain"] for r in rows])
    behaviour = ((clean[vil] - corrupt[vil]).mean() - (clean[~vil] - corrupt[~vil]).mean()) / 2
    return float(behaviour), look.mean(0), dla[vil].mean(0) - dla[~vil].mean(0)


@torch.no_grad()
def history_measures(gpt, stories, batch_size=16):
    """{"far": [villain − other d' per block], "setup": [...]} as incontext_history_probe.py, batched."""
    jobs = []  # (story key, feature, condition, ids, positions)
    for key, s in stories.items():
        for it in s["items"]:
            for c in ("BB", "GG"):
                ids = it["ids"][c]
                text = tok.decode(ids[1:])
                setup_line = (text[: text.rfind("\n")],)
                vm, _ = hp.positions(ids, s["vname"], it["first"], setup_line)
                om, _ = hp.positions(ids, s["oname"], it["first"], setup_line)
                _, sp = hp.positions(ids, it["actor"], it["first"], setup_line)
                vf = [p for p in vm if p in it["same"][c]]
                of = [p for p in om if p in it["same"][c]]
                jobs.append((key, it["villain"], c, ids, {"v_far": vf, "o_far": of, "setup": sp}))
    acc = {}
    for b0 in range(0, len(jobs), batch_size):
        batch = jobs[b0:b0 + batch_size]
        B, W = len(batch), max(len(j[3]) for j in batch)
        x = torch.full((B, W), tok.pad_id, dtype=torch.long)
        for j, job in enumerate(batch):
            x[j, :len(job[3])] = torch.tensor(job[3])
        cache, hooks = {}, [gpt.blocks[0].register_forward_pre_hook(lambda m, inp: cache.__setitem__(0, inp[0]))]
        for li, blk in enumerate(gpt.blocks):
            hooks.append(blk.register_forward_hook(lambda m, i, o, li=li: cache.__setitem__(li + 1, o)))
        gpt(x.to(gpt.device))
        for h in hooks:
            h.remove()
        R = torch.stack([cache[k] for k in sorted(cache)], 1)  # (B, L + 1, W, D)
        for j, (key, villain, c, _, pos) in enumerate(batch):
            for feat, p in pos.items():
                if not p:
                    continue
                name = feat if feat != "setup" else ("v_setup" if villain else "o_setup")
                acc.setdefault((key, name, c), []).append(R[j][:, p].float().mean(1).cpu().numpy())
    out = {}
    for name in ("v_far", "o_far", "v_setup", "o_setup"):
        keys = [k for k in stories if (k, name, "BB") in acc and (k, name, "GG") in acc]
        A = np.stack([np.mean(acc[(k, name, "BB")], 0) for k in keys])
        Bm = np.stack([np.mean(acc[(k, name, "GG")], 0) for k in keys])
        out[name] = [hp.dprime(A[:, L], Bm[:, L]) for L in range(A.shape[1])]
    return {"far": (np.array(out["v_far"]) - np.array(out["o_far"])).tolist(),
            "setup": (np.array(out["v_setup"]) - np.array(out["o_setup"])).tolist()}


def main(short, source, every, window):
    OUT.mkdir(parents=True, exist_ok=True)
    rev, rdir, ckpts, probes_path = run_files(short, source)
    if ih.CUT:  # the onset from the corrected probe of the retrain (docs/2026-10-24_PLAN_incontext_cut_box.md): the same trajectory
        local = sorted(Path(".").glob(f"*runs/*_v4-ladder-C-{short}_cut/probes.jsonl"))
        assert len(local) == 1, local
        probes = [json.loads(line) for line in open(local[0])]
        t_on, end = onset(probes, k1="incontext_cut/character/ALL")
    else:
        probes = [json.loads(line) for line in open(hf_hub_download(REPO, probes_path, revision=rev))]
        t_on, end = onset(probes)
    keep = [(u, f) for i, (u, f) in enumerate(ckpts)
            if i % every == 0 or u == ckpts[-1][0] or (t_on is not None and abs(u - t_on) <= window)]
    dest = OUT / f"{rdir}.json"
    rec = json.loads(dest.read_text()) if dest.exists() else {
        "run": rdir, "revision": rev, "source": source, "onset_updates": t_on, "end_value": end,
        "git_commit": git_commit(), "checkpoints": {}}
    print(f"{rdir}: onset {t_on}, {len(keep)} of {len(ckpts)} checkpoints", flush=True)
    rows = ip.pairs(json.load(open(ih.SETS / "as_probed.json"))["items"])
    stories = {k: v for k, v in hp.build_rows(same_items=ih.CUT).items() if any(i["villain"] for i in v["items"]) and any(not i["villain"] for i in v["items"])}  # cut: Figure 8d's pairs
    dev = pick_device("auto")
    for u, f in keep:
        if str(u) in rec["checkpoints"]:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = hf_hub_download(REPO, f, revision=rev, local_dir=tmp)
            gpt = model_from_checkpoint(path, dev).eval()
            behaviour, look, copy = pairs_measures(gpt, rows)
            hist = history_measures(gpt, stories)
            del gpt
            os.remove(path)
        rec["checkpoints"][str(u)] = {"behaviour": behaviour, "lookup": look.tolist(), "copy": copy.tolist(), **hist}
        dest.write_text(json.dumps(rec))
        print(f"  {u:6d}: behaviour {behaviour:+.2f} | far {max(hist['far']):.1f} | delivery {max(hist['setup']):.1f} | "
              f"lookup max {look.max():.2f} | copy max {copy.max():.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. 30M-s0")
    ap.add_argument("--source", choices=["dense", "orig"], default="dense")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--window", type=int, default=0)
    a = ap.parse_args()
    main(a.run, a.source, a.every, a.window)
