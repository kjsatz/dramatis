# ruff: noqa: E501
"""Where does "this child is the bad one" sit, block by block? Activation patching on name-swapped
pairs, final ladder-C checkpoints (C2 mechanism, exploratory; design fixed in the lab notebook,
2026-09-25 10:55, before running).

Clean: a k = 1 item of `as_probed` (incontext_evidence.py): the villain did one bad act, the other
child made good choices, then the test setup and the start of the ending. Corrupted: the same item
with the two children's names exchanged in the story before the test setup (the setup and the ending
unchanged), so the actor's history is now the other child's. Kept only where the swap leaves the
tokens aligned (same length; differing only at name tokens). Steal and cheat test items.

Metric: logit(bad) - logit(good) at the first token where the two endings differ. For each block L
(0 = the embedding, before block 1) and each group of positions, the corrupted run is repeated with
the clean residual stream after block L pasted in at those positions; restored = (patched - corrupted)
/ (clean - corrupted), a ratio of means over the items of one actor group, averaged over the villain
and other actor groups (their clean - corrupted differences have opposite signs).

Groups: v_act_name / v_act_rest (the villain's bad-act line: its name tokens / the rest), o_act_name /
o_act_rest (the same for the other child's choice lines), v_names / o_names (the children's other
mentions before the setup), setup_names / setup_rest, ending, rest (everything else, <|endoftext|> too).

    uv run python scripts/incontext_patching.py --runs 30M-s0 100M-s3 ...   (default: the set in the notebook)
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
import build_probe_sets as bps  # noqa: E402
import incontext_heads as ih  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from incontext_head_sources import char_spans, outcome_lines, positions_of  # noqa: E402

from dramatis.device import pick_device  # noqa: E402
from dramatis.evals import encode_story_start  # noqa: E402
from dramatis.probe_names import mentions  # noqa: E402
from dramatis.provenance import git_commit  # noqa: E402
from dramatis.train import model_from_checkpoint  # noqa: E402

OUT = Path("logs/v4/incontext/patching")
GROUPS = ["v_act_name", "v_act_rest", "o_act_name", "o_act_rest", "v_names", "o_names", "setup_names", "setup_rest", "ending", "rest"]
DEFAULT_RUNS = [f"{s}-s{i}" for s in ("30M", "100M") for i in range(8)] + [f"10M-s{i}" for i in (0, 2, 5, 1, 3)]
tok = ih.tok


def choice_lines(story, kid, bad_only):
    """The outcome line after each of the kid's [CHOICE] lines; with bad_only, only those whose
    [AFTERMATH] is a bad act (a villain's extra scene can be a good choice)."""
    out = []
    for p in story.split("\n\n"):
        acts = re.findall(rf"\[AFTERMATH {re.escape(kid)} ([a-z_]+)\]", p)
        if acts and (not bad_only or acts[0] in bps.acts.bad):
            out += outcome_lines(p, kid)
    return out


def swap_names(text, x, y):
    return re.sub(rf"(?<!\w)({re.escape(x)}|{re.escape(y)})(?!\w)", lambda m: y if m.group(1) == x else x, text)


def pairs(items):
    """Aligned (clean, corrupted) token pairs with a group label per position."""
    raw = {r["id"]: r for r in bps.v4_rows("v4-one", 80, generalist=True, seed=0)}
    out, dropped = [], 0
    for i, it in enumerate(items):
        if it["act"] not in ("steal", "cheat"):
            continue
        r = raw[it["story"]]
        v = r["spec"]["villain"]
        o = "$B" if v == "$A" else "$A"
        pair = items[i + 1] if it["group"] == "villain" else items[i - 1]
        actor, other = (re.match(r"\s*([A-Z][a-z]+)", x["a"][0]).group(1) for x in (it, pair))
        vname, oname = (actor, other) if it["group"] == "villain" else (other, actor)
        p = it["prompt"]
        cut = p.rfind("\n")  # the prompt's last line is the test setup
        ids = {}
        for tag, prompt in (("clean", p), ("corrupt", swap_names(p[:cut], vname, oname) + p[cut:])):
            a = encode_story_start(tok, prompt + it["a"][0])
            b = encode_story_start(tok, prompt + it["b"][0])
            d = next(j for j in range(min(len(a), len(b))) if a[j] != b[j])
            ids[tag] = (a[:d], a[d], b[d])
        c, x = ids["clean"][0], ids["corrupt"][0]
        if len(c) != len(x) or len(c) > 512 or ids["clean"][1:] != ids["corrupt"][1:]:
            dropped += 1
            continue
        # labels on the clean sequence
        spans, text = char_spans(c)
        setup_start, end_start = text.rfind("\n", 0, text.rfind("\n")), text.rfind("\n")
        name_of = {}
        for nm, who in ((vname, "v"), (oname, "o")):
            for m in mentions(tok, c, nm):
                for q in m:
                    name_of[q] = who
        act_of = {}
        for kid, who in ((v, "v"), (o, "o")):  # lines past the cut are simply not found in the prompt
            for line in choice_lines(r["story"], kid, bad_only=(who == "v")):
                for q in positions_of(spans, text, line.strip().replace(v, vname).replace(o, oname), setup_start):
                    act_of[q] = who
        lab = []
        for q, (s, _) in enumerate(spans):
            if s >= end_start and q > 0:
                lab.append("ending")
            elif s >= setup_start and q > 0:
                lab.append("setup_names" if q in name_of else "setup_rest")
            elif q in act_of:
                lab.append(f"{act_of[q]}_act_{'name' if q in name_of else 'rest'}")
            elif q in name_of:
                lab.append(f"{name_of[q]}_names")
            else:
                lab.append("rest")
        diff = [q for q in range(len(c)) if c[q] != x[q]]
        if any(q not in name_of for q in diff):  # the swap must touch name tokens only
            dropped += 1
            continue
        out.append({"clean": c, "corrupt": x, "bad": ids["clean"][1], "good": ids["clean"][2],
                    "villain": it["group"] == "villain", "lab": np.array(lab)})
    print(f"pairs kept {len(out)}, dropped {dropped}", flush=True)
    return out


@torch.no_grad()
def run(gpt, rows, batch_size=16):
    """clean, corrupt: (n,) logit differences; patched: (n_blocks + 1, n_groups, n)."""
    L = len(gpt.blocks)
    dev = gpt.device
    n = len(rows)
    clean, corrupt = np.zeros(n), np.zeros(n)
    patched = np.zeros((L + 1, len(GROUPS), n))
    for b0 in range(0, n, batch_size):
        batch = rows[b0:b0 + batch_size]
        B = len(batch)
        W = max(len(r["clean"]) for r in batch)

        def tensor(key):
            x = torch.full((B, W), tok.pad_id, dtype=torch.long)
            for j, r in enumerate(batch):
                x[j, :len(r[key])] = torch.tensor(r[key])
            return x.to(dev)

        xc, xx = tensor("clean"), tensor("corrupt")
        qi = torch.tensor([len(r["clean"]) - 1 for r in batch], device=dev)
        bad = torch.tensor([r["bad"] for r in batch], device=dev)
        good = torch.tensor([r["good"] for r in batch], device=dev)

        def ld(logits):
            lq = logits[torch.arange(B, device=dev), qi].float()
            return (lq.gather(1, bad[:, None]) - lq.gather(1, good[:, None])).squeeze(1).cpu().numpy()

        # the clean residual after every block (index 0: the input to block 1)
        cache, hooks = {}, []
        hooks.append(gpt.blocks[0].register_forward_pre_hook(lambda m, inp: cache.__setitem__(0, inp[0].clone())))
        for li, blk in enumerate(gpt.blocks):
            hooks.append(blk.register_forward_hook(lambda m, inp, out, li=li: cache.__setitem__(li + 1, out.clone())))
        clean[b0:b0 + B] = ld(gpt(xc))
        for h in hooks:
            h.remove()
        corrupt[b0:b0 + B] = ld(gpt(xx))

        for level in range(L + 1):
            for gi, g in enumerate(GROUPS):
                mask = torch.zeros(B, W, 1, dtype=torch.bool, device=dev)
                for j, r in enumerate(batch):
                    mask[j, :len(r["lab"]), 0] = torch.tensor(r["lab"] == g, device=dev)
                src = cache[level]
                if level == 0:
                    h = gpt.blocks[0].register_forward_pre_hook(lambda m, inp: (torch.where(mask, src, inp[0]),))
                else:
                    h = gpt.blocks[level - 1].register_forward_hook(lambda m, inp, out: torch.where(mask, src, out))
                patched[level, gi, b0:b0 + B] = ld(gpt(xx))
                h.remove()
    return clean, corrupt, patched


def main(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    items = json.load(open(ih.SETS / "as_probed.json"))["items"]
    rows = pairs(items)
    vil = np.array([r["villain"] for r in rows])
    counts = {g: float(np.mean([np.sum(r["lab"] == g) for r in rows])) for g in GROUPS}
    print("mean positions per group:", {g: round(c, 1) for g, c in counts.items()}, flush=True)
    for short in runs:
        run_id = f"v4-ladder-C-{short}"
        dest = OUT / f"{run_id}.json"
        if dest.exists():
            continue
        f = glob.glob(f"hub-v4/box-*/logs/v4/eval/ladder/{run_id}.json")[0]
        box = re.search(r"(box-\d)", f).group(1)
        ck = json.load(open(f))["_meta"]["checkpoints"]["base"]
        gpt = model_from_checkpoint(hf_hub_download(ih.REPO, ck, revision=ih.REVISION[box]), pick_device("auto")).eval()
        clean, corrupt, patched = run(gpt, rows)
        restored = []
        for grp in (vil, ~vil):
            gap = clean[grp].mean() - corrupt[grp].mean()
            restored.append((patched[:, :, grp].mean(-1) - corrupt[grp].mean()) / gap)
        rec = {"run": run_id, "checkpoint": ck, "git_commit": git_commit(), "groups": GROUPS, "n_pairs": len(rows),
               "positions_per_group": counts,
               "clean_minus_corrupt": {"villain": float(clean[vil].mean() - corrupt[vil].mean()),
                                        "other": float(clean[~vil].mean() - corrupt[~vil].mean())},
               "restored": ((restored[0] + restored[1]) / 2).tolist(),
               "restored_villain": restored[0].tolist(), "restored_other": restored[1].tolist(),
               "raw_patched_minus_corrupt_villain": (patched[:, :, vil].mean(-1) - corrupt[vil].mean()).tolist()}
        dest.write_text(json.dumps(rec))
        r = np.array(rec["restored"])
        print(f"{run_id} gap V {rec['clean_minus_corrupt']['villain']:+.2f} O {rec['clean_minus_corrupt']['other']:+.2f} | "
              + " ".join(f"{g}@{np.argmax(r[:, gi])}:{r[:, gi].max():.2f}" for gi, g in enumerate(GROUPS)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    a = ap.parse_args()
    main(a.runs)
