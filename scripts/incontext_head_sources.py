# ruff: noqa: E501
"""Where do the heads that write the in-context character read from? (C2 mechanism, exploratory;
Kevin, 2026-09-25.) For each model's top heads by direct logit attribution (incontext_heads.py), the
head's write at the decision token onto E[bad] - E[good] (final layer norm's scale frozen) is split
by source position: each position's attention weight times its value through the head's output
projection. Sources, on the k = 1 items (`as_probed`):

  v_act      the villain's bad-act line(s) in the story so far (the line after its [CHOICE])
  o_acts     the other child's temptation outcome lines (good choices)
  v_names    other mentions of the villain's name in the story so far
  o_names    other mentions of the other child's name
  setup_names  the children's names in the appended fork setup (the test temptation)
  setup      the rest of the fork setup
  ending     the ending so far (the actor's name on its new line, up to the decision token)
  bos        <|endoftext|>
  rest       everything else in the story so far

Reported per source: attention mass and the write (nats at the decision token), villain-actor items
minus other-actor items, and the same for each group alone.

    uv run python scripts/incontext_head_sources.py --sizes 10M 30M 100M --top 3
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

from dramatis.device import pick_device  # noqa: E402
from dramatis.probe_names import _piece, mentions  # noqa: E402
from dramatis.provenance import git_commit  # noqa: E402
from dramatis.train import model_from_checkpoint  # noqa: E402

OUT = Path("logs/v4/incontext/head_sources")
CATS = ["v_act", "o_acts", "v_names", "o_names", "setup_names", "setup", "ending", "bos", "rest"]
tok = ih.tok


def outcome_lines(story, kid):
    """The line after each of the kid's [CHOICE] lines (its outcome), with placeholders."""
    lines = story.split("\n")
    return [lines[i + 1] for i, ln in enumerate(lines[:-1]) if re.match(rf"\s*\[CHOICE {re.escape(kid)}", ln)]


def char_spans(ids):
    spans, pos = [], 0
    for t in ids:
        n = len(_piece(tok, t))
        spans.append((pos, pos + n))
        pos += n
    return spans, "".join(_piece(tok, t) for t in ids)


def positions_of(text_spans, text, needle, limit):
    """Token positions covering `needle` in the decoded text, before char `limit`."""
    out = []
    for m in re.finditer(re.escape(needle), text[:limit]):
        out += [i for i, (s, e) in enumerate(text_spans) if s < m.end() and e > m.start()]
    return out


def source_rows(items):
    """ih.decision_rows plus, per item, a category label for every position."""
    raw = {r["id"]: r for r in bps.v4_rows("v4-one", 80, generalist=True, seed=0)}
    rows = ih.decision_rows(items)
    missing = 0
    for i, (it, row) in enumerate(zip(items, rows)):
        r = raw[it["story"]]
        v = r["spec"]["villain"]
        o = "$B" if v == "$A" else "$A"
        pair = items[i + 1] if it["group"] == "villain" else items[i - 1]
        actor, other = (re.match(r"\s*([A-Z][a-z]+)", x["a"][0]).group(1) for x in (it, pair))
        vname, oname = (actor, other) if it["group"] == "villain" else (other, actor)
        idx = bps.choices(r["story"], v)
        k1 = r["story"][: idx[1]]
        ids = row["ids"]
        spans, text = char_spans(ids)
        pre_end = text.rfind("\n", 0, text.rfind("\n"))  # the prompt's last line is the setup, then the ending's "\n"
        setup_end = text.rfind("\n")
        cat = np.full(len(ids), "rest", dtype=object)
        cat[0] = "bos"
        for p, (s, e) in enumerate(spans):
            if s >= setup_end:
                cat[p] = "ending"
            elif s >= pre_end:
                cat[p] = "setup"
        for kid, name, label in ((v, vname, "v_act"), (o, oname, "o_acts")):
            for line in outcome_lines(k1, kid):
                line = line.strip().replace(v, vname).replace(o, oname)
                ps = positions_of(spans, text, line, pre_end)
                missing += not ps
                cat[[p for p in ps if cat[p] == "rest"]] = label
        for name, label in ((vname, "v_names"), (oname, "o_names")):
            for m in mentions(tok, ids, name):
                cat[[p for p in m if cat[p] == "rest"]] = label
                cat[[p for p in m if cat[p] == "setup"]] = "setup_names"
        row["cat"] = cat
    print(f"outcome lines not found: {missing}", flush=True)
    return rows


@torch.no_grad()
def head_sources(gpt, rows, heads, batch_size=16):
    """(n_items, n_heads, len(CATS), 2): attention mass and write (nats) per source category."""
    H = gpt.config.n_head
    d = gpt.config.d_model // H
    E = gpt.token_embedding.weight.float()
    g = gpt.norm_final.weight.float()
    out = np.zeros((len(rows), len(heads), len(CATS), 2))
    for b0 in range(0, len(rows), batch_size):
        batch = rows[b0:b0 + batch_size]
        width = max(len(r["ids"]) for r in batch)
        x = torch.full((len(batch), width), tok.pad_id, dtype=torch.long)
        for j, r in enumerate(batch):
            x[j, :len(r["ids"])] = torch.tensor(r["ids"])
        qi = torch.tensor([r["q"] for r in batch])
        direction = torch.stack([E[r["bad"]] - E[r["good"]] for r in batch]) * g  # (B, D)
        direction = direction - direction.mean(-1, keepdim=True)  # the layer norm centres
        per_pos = {}
        hooks = []
        for li in sorted({li for li, _ in heads}):
            def pre(module, inputs, li=li):
                xx = inputs[0]
                B, T, _ = xx.shape
                q = module.query(xx).view(B, T, H, d)
                k = module.key(xx).view(B, T, H, d)
                v = module.value(xx).view(B, T, H, d)
                s = torch.einsum("bthd,bshd->bhts", q, k) / d ** 0.5
                w = torch.softmax(s.masked_fill(~module.mask[:T, :T], float("-inf")), -1)
                wq = w[torch.arange(B), :, qi.to(xx.device), :]  # (B, H, T)
                Wo = module.out_proj.weight
                for h in [hh for l2, hh in heads if l2 == li]:
                    vo = v[:, :, h].float() @ Wo[:, h * d:(h + 1) * d].T.float()  # (B, T, D) each position's value, written out
                    proj = (vo * direction[:, None].to(vo.device)).sum(-1)  # (B, T), before the scale
                    per_pos[(li, h)] = (wq[:, h].float().cpu(), (wq[:, h].float() * proj).cpu())
            hooks.append(gpt.blocks[li].attention.register_forward_pre_hook(pre))
        resid = {}
        hooks.append(gpt.norm_final.register_forward_pre_hook(lambda m, inp: resid.__setitem__("x", inp[0])))
        gpt(x.to(gpt.device))
        for hk in hooks:
            hk.remove()
        xq = resid["x"][torch.arange(len(batch)), qi.to(gpt.device)].float()
        scale = (xq.var(-1, unbiased=False) + 1e-5).sqrt().cpu()
        for hi, key in enumerate(heads):
            att, wr = per_pos[key]
            for j, r in enumerate(batch):
                n = len(r["ids"])
                for ci, c in enumerate(CATS):
                    m = torch.tensor(r["cat"] == c)
                    out[b0 + j, hi, ci, 0] = float(att[j, :n][m].sum())
                    out[b0 + j, hi, ci, 1] = float(wr[j, :n][m].sum() / scale[j])
    return out


def main(sizes, seeds, top):
    OUT.mkdir(parents=True, exist_ok=True)
    items = json.load(open(ih.SETS / "as_probed.json"))["items"]
    rows = source_rows(items)
    villain = np.array([r["villain"] for r in rows])
    for size in sizes:
        for seed in seeds:
            run = f"v4-ladder-C-{size}-s{seed}"
            dest = OUT / f"{run}.json"
            if dest.exists():
                continue
            hs = json.load(open(ih.OUT / f"{run}.json"))
            dla = np.array(hs["dla"])
            heads = [tuple(int(v) for v in np.unravel_index(i, dla.shape)) for i in np.argsort(dla.ravel())[::-1][:top]]
            f = glob.glob(f"hub-v4/box-*/logs/v4/eval/ladder/{run}.json")[0]
            box = re.search(r"(box-\d)", f).group(1)
            gpt = model_from_checkpoint(hf_hub_download(ih.REPO, hs["checkpoint"], revision=ih.REVISION[box]), pick_device("auto")).eval()
            src = head_sources(gpt, rows, heads)
            rec = {"run": run, "rungC": hs["rungC"], "git_commit": git_commit(), "cats": CATS, "heads": []}
            for hi, (li, h) in enumerate(heads):
                s = src[:, hi]
                rec["heads"].append({"block": li, "head": h, "dla": float(dla[li, h]),
                                     "attn_villain": s[villain, :, 0].mean(0).tolist(), "attn_other": s[~villain, :, 0].mean(0).tolist(),
                                     "write_villain": s[villain, :, 1].mean(0).tolist(), "write_other": s[~villain, :, 1].mean(0).tolist()})
                wd = s[villain, :, 1].mean(0) - s[~villain, :, 1].mean(0)
                print(f"{run} C={hs['rungC']:.2f} L{li}H{h} dla={dla[li, h]:.2f} write V-O by source: "
                      + " ".join(f"{c}={w:+.2f}" for c, w in zip(CATS, wd))
                      + " | attn(V items): " + " ".join(f"{c}={a:.2f}" for c, a in zip(CATS, s[villain, :, 0].mean(0))), flush=True)
            dest.write_text(json.dumps(rec))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["10M", "30M", "100M"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    main(a.sizes, a.seeds, a.top)
