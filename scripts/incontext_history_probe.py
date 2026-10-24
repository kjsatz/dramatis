# ruff: noqa: E501
"""Is "this child did bad things" written on a child's name tokens? A history separation on every
ladder-C model, switched on or not (C2 mechanism, exploratory; Kevin, 2026-09-25).

The transplant arms of incontext_evidence.py give the same story, the same names and the same order
twice: the villain's two act scenes both bad (T_BB) or both good (T_GG); nothing else differs. At a
child's name tokens, after every block, does the residual stream separate the two versions?

  mentions  the child's name tokens in the story from the first changed token (the villain's first act
            scene) to the test setup, averaged per story (inside the changed scenes the name's own
            neighbours differ, so this includes plain local context)
  far       the same, only mentions in text that is identical in the two versions (e.g. "Jake walked
            along the beach" after the act): any difference there arrived by attention from the
            changed scenes; "character carried to later mentions"
  setup     the actor's name in the test setup (the lookup head's query), averaged per story over the
            two test forks

Per child (the villain, whose history changed; the other child, whose did not), a cross-validated d':
the BB-minus-GG mean direction fitted on half the stories, d' of BB against GG on the other half (then
the reverse, averaged). Reported: villain d', other d', and their difference (a story-wide mood moves
both; a child-specific history moves only the villain's). No child-order confound: order and names are
the same in both versions.

    uv run python scripts/incontext_history_probe.py            (all 40 ladder-C models)
    uv run python scripts/incontext_history_probe.py --k2-only  (the 66 stories of Figure 8a's k = 2, to history_probe_66)
    uv run python scripts/incontext_history_probe.py --same-items  (Figure 8d: 8a's own k = 2 items, the story's scenes
        against incontext_same_items.py's good version, with a permutation null; to history_probe_same)

The permutation null (--same-items): per story the two versions' labels are swapped at random (the same swap for both
children's items), 200 times; for each block the null's mean and 95th percentile of villain d' minus other d'.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import difflib

import numpy as np
import torch

sys.path.insert(0, "scripts")
import incontext_heads as ih  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from dramatis.device import pick_device  # noqa: E402
from dramatis.evals import encode_story_start  # noqa: E402
from dramatis.probe_names import mentions  # noqa: E402
from dramatis.provenance import git_commit  # noqa: E402
from dramatis.train import model_from_checkpoint  # noqa: E402

OUT = Path("logs/v4/incontext/history_probe")
OUT_K2 = Path("logs/v4/incontext/history_probe_66")  # --k2-only (Kevin, 2026-10-24)
OUT_SAME = Path("logs/v4/incontext/history_probe_same")  # --same-items (Kevin, 2026-10-24)
SAME_SETS = Path("logs/v4/incontext/same_items/sets")
N_PERM = 200
SIZES = ["1M", "3M", "10M", "30M", "100M"]
tok = ih.tok


def name_of(item):
    return re.match(r"\s*([A-Z][a-z]+)", item["a"][0]).group(1)  # endings start with the actor


def k2_stories():
    """The test stories whose villain's first two choices are both bad acts (a villain's extra scene can be a held-out
    temptation they resist): Figure 8a's k = 2 and 8c, 66 of the 80."""
    import build_probe_sets as bps

    keep = set()
    for r in bps.v4_rows("v4-one", 80, generalist=True, seed=0):
        kid = r["spec"]["kids"][0 if r["spec"]["villain"] == "$A" else 1]
        choices = [a in bps.acts.bad for a in kid["acts"] if a in bps.acts.bad or bps.sit.of_act(a)]  # neutral acts: no choice
        if all(choices[:2]):
            keep.add(r["id"])
    return keep


def build_rows(same_items=False):
    """Per story: for BB and GG, the token ids of each item and the positions to read. same_items: BB = Figure 8a's own k = 2
    items (the villain's two bad acts as written), GG = the same items with those two scenes made good."""
    if same_items:
        sets = {"BB": json.load(open(SAME_SETS / "own.json"))["items"], "GG": json.load(open(SAME_SETS / "GG.json"))["items"]}
    else:
        sets = {c: json.load(open(ih.SETS / f"T_{c}.json"))["items"] for c in ("BB", "GG")}
    assert [i["story"] for i in sets["BB"]] == [i["story"] for i in sets["GG"]]
    stories = {}
    for i, (bb, gg) in enumerate(zip(sets["BB"], sets["GG"])):
        pair = sets["BB"][i + 1] if bb["group"] == "villain" else sets["BB"][i - 1]
        vname, oname = (name_of(bb), name_of(pair)) if bb["group"] == "villain" else (name_of(pair), name_of(bb))
        ids = {c: encode_story_start(tok, it["prompt"]) for c, it in (("BB", bb), ("GG", gg))}
        if max(len(x) for x in ids.values()) > 512:
            continue
        first = next(j for j in range(min(len(ids["BB"]), len(ids["GG"]))) if ids["BB"][j] != ids["GG"][j])
        same = {"BB": set(), "GG": set()}  # positions in text identical in the two versions
        for blk in difflib.SequenceMatcher(None, ids["BB"], ids["GG"], autojunk=False).get_matching_blocks():
            if blk.size >= 8:  # whole unchanged stretches, not stray shared tokens
                same["BB"] |= set(range(blk.a + 2, blk.a + blk.size))  # skip the first two: their context changed
                same["GG"] |= set(range(blk.b + 2, blk.b + blk.size))
        s = stories.setdefault(bb["story"], {"vname": vname, "oname": oname, "items": []})
        s["items"].append({"villain": bb["group"] == "villain", "ids": ids, "first": first, "same": same,
                           "actor": vname if bb["group"] == "villain" else oname})
    return stories


def positions(ids, name, first, setup_line):
    """Mentions of `name` from token `first` to the setup line / in the setup line."""
    n_before_setup = len(encode_story_start(tok, setup_line[0]))  # tokens before the setup line
    ms = [p for m in mentions(tok, ids, name) for p in m]
    return [p for p in ms if first <= p < n_before_setup], [p for p in ms if p >= n_before_setup]


@torch.no_grad()
def states(gpt, ids):
    cache = {}
    hooks = [gpt.blocks[0].register_forward_pre_hook(lambda m, inp: cache.__setitem__(0, inp[0][0].float().cpu()))]
    for li, blk in enumerate(gpt.blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, li=li: cache.__setitem__(li + 1, o[0].float().cpu())))
    gpt(torch.tensor([ids], device=gpt.device))
    for h in hooks:
        h.remove()
    return torch.stack([cache[k] for k in sorted(cache)]).numpy()  # (L + 1, T, D)


def dprime(A, B):
    """A, B: (n_stories, D) paired by story; two-fold d' of A against B along the fitted mean difference."""
    n = len(A)
    half = np.arange(n) % 2 == 0
    out = []
    for fit, test in ((half, ~half), (~half, half)):
        u = A[fit].mean(0) - B[fit].mean(0)
        u /= np.linalg.norm(u) + 1e-9
        a, b = A[test] @ u, B[test] @ u
        out.append((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-9))
    return float(np.mean(out))


def null_dprime(A, B, A2, B2, n_perm=N_PERM, seed=0):
    """Permutation null per block: each story's two versions swapped at random (one swap per story, shared by the villain's
    features A, B and the other child's A2, B2). Returns {"v", "o", "diff"}: the 95th percentile, per block, of the villain's
    d', the other child's, and the villain's minus the other's; and "diff_mean", the difference's mean."""
    rng = np.random.default_rng(seed)
    v, o = [], []
    for _ in range(n_perm):
        flip = rng.random(len(A)) < 0.5
        f = flip[:, None, None]
        Ap, Bp = np.where(f, B, A), np.where(f, A, B)
        Ap2, Bp2 = np.where(f, B2, A2), np.where(f, A2, B2)
        v.append([dprime(Ap[:, L], Bp[:, L]) for L in range(A.shape[1])])
        o.append([dprime(Ap2[:, L], Bp2[:, L]) for L in range(A.shape[1])])
    v, o = np.array(v), np.array(o)
    p95 = lambda x: np.percentile(x, 95, axis=0).tolist()  # noqa: E731
    return {"v": p95(v), "o": p95(o), "diff": p95(v - o), "diff_mean": (v - o).mean(0).tolist()}


def main(runs, k2_only=False, same_items=False):
    out = OUT_SAME if same_items else OUT_K2 if k2_only else OUT
    out.mkdir(parents=True, exist_ok=True)
    stories = build_rows(same_items)
    items_by_story = {s: v for s, v in stories.items() if any(it["villain"] for it in v["items"]) and any(not it["villain"] for it in v["items"])}
    if k2_only:
        keep = k2_stories()
        items_by_story = {s: v for s, v in items_by_story.items() if s in keep}
    print(f"stories {len(items_by_story)}", flush=True)
    for run_id in runs:
        dest = out / f"{run_id}.json"
        if dest.exists():
            continue
        f = glob.glob(f"hub-v4/box-*/logs/v4/eval/ladder/{run_id}.json")[0]
        box = re.search(r"(box-\d)", f).group(1)
        ev = json.load(open(f))
        ck = ev["_meta"]["checkpoints"]["base"]
        gpt = model_from_checkpoint(hf_hub_download(ih.REPO, ck, revision=ih.REVISION[box]), pick_device("auto")).eval()
        feats = {k: {"BB": [], "GG": []} for k in ("v_mentions", "o_mentions", "v_far", "o_far", "v_setup", "o_setup")}
        for s in items_by_story.values():
            acc = {k: {"BB": [], "GG": []} for k in feats}
            for it in s["items"]:
                for c in ("BB", "GG"):
                    ids = it["ids"][c]
                    text = tok.decode(ids[1:])
                    setup_line = (text[: text.rfind("\n")],)
                    X = states(gpt, ids)
                    vm, _ = positions(ids, s["vname"], it["first"], setup_line)
                    om, _ = positions(ids, s["oname"], it["first"], setup_line)
                    _, sp = positions(ids, it["actor"], it["first"], setup_line)
                    if vm:
                        acc["v_mentions"][c].append(X[:, vm].mean(1))
                    if om:
                        acc["o_mentions"][c].append(X[:, om].mean(1))
                    vf = [p for p in vm if p in it["same"][c]]
                    of = [p for p in om if p in it["same"][c]]
                    if vf:
                        acc["v_far"][c].append(X[:, vf].mean(1))
                    if of:
                        acc["o_far"][c].append(X[:, of].mean(1))
                    if sp:
                        acc["v_setup" if it["villain"] else "o_setup"][c].append(X[:, sp].mean(1))
            for k in feats:
                if acc[k]["BB"] and acc[k]["GG"]:
                    for c in ("BB", "GG"):
                        feats[k][c].append(np.mean(acc[k][c], 0))
        rec = {"run": run_id, "checkpoint": ck, "git_commit": git_commit(), "stories": sorted(items_by_story),
               "rungC": ev["incontext_full_v4"]["summary"]["incontext_full/character/ALL"]}
        for k in feats:
            A, B = np.stack(feats[k]["BB"]), np.stack(feats[k]["GG"])  # (n_stories, L + 1, D)
            rec[k] = [dprime(A[:, L], B[:, L]) for L in range(A.shape[1])]
            rec[f"n_{k}"] = len(A)
        if same_items and len(feats["v_setup"]["BB"]) == len(feats["o_setup"]["BB"]):  # one row per story for both children
            null = null_dprime(*(np.stack(feats[k][c]) for k in ("v_setup", "o_setup") for c in ("BB", "GG")))
            rec["null_setup_mean"], rec["null_setup_p95"] = null["diff_mean"], null["diff"]
            rec["null_v_setup_p95"], rec["null_o_setup_p95"] = null["v"], null["o"]  # each child's own chance level
        dest.write_text(json.dumps(rec))
        dm = np.array(rec["v_mentions"]) - np.array(rec["o_mentions"])
        df = np.array(rec["v_far"]) - np.array(rec["o_far"])
        ds = np.array(rec["v_setup"]) - np.array(rec["o_setup"])
        print(f"{run_id} C={rec['rungC']:.2f} | mentions V-O max {dm.max():.1f} @ {dm.argmax()} | far V-O max {df.max():.1f} @ {df.argmax()} (n {rec['n_v_far']}) | setup V-O max {ds.max():.1f} @ {ds.argmax()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[f"v4-ladder-C-{s}-s{i}" for s in SIZES for i in range(8)])
    ap.add_argument("--k2-only", action="store_true", help="only the 66 stories whose villain's first two choices are bad")
    ap.add_argument("--same-items", action="store_true", help="Figure 8a's own k = 2 items against their good version, with a null")
    a = ap.parse_args()
    main(a.runs, a.k2_only, a.same_items)
