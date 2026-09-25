# ruff: noqa: E501
"""What evidence does the in-context character (C2, rungs C / D) come from? Exploratory probes on
the final ladder-C checkpoints (Kevin, 2026-09-25; the frozen 3a readouts are unchanged).

Two sets of arms, each built from the same 80 held-out generalist v4-one stories as the frozen
`incontext_full_v4` sets, with one draw of names and test forks per story shared by every arm (so
arms differ only in the prefix):

  source (the k = 1 cut, before the villain's second *bad* choice; the frozen cut counts every choice,
  so in 4 of the 80 stories its k = 1 prefix has no bad act, and in 14 its k = 2 prefix has one):
    as_probed    the prefix as is: the villain's first bad act and the other child's good choices
    no_other     the other child's temptation paragraphs removed (only the villain's bad act left)
    no_villain   the villain's bad-act paragraph removed (only the other child's good choices left)
    neither      both removed
  mixed (the k = 2 cut, after the villain's second bad act): the villain's two act
  paragraphs replaced by fork paragraphs of the same situations (the fork's setup and its bad or
  good ending, so bad and good versions share their setup text):
    BB, BG, GB, GG   first / second act bad (B) or good (G);  as_probed_k2: the original paragraphs
  The fork paragraphs turned out to carry almost no evidence either way (their endings are minimal
  pairs; the corpus marks a bad act: "... until $B cried", "... and hid it", "I win!"), so:
    T_BB, T_BG, T_GB, T_GG   the villain's two act paragraphs replaced by corpus paragraphs of the
                 same situations, bad or good, transplanted from other held-out v4 stories with the
                 same adult (placeholders swapped so the donor's actor is the villain), as DATA.md §18

Test items as frozen: a fork of each performed situation the prefix's acts did not use, with the
villain and with the other child as actor; character = villain - other (nats).

    uv run python scripts/incontext_evidence.py build
    uv run python scripts/incontext_evidence.py score --sizes 10M 30M 100M
"""

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import build_probe_sets as bps  # noqa: E402  (the frozen builders: rows, forks, assembly)

OUT = Path("logs/v4/incontext")
REPO = "kjsatz/dramatis-v4-runs"
REVISION = {"box-1": None, "box-4": None, "box-2": "box-2", "box-3": "box-3"}  # the Hub layout
SOURCE_ARMS = ["as_probed", "no_other", "no_villain", "neither"]
MIXED_ARMS = ["as_probed_k2", "BB", "BG", "GB", "GG"]
TRANSPLANT_ARMS = ["T_BB", "T_BG", "T_GB", "T_GG"]


def donor_paragraphs():
    """(situation, "B" | "G", adult) -> scene paragraphs of held-out v4 stories: one child's
    temptation of that situation with its bad or good outcome, the actor as "@X", the other "@Y"."""
    out = {}
    for batch in ("v4-one", "v4-both", "v4-base"):
        for r in bps.v4_rows(batch, 10**6, seed=0):
            for p in r["story"].split("\n\n"):
                m = re.findall(r"\[AFTERMATH (\$[AB]) ([a-z_]+)\]", p)
                if len(m) != 1 or len(bps.choices(p, m[0][0])) != 1:
                    continue
                kid, act = m[0]
                sid = bps.sit.of_act(act) if act in bps.acts.bad or act in {bps.sit.by_id[x]["good"] for x in bps.sit.performed} else None
                if sid not in bps.sit.performed:
                    continue
                other = "$B" if kid == "$A" else "$A"
                text = p.replace(kid, "@X").replace(other, "@Y")
                out.setdefault((sid, "B" if act in bps.acts.bad else "G", r["spec"]["adult"]), []).append((r["id"], text))
    return out


def paragraphs_with_choice(paras, kid):
    """Indices of the paragraphs holding a [CHOICE kid] line (the kid's temptations), in order."""
    return [i for i, p in enumerate(paras) if re.search(rf"\[CHOICE {re.escape(kid)}(?: [a-z_]+)?\]", p)]


def fork_paragraph(f, v, o, adult, bad):
    """A villain act rendered from a fork item: its setup, then the bad or the good ending."""
    b, g = f.endings(v, o)
    return f.setup_text(v, o, adult) + (b if bad else g)


def test_items(r, prefix_by_arm, exclude, names, forks_for, v):
    """Fork items appended to each arm's prefix: the same forks (one per situation not in
    `exclude`) and names in every arm; villain and other child as actor."""
    o = "$B" if v == "$A" else "$A"
    adult = r["spec"]["adult"]
    out = {arm: [] for arm in prefix_by_arm}
    for sid in bps.sit.performed:
        if sid in exclude:
            continue
        f = forks_for[sid]
        for actor, group in ((v, "villain"), (o, "other")):
            other = o if actor == v else v
            setup = f.setup_text(actor, other, adult)
            bad, good = f.endings(actor, other)
            for arm, raw in prefix_by_arm.items():
                text, _ = bps.assemble_story(raw + "\n" + setup, r["id"], bps.acts.valence, bps.banks, "off",
                                             bps.pool, 0, names=names)
                show = lambda t: t.replace("$A", names["$A"]).replace("$B", names["$B"])  # noqa: E731
                out[arm].append({"prompt": text, "a": [show(bad)], "b": [show(good)], "group": group,
                                 "act": f.act, "story": r["id"]})
    return out


def build(n=80, seed=0, only=None):
    rng = np.random.default_rng(seed)
    drng = np.random.default_rng(seed + 1)  # the donors' own stream: the arms above are unchanged
    donors = donor_paragraphs()
    source = {a: [] for a in SOURCE_ARMS}
    mixed = {a: [] for a in MIXED_ARMS + TRANSPLANT_ARMS}
    for r in bps.v4_rows("v4-one", n, generalist=True, seed=seed):
        v = r["spec"]["villain"]
        o = "$B" if v == "$A" else "$A"
        story = r["story"]
        paras = story.split("\n\n")
        # the villain's bad-act paragraphs (its extra scene may be a good [CHOICE] of a held-out situation)
        vb = [i for i, p in enumerate(paras) if any(a in bps.acts.bad for a in re.findall(rf"\[AFTERMATH {re.escape(v)} ([a-z_]+)\]", p))]
        if len(vb) < 2:
            continue
        a, b = rng.choice(bps.pool, size=2, replace=False)
        names = {"$A": str(a), "$B": str(b)}
        forks_for = {}
        for sid in bps.sit.performed:  # one fork per situation, shared by every arm and slot
            fs = bps.FORKS["set"].by_act(bps.bad_of[sid])
            forks_for[sid] = fs[int(rng.integers(len(fs)))]
        kid = r["spec"]["kids"][0 if v == "$A" else 1]
        bad_sits = [s for a_, s in zip(kid["acts"], kid["situations"]) if a_ in bps.acts.bad]

        # source arms: cut before the villain's second bad choice (its setup kept, as frozen), paragraphs removed
        second = paras[vb[1]]
        cut = re.search(rf"^[ \t]*\[CHOICE {re.escape(v)}", second, re.MULTILINE).start()
        k1 = paras[: vb[1]] + [second[:cut].rstrip()]
        oi = paragraphs_with_choice(k1, o)
        drop = {"as_probed": set(), "no_other": set(oi), "no_villain": {vb[0]}, "neither": set(oi) | {vb[0]}}
        prefixes = {arm: "\n\n".join(p for i, p in enumerate(k1) if i not in d) for arm, d in drop.items()}
        for arm, its in test_items(r, prefixes, {bad_sits[0]}, names, forks_for, v).items():
            source[arm] += its

        # mixed arms: cut after the villain's second bad act, its two act paragraphs replaced
        paras = paras[: vb[1] + 1]
        slots = vb[:2]
        prefixes = {"as_probed_k2": "\n\n".join(paras)}
        for arm in ("BB", "BG", "GB", "GG"):
            ps = list(paras)
            for j, i in enumerate(slots):
                ps[i] = fork_paragraph(forks_for[bad_sits[j]], v, o, r["spec"]["adult"], arm[j] == "B")
            prefixes[arm] = "\n\n".join(ps)
        picks = {}
        for j in range(2):  # one bad and one good donor per slot, shared by the T_ arms
            for val in "BG":
                pool = [d for d in donors.get((bad_sits[j], val, r["spec"]["adult"]), []) if d[0] != r["id"]]
                picks[j, val] = pool[int(drng.integers(len(pool)))][1].replace("@X", v).replace("@Y", o)
        for arm in TRANSPLANT_ARMS:
            ps = list(paras)
            for j, i in enumerate(slots):
                ps[i] = picks[j, arm[2 + j]]
            prefixes[arm] = "\n\n".join(ps)
        for arm, its in test_items(r, prefixes, set(bad_sits[:2]), names, forks_for, v).items():
            mixed[arm] += its

    (OUT / "sets").mkdir(parents=True, exist_ok=True)
    meta = {"script": "scripts/incontext_evidence.py", "stories": n, "seed": seed, **bps.FORKS["set"].meta,
            "git_commit": bps.git_commit(), "time": bps.utc_now()}
    for arms in (source, mixed):
        for arm, items in arms.items():
            if only is not None and arm not in only:
                continue
            spec = {"name": arm, "items": items, "contrasts": {"character": ["villain", "other"]}, "meta": meta}
            (OUT / "sets" / f"{arm}.json").write_text(json.dumps(spec, indent=1))
            print(f"{arm}: {len(items)} items")


def score(sizes, seeds, arms=None):
    from huggingface_hub import hf_hub_download

    from dramatis.device import pick_device
    from dramatis.light_probes import ProbeSet
    from dramatis.provenance import git_commit
    from dramatis.tokenizer import StoryTokenizer
    from dramatis.train import model_from_checkpoint

    tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
    arms = arms or SOURCE_ARMS + MIXED_ARMS
    paths = {arm: OUT / "sets" / f"{arm}.json" for arm in arms}
    sets = {arm: ProbeSet(p, tok, 512) for arm, p in paths.items()}
    (OUT / "evidence").mkdir(parents=True, exist_ok=True)
    for size in sizes:
        for seed in seeds:
            run = f"v4-ladder-C-{size}-s{seed}"
            dest = OUT / "evidence" / f"{run}.json"
            rec = json.loads(dest.read_text()) if dest.exists() else {"_meta": {"sets": {}}}
            if all(a in rec for a in arms):
                continue
            f = glob.glob(f"hub-v4/box-*/logs/v4/eval/ladder/{run}.json")[0]
            box = re.search(r"(box-\d)", f).group(1)
            ck = json.load(open(f))["_meta"]["checkpoints"]["base"]
            gpt = model_from_checkpoint(hf_hub_download(REPO, ck, revision=REVISION[box]), pick_device("auto")).eval()
            rec["_meta"].update({"run": run, "checkpoint": ck, "revision": REVISION[box], "git_commit": git_commit()})
            rec["_meta"]["sets"].update({a: hashlib.sha256(paths[a].read_bytes()).hexdigest() for a in arms})
            for arm, s in sets.items():
                sc = s.score(gpt)
                rec[arm] = {"summary": s.summarize(sc), "scores": sc}
            dest.write_text(json.dumps(rec))
            print(run, {a: rec[a]["summary"][f"{a}/character/ALL"] for a in arms}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--arms", nargs="+", help="build / score only these arms (default: all but the T_ arms)")
    ap.add_argument("--sizes", nargs="+", default=["10M", "30M", "100M"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    a = ap.parse_args()
    build(only=a.arms) if a.cmd == "build" else score(a.sizes, a.seeds, a.arms)
